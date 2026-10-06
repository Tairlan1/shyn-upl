#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_verifier.py
=====================================================================
Обучает НОВЫЙ open-set верификатор авторства (author_verification.py) -
StyleEncoder + PairwiseVerifier - и калибрует пороги MATCH/MISMATCH.

В ОТЛИЧИЕ ОТ train_model.py:
  - здесь НЕТ "5 классов" и LabelEncoder на авторов;
  - обучающие примеры - это ПАРЫ (текст, профиль) -> same/different, а
    не (текст) -> author_id;
  - один автор (по умолчанию MarkTwain - см. OPEN_SET_AUTHOR) ПОЛНОСТЬЮ
    исключён из обучения энкодера и верификатора и используется только
    в evaluate.py как "неизвестный автор" (п.5 задания, open-set test) -
    ровно тот же пример, что в самом тех.задании (train: Doyle/Poe/
    Wells/London, test: Twain).

СХЕМА РАЗБИЕНИЯ (без утечек, всё - на уровне КНИГИ, п.4 задания)
-------------------------------------------------------------------
Для 4 "известных" авторов книги разбиваются на 3 непересекающихся пула:

  TRAIN_POOL       (~60% книг)  - энкодер (char/word TF-IDF + стилометрия)
                                  обучается ИСКЛЮЧИТЕЛЬНО на чанках из
                                  этого пула; из него же строятся ВСЕ
                                  профили для обучения PairwiseVerifier
                                  (leave-ONE-BOOK-out внутри пула - см.
                                  _make_training_pairs).
  VALIDATION_POOL  (~20% книг)  - НИКОГДА не участвует в fit энкодера и
                                  НИКОГДА не входит ни в один профиль.
                                  Тексты отсюда используются только как
                                  new_text-запросы для КАЛИБРОВКИ порогов
                                  (п.7 задания: "рассчитывай threshold
                                  только на validation set").
  TEST_POOL        (~20% книг)  - тоже никогда не видится энкодером/
                                  верификатором/калибровкой. Используется
                                  ТОЛЬКО в evaluate.py для финального,
                                  честного отчёта.

MarkTwain целиком исключён из TRAIN/VALIDATION/TEST_POOL выше - это
отдельный, полностью "чужой" open-set автор для evaluate.py.

Запуск:
    python3 train_verifier.py --data data_processed/dataset.jsonl \
        --model-dir model_verifier
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

import author_verification as av
import config

OPEN_SET_AUTHOR = "MarkTwain"
RANDOM_SEED = 42

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
# остаток (~0.20) идёт в TEST_POOL

MIN_PROFILE_BOOKS = 2
MAX_PROFILE_BOOKS_TRAIN = 6  # разнообразие размера профиля уже на этапе обучения верификатора


def load_corpus_by_author_book(path: Path) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("scale", "large") != "large":
                continue  # используем только исходные ~1500-словные чанки, без multiscale-производных
            out[rec["author"]][rec["book"]].append(rec["text"])
    return out


def split_books(rng: random.Random, books: list[str]) -> tuple[list[str], list[str], list[str]]:
    books = list(books)
    rng.shuffle(books)
    n = len(books)
    n_train = max(2, round(n * TRAIN_FRACTION))
    n_val = max(1, round(n * VALIDATION_FRACTION))
    n_train = min(n_train, n - 2) if n > 2 else n_train  # оставить хоть что-то test/val
    train_books = books[:n_train]
    val_books = books[n_train:n_train + n_val]
    test_books = books[n_train + n_val:]
    if not test_books:  # совсем маленький корпус автора - берём хотя бы 1 книгу в test
        test_books = [train_books.pop()]
    if not val_books:
        val_books = [train_books.pop()]
    return train_books, val_books, test_books


def _make_training_pairs(rng: random.Random, encoder: av.StyleEncoder,
                          train_pool: dict[str, dict[str, list[str]]],
                          pairs_per_author: int = 60) -> tuple[list[dict], list[int]]:
    """Leave-ONE-BOOK-out генерация обучающих пар ИСКЛЮЧИТЕЛЬНО внутри
    TRAIN_POOL: профиль автора A строится из случайного подмножества ЕГО
    книг (кроме одной, отложенной), запрос берётся из этой отложенной
    книги - тот же чанк никогда не попадает одновременно в профиль и в
    запрос (п.4 задания)."""
    authors = list(train_pool.keys())
    feats_list: list[dict] = []
    labels: list[int] = []

    for author in authors:
        books = list(train_pool[author].keys())
        if len(books) < MIN_PROFILE_BOOKS + 1:
            print(f"  [verifier-train] {author}: слишком мало книг в TRAIN_POOL "
                  f"({len(books)}) для leave-one-book-out - пропущен.")
            continue
        for _ in range(pairs_per_author):
            held_out_book = rng.choice(books)
            other_books = [b for b in books if b != held_out_book]
            k = rng.randint(MIN_PROFILE_BOOKS, min(MAX_PROFILE_BOOKS_TRAIN, len(other_books)))
            profile_books = rng.sample(other_books, k)
            profile_texts = [t for b in profile_books for t in train_pool[author][b]]
            if len(profile_texts) > 20:  # не раздуваем один профиль сотнями чанков без нужды
                profile_texts = rng.sample(profile_texts, 20)
            profile = av.build_author_profile(profile_texts, encoder, candidate_author_id=author)

            query_text = rng.choice(train_pool[author][held_out_book])
            feats_list.append(av.similarity_features(query_text, profile, encoder))
            labels.append(1)

            # негативный пример: тот же query_text против профиля ДРУГОГО автора
            other_authors = [a for a in authors if a != author and len(train_pool[a]) >= MIN_PROFILE_BOOKS + 1]
            if not other_authors:
                continue
            neg_author = rng.choice(other_authors)
            neg_books = list(train_pool[neg_author].keys())
            k2 = rng.randint(MIN_PROFILE_BOOKS, min(MAX_PROFILE_BOOKS_TRAIN, len(neg_books)))
            neg_profile_books = rng.sample(neg_books, k2)
            neg_profile_texts = [t for b in neg_profile_books for t in train_pool[neg_author][b]]
            if len(neg_profile_texts) > 20:
                neg_profile_texts = rng.sample(neg_profile_texts, 20)
            neg_profile = av.build_author_profile(neg_profile_texts, encoder, candidate_author_id=neg_author)
            feats_list.append(av.similarity_features(query_text, neg_profile, encoder))
            labels.append(0)

    return feats_list, labels


def _make_validation_scores(rng: random.Random, encoder: av.StyleEncoder,
                             verifier: av.PairwiseVerifier | None,
                             train_pool: dict[str, dict[str, list[str]]],
                             query_pool: dict[str, dict[str, list[str]]],
                             queries_per_author: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Профили строятся из TRAIN_POOL (энкодер их уже видел - это нормально,
    он не классификатор с метками, см. README), а new_text-запросы берутся
    ИСКЛЮЧИТЕЛЬНО из query_pool (validation ИЛИ test - в зависимости от
    вызова), который энкодер и верификатор не видели никогда."""
    authors = [a for a in train_pool if a in query_pool]
    pos_scores, neg_scores = [], []

    profiles = {}
    for author in authors:
        books = list(train_pool[author].keys())
        texts = [t for b in books for t in train_pool[author][b]]
        if len(texts) > 40:
            texts = rng.sample(texts, 40)
        profiles[author] = av.build_author_profile(texts, encoder, candidate_author_id=author)

    def _score(query_text: str, profile: av.AuthorProfile) -> float:
        feats = av.similarity_features(query_text, profile, encoder)
        if verifier is not None:
            return verifier.predict_proba_same(feats)
        return av._heuristic_combine(feats)

    for author in authors:
        q_books = list(query_pool[author].keys())
        q_texts = [t for b in q_books for t in query_pool[author][b]]
        rng.shuffle(q_texts)
        q_texts = q_texts[:queries_per_author]

        for qt in q_texts:
            pos_scores.append(_score(qt, profiles[author]))

        other_authors = [a for a in authors if a != author]
        for qt in q_texts:
            neg_author = rng.choice(other_authors)
            neg_scores.append(_score(qt, profiles[neg_author]))

    return np.asarray(pos_scores), np.asarray(neg_scores)


def train(data_path: Path, model_dir: Path, seed: int = RANDOM_SEED) -> dict:
    rng = random.Random(seed)
    print(f"Загрузка корпуса: {data_path}")
    by_author_book = load_corpus_by_author_book(data_path)
    if OPEN_SET_AUTHOR not in by_author_book:
        print(f"ВНИМАНИЕ: open-set автор {OPEN_SET_AUTHOR} не найден в датасете - "
              f"open-set тест в evaluate.py будет недоступен.", file=sys.stderr)

    known_authors = [a for a in by_author_book if a != OPEN_SET_AUTHOR]
    print(f"Известные (обучающие) авторы: {known_authors}")
    print(f"Open-set (полностью исключённый) автор: {OPEN_SET_AUTHOR}")

    train_pool: dict[str, dict[str, list[str]]] = {}
    val_pool: dict[str, dict[str, list[str]]] = {}
    test_pool: dict[str, dict[str, list[str]]] = {}

    for author in known_authors:
        books = list(by_author_book[author].keys())
        tr, va, te = split_books(rng, books)
        train_pool[author] = {b: by_author_book[author][b] for b in tr}
        val_pool[author] = {b: by_author_book[author][b] for b in va}
        test_pool[author] = {b: by_author_book[author][b] for b in te}
        print(f"  {author}: {len(books)} книг -> train={len(tr)} val={len(va)} test={len(te)}")

    # ---- StyleEncoder: обучается ТОЛЬКО на TRAIN_POOL ----
    train_texts = [t for a in train_pool for b in train_pool[a] for t in train_pool[a][b]]
    print(f"\nОбучение StyleEncoder на {len(train_texts)} чанках TRAIN_POOL...")
    t0 = time.time()
    encoder = av.StyleEncoder()
    encoder.fit(train_texts)
    print(f"  готово за {time.time() - t0:.1f}s")

    # ---- Обучающие пары для PairwiseVerifier (leave-one-book-out внутри TRAIN_POOL) ----
    print("\nГенерация обучающих пар (leave-one-book-out, только TRAIN_POOL)...")
    feats_list, labels = _make_training_pairs(rng, encoder, train_pool)
    n_pos = sum(labels)
    print(f"  пар: {len(labels)} (same={n_pos}, different={len(labels) - n_pos})")

    verifier = av.PairwiseVerifier()
    verifier.fit(feats_list, labels)
    # verifier.model - теперь Pipeline(StandardScaler, LogisticRegression) -
    # см. PairwiseVerifier docstring про то, зачем понадобился скейлер.
    logreg = verifier.model.named_steps["logisticregression"]
    coefs = dict(zip(av.SIMILARITY_FEATURE_NAMES, logreg.coef_[0].tolist()))
    print(f"  веса логрегрессии верификатора (после StandardScaler): {coefs}")

    # ---- Калибровка порогов ИСКЛЮЧИТЕЛЬНО на VALIDATION_POOL ----
    print("\nКалибровка порогов на VALIDATION_POOL (профили из TRAIN_POOL, запросы из VALIDATION_POOL)...")
    val_pos, val_neg = _make_validation_scores(rng, encoder, verifier, train_pool, val_pool)
    print(f"  validation: n_pos={len(val_pos)} n_neg={len(val_neg)}")
    threshold_match = av.calibrate_threshold_for_far(val_neg, config.VERIFIER_TARGET_FAR)
    threshold_mismatch = av.calibrate_threshold_for_low_genuine_percentile(
        val_pos, config.VERIFIER_GENUINE_LOW_PERCENTILE)
    if threshold_mismatch >= threshold_match:
        # На маленьких/шумных validation-выборках процентили и FAR-порог
        # иногда пересекаются - тогда честно сужаем зону UNCERTAIN до нуля
        # вместо того чтобы получить логически противоречивые границы.
        mid = (threshold_match + threshold_mismatch) / 2.0
        threshold_match, threshold_mismatch = mid, mid
    val_report = av.full_metrics_report(val_pos, val_neg, config.VERIFIER_TARGET_FAR)
    print(f"  threshold_match={threshold_match:.4f}  threshold_mismatch={threshold_mismatch:.4f}")
    print(f"  validation ROC-AUC={val_report['roc_auc']:.4f}  EER={val_report['eer']:.4f}")

    # ---- Сохранение артефактов ----
    model_dir.mkdir(parents=True, exist_ok=True)
    encoder.save(model_dir / av.ENCODER_FILE)
    verifier.save(model_dir / av.VERIFIER_FILE)

    meta = {
        "open_set_author_excluded": OPEN_SET_AUTHOR,
        "known_authors": known_authors,
        "book_split": {
            a: {"train": list(train_pool[a].keys()), "validation": list(val_pool[a].keys()),
                "test": list(test_pool[a].keys())}
            for a in known_authors
        },
        "n_training_pairs": len(labels),
        "n_training_pairs_positive": n_pos,
        "verifier_coefficients": coefs,
        "verifier_intercept": float(logreg.intercept_[0]),
        "threshold_match": threshold_match,
        "threshold_mismatch": threshold_mismatch,
        "target_far": config.VERIFIER_TARGET_FAR,
        "validation_metrics": val_report,
        "random_seed": seed,
    }
    (model_dir / av.META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено в {model_dir}: {av.ENCODER_FILE}, {av.VERIFIER_FILE}, {av.META_FILE}")

    return {
        "encoder": encoder, "verifier": verifier, "meta": meta,
        "train_pool": train_pool, "val_pool": val_pool, "test_pool": test_pool,
        "by_author_book": by_author_book,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data_processed/dataset.jsonl"))
    ap.add_argument("--model-dir", type=Path, default=Path("model_verifier"))
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args(argv)
    train(args.data, args.model_dir, seed=args.seed)


if __name__ == "__main__":
    main()
