#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py
=====================================================================
Обучает и применяет модель атрибуции авторства на датасете, созданном
`preprocess_corpus.py` (dataset.jsonl: author, book, text, ...).

Подход основан на классической стилометрии, которая на корпусах такого
масштаба (несколько авторов, десятки-сотни тысяч слов на автора) стабильно
превосходит дообучение больших нейросетей "с нуля" и является эталоном в
задачах Authorship Attribution (в т.ч. в конкурсах PAN@CLEF):

  - символьные n-граммы (char n-grams, TF-IDF) - главный сигнал, ловит
    орфографические, пунктуационные и морфологические привычки автора
    независимо от темы текста;
  - частоты служебных/функциональных слов (Burrows' Delta-подобный сигнал) -
    авторский "синтаксический скелет";
  - явные стилометрические признаки: длина предложений и её вариативность,
    богатство словаря (TTR, hapax legomena), пунктуация, доля диалога,
    структура абзацев.

Финальная модель - калиброванный стэкинг (soft-voting/stacking) нескольких
линейных классификаторов поверх этих признаков, что даёт как точный класс,
так и откалиброванную вероятность принадлежности каждому автору.

Обучение (документ-уровневая кросс-валидация, чтобы избежать утечки
данных между чанками одной и той же книги):

    python train_model.py train \
        --data /data_processed/dataset.jsonl \
        --model-dir /model

Инференс на новом (неизвестном) тексте:

    python train_model.py predict \
        --model-dir /model \
        --input /path/to/unknown_text.txt

    python train_model.py predict \
        --model-dir /model \
        --text "some raw text ..."
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                      StratifiedGroupKFold)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
import joblib

from safe_load import safe_joblib_load

# Переиспользуем разбиение на предложения/абзацы и чанкинг из скрипта
# предобработки, чтобы инференс на "сыром" неизвестном тексте проходил
# через ТОТ ЖЕ пайплайн очистки/нарезки, что и обучающие данные.
import preprocess_corpus as prep

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_FILE = "author_style_pipeline.joblib"
LABELS_FILE = "label_encoder.joblib"
META_FILE = "model_meta.json"
OOD_FILE = "ood_novelty.joblib"

# Сколько компонент оставляем при снижении размерности признакового
# пространства (char/word TF-IDF + стилометрия) перед детектором новизны -
# полная TF-IDF-матрица (десятки тысяч признаков) для IsolationForest
# избыточна и шумна, урезанное плотное представление ловит "форму"
# распределения обучающих авторов гораздо надёжнее.
OOD_SVD_COMPONENTS = 100
# Доля обучающих чанков, которую IsolationForest вправе счесть выбросами
# ВНУТРИ самого обучающего корпуса (шумные/нетипичные чанки у настоящих
# авторов тоже бывают) - не путать с порогом принятия решения на инференсе,
# который считается отдельно через перцентили (см. train()).
OOD_CONTAMINATION = 0.02


# =============================================================================
# РУЧНЫЕ СТИЛОМЕТРИЧЕСКИЕ ПРИЗНАКИ
# =============================================================================

# Компактный, но информативный список английских служебных слов
# (предлоги, союзы, местоимения, вспомогательные глаголы, детерминаторы).
# Именно частоты ЭТИХ слов - основа классической атрибуции авторства
# (Burrows' Delta, Mosteller-Wallace) поскольку их употребление почти не
# зависит от темы произведения и сильно зависит от индивидуального стиля.
FUNCTION_WORDS = """
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself
yourselves upon whilst thou thee thy ye
""".split()
FUNCTION_WORDS = sorted(set(FUNCTION_WORDS))

_PUNCT_CHARS = [",", ";", ":", "-", "--", "!", "?", "...", '"', "'", "(", ")"]


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def extract_stylometric_features(text: str) -> dict:
    """Считает набор явных стилометрических признаков для одного чанка."""
    sentences = prep.split_sentences(text)
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    words_lower = [w.strip(".,;:!?\"'()").lower() for w in words]
    words_lower = [w for w in words_lower if w]
    n_words = max(1, len(words))
    n_sentences = max(1, len(sentences))
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    n_paragraphs = max(1, len(paragraphs))

    sent_lengths = [len(s.split()) for s in sentences] or [n_words]
    word_lengths = [len(w) for w in words_lower] or [0]

    vocab = set(words_lower)
    freq = {}
    for w in words_lower:
        freq[w] = freq.get(w, 0) + 1
    hapax = sum(1 for c in freq.values() if c == 1)

    feats = {
        "avg_sentence_len": float(np.mean(sent_lengths)),
        "std_sentence_len": float(np.std(sent_lengths)),
        "avg_word_len": float(np.mean(word_lengths)),
        "std_word_len": float(np.std(word_lengths)),
        "ttr": _safe_div(len(vocab), n_words),                 # type-token ratio
        "hapax_ratio": _safe_div(hapax, n_words),               # богатство словаря
        "avg_paragraph_len_words": _safe_div(n_words, n_paragraphs),
        "sentences_per_paragraph": _safe_div(n_sentences, n_paragraphs),
        "dialogue_paragraph_ratio": _safe_div(
            sum(1 for p in paragraphs if prep.is_dialogue_paragraph(p)), n_paragraphs),
        "comma_per_100w": 100 * _safe_div(text.count(","), n_words),
        "semicolon_per_100w": 100 * _safe_div(text.count(";"), n_words),
        "colon_per_100w": 100 * _safe_div(text.count(":"), n_words),
        "exclaim_per_100w": 100 * _safe_div(text.count("!"), n_words),
        "question_per_100w": 100 * _safe_div(text.count("?"), n_words),
        "dash_per_100w": 100 * _safe_div(text.count("--"), n_words),
        "quote_per_100w": 100 * _safe_div(text.count('"'), n_words),
        "capitalized_word_ratio": _safe_div(
            sum(1 for w in words if w[:1].isupper()), n_words),
        "long_word_ratio": _safe_div(
            sum(1 for w in words_lower if len(w) >= 7), n_words),
        "short_word_ratio": _safe_div(
            sum(1 for w in words_lower if len(w) <= 3), n_words),
    }
    for fw in FUNCTION_WORDS:
        feats[f"fw_{fw}"] = 1000 * _safe_div(freq.get(fw, 0), n_words)
    return feats


class StylometricFeaturizer(BaseEstimator, TransformerMixin):
    """sklearn-совместимый трансформер: текст -> вектор явных
    стилометрических признаков (для использования в Pipeline/FeatureUnion)."""

    def fit(self, X, y=None):
        sample = extract_stylometric_features(X[0] if len(X) else "")
        self.feature_names_ = sorted(sample.keys())
        return self

    def transform(self, X):
        rows = []
        for text in X:
            feats = extract_stylometric_features(text)
            rows.append([feats.get(name, 0.0) for name in self.feature_names_])
        return np.asarray(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_)


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Приводит разреженную матрицу к плотной (нужно StandardScaler-у)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if sparse.issparse(X) else X


# =============================================================================
# ЗАГРУЗКА ДАННЫХ
# =============================================================================

def load_dataset(path: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    """Возвращает (тексты, метки_автора, id_книги_для_группировки, масштаб).

    Поле "scale" есть только в датасетах, построенных
    train_model_multiscale.py (large/medium/semi_small/window/sentence/
    phrase - см. этот файл). Обычный dataset.jsonl от preprocess_corpus.py
    поля "scale" не содержит - в этом случае всем чанкам присваивается
    "large", что корректно отражает реальность (это единственный масштаб,
    на котором такой датасет обучает модель) и не ломает обратную
    совместимость вызова train() на старых файлах."""
    texts, authors, groups, scales = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            texts.append(rec["text"])
            authors.append(rec["author"])
            groups.append(f"{rec['author']}::{rec['book']}")
            scales.append(rec.get("scale", "large"))
    return texts, authors, groups, scales


# =============================================================================
# ПОСТРОЕНИЕ MULTISCALE-ДАТАСЕТА (команда build-multiscale)
# =============================================================================
#
# ПРОБЛЕМА
# -----------------------------------------------------------------------
# Модель выше обучена ИСКЛЮЧИТЕЛЬНО на больших (~1500 слов) чанках
# (data_processed/dataset.jsonl, см. preprocess_corpus.py). Она никогда не
# видела короткие фрагменты во время обучения. Поэтому когда приложение
# просит её оценить стиль по окну подсветки (~180 слов в app.py) или тем
# более по одному предложению, она оказывается вне своего обучающего
# распределения и даёт шумные, малоточные оценки. Это классическое
# несоответствие длины train/inference, а не баг кода приложения - его
# нельзя починить настройкой порогов, только переобучением на данных того
# же масштаба, что и реальный инференс.
#
# РЕШЕНИЕ
# -----------------------------------------------------------------------
# Команда `build-multiscale` берёт уже существующий, размеченный по
# авторам dataset.jsonl и достраивает к нему версии ТЕХ ЖЕ текстов,
# нарезанных на несколько более мелких масштабов:
#
#   large       ~1500 слов   (оригинальные чанки, без изменений)
#   medium       600-900 слов
#   semi_small   250-400 слов
#   window        60-180 слов  (= РЕАЛЬНОЕ окно подсветки в app.py прямо
#                                сейчас: WINDOW_MIN_WORDS..WINDOW_TARGET_WORDS -
#                                если поменяете эти константы в app.py,
#                                поменяйте и здесь, чтобы train-распределение
#                                длин продолжало соответствовать проду)
#   sentence      отдельные РЕАЛЬНЫЕ предложения текста (не окно
#                                фиксированного размера, а именно та единица,
#                                по которой нужна подсветка "по предложениям" -
#                                длина естественно варьируется, обычно
#                                5-40 слов)
#   phrase         10-25 слов  (короткие словосочетания/обрывки предложений -
#                                запас на случай ещё более мелкой подсветки)
#
# Все производные фрагменты - это подстроки РЕАЛЬНОГО, уже вычитанного
# текста конкретного автора (не синтетика и не перефразировка), поэтому
# разметка (author) остаётся достоверной по построению - мы просто режем
# уже правильно размеченный текст на куски поменьше. Модель, обученная на
# всех масштабах одновременно, учится распознавать авторский стиль
# независимо от длины входного текста.
#
# ЧЕСТНОСТЬ ОЦЕНКИ - ПОЧЕМУ ЭТО НЕ УТЕЧКА ДАННЫХ
# -----------------------------------------------------------------------
# train()/GroupKFold выше группирует по книге (author::book), а не по
# отдельному фрагменту. Мелкие производные фрагменты одного и того же
# чанка/книги всегда попадают в ту же группу (train ИЛИ test целиком), что
# и их "родитель" - иначе один и тот же исходный текст в разной нарезке мог
# бы одновременно оказаться и в train, и в test, что завысило бы метрики
# обманчиво. build_multiscale_dataset() сохраняет исходное поле "book" у
# каждого производного фрагмента ровно поэтому.
#
# Запуск:
#   python train_model.py build-multiscale \
#       --data data_processed/dataset.jsonl \
#       --output data_processed/dataset_multiscale.jsonl
#   python train_model.py train \
#       --data data_processed/dataset_multiscale.jsonl \
#       --model-dir ./model_multiscale
#
# Модель НЕ перезаписывает ./model автоматически - вы сами указываете
# --model-dir. Рекомендуется обучить в отдельную папку и сравнить
# held_out_metrics_by_scale (см. train()) со старой моделью перед тем, как
# переключать app.py/api_analyze.py на новую.

MULTISCALE_RANDOM_SEED = 42

# (имя_масштаба, мин_слов, макс_слов, макс_фрагментов_на_один_исходный_чанк)
# Ограничение "макс_фрагментов" не даёт мелким масштабам задавить датасет
# числом почти дублирующих друг друга фрагментов из одной и той же книги -
# без него, например, "phrase" дал бы в разы больше примеров, чем "large",
# и модель могла бы начать игнорировать крупный масштаб при обучении.
# "sentence" обрабатывается отдельной функцией (реальные предложения, а не
# фиксированное окно), поэтому в этом списке не участвует.
MULTISCALE_SCALES = [
    ("medium", 600, 900, 2),
    ("semi_small", 250, 400, 3),
    ("window", 60, 180, 3),
    ("phrase", 10, 25, 4),
]

# Максимум реальных предложений, которые берём из одного чанка, и минимальная
# длина предложения в словах, ниже которой это, скорее всего, не предложение,
# а артефакт сегментации (одинокая кавычка, инициал и т.п.) - такие короткие
# "предложения" всё равно попадут в обучение через масштаб "phrase".
SENTENCE_MAX_PER_CHUNK = 6
SENTENCE_MIN_WORDS = 4
# Настоящие предложения изредка длиннее любого разумного "предложенческого"
# окна (сложносочинённые конструкции случаются) - такие выбросы отбрасываем
# из масштаба "sentence", а не обрезаем, чтобы не создавать полу-предложения
# с ложной меткой "это целое предложение".
SENTENCE_MAX_WORDS = 60


def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_windows(sentences: list[str], lo: int, hi: int,
                       max_windows: int, rng: random.Random) -> list[str]:
    """Копит предложения подряд, пока суммарная длина не попадёт в [lo, hi],
    затем начинает копить заново с нуля (не строго непересекающиеся окна -
    нам нужно разнообразие длины и содержания, а не идеальное покрытие без
    повторов). Если кандидатов получилось больше max_windows - берётся
    воспроизводимая случайная подвыборка."""
    candidates = []
    buf: list[str] = []
    buf_words = 0
    for sentence in sentences:
        buf.append(sentence)
        buf_words += _word_count(sentence)
        if buf_words >= lo:
            if buf_words <= hi:
                candidates.append(" ".join(buf))
            buf = []
            buf_words = 0
    if not candidates:
        return []
    if len(candidates) <= max_windows:
        return candidates
    return rng.sample(candidates, max_windows)


def _phrase_windows(words: list[str], lo: int, hi: int,
                     max_windows: int, rng: random.Random) -> list[str]:
    """Для масштаба 'phrase' предложения почти всегда длиннее верхней
    границы целиком, поэтому здесь режем не по границам предложений, а
    чисто по словесным окнам фиксированного размера - это и есть
    "словосочетания", а не обязательно грамматически полные фразы."""
    if len(words) < lo:
        return []
    size = min(hi, max(lo, len(words)))
    candidates = []
    i = 0
    while i + lo <= len(words):
        end = min(i + size, len(words))
        candidates.append(" ".join(words[i:end]))
        i += size
    if len(candidates) <= max_windows:
        return candidates
    return rng.sample(candidates, max_windows)


def _real_sentence_fragments(sentences: list[str], rng: random.Random) -> list[str]:
    """Берёт реальные предложения как есть (без склейки/обрезки по словам) -
    это и есть целевая единица подсветки "по предложениям". Отфильтровывает
    слишком короткие (артефакты сегментации) и слишком длинные (выбросы)
    предложения, затем берёт воспроизводимую случайную подвыборку, если
    предложений в чанке больше SENTENCE_MAX_PER_CHUNK."""
    candidates = [s for s in sentences
                  if SENTENCE_MIN_WORDS <= _word_count(s) <= SENTENCE_MAX_WORDS]
    if not candidates:
        return []
    if len(candidates) <= SENTENCE_MAX_PER_CHUNK:
        return candidates
    return rng.sample(candidates, SENTENCE_MAX_PER_CHUNK)


def build_multiscale_dataset(source_path: Path, output_path: Path) -> None:
    """Читает dataset.jsonl (author/book/text/chunk_id) и пишет расширенный
    dataset_multiscale.jsonl с дополнительными производными фрагментами на
    масштабах MULTISCALE_SCALES + "sentence" - см. пояснение в комментарии
    к секции выше. Каждая запись сохраняет "book" исходного чанка (для
    группировки в train()) и получает новое поле "scale"."""
    if not source_path.exists():
        raise SystemExit(
            f"Не найден {source_path} - сначала запустите preprocess_corpus.py, "
            f"как описано в README проекта, чтобы получить исходный dataset.jsonl."
        )

    rng = random.Random(MULTISCALE_RANDOM_SEED)
    out_records = []
    n_source = 0

    with source_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_source += 1
            text = rec["text"]
            author = rec["author"]
            book = rec["book"]
            chunk_id = rec["chunk_id"]

            # Оригинальный большой чанк - без изменений, помечен scale="large".
            out_records.append({
                "author": author, "book": book, "chunk_id": chunk_id,
                "text": text, "word_count": rec.get("word_count", _word_count(text)),
                "scale": "large",
            })

            sentences = prep.split_sentences(text)
            words = text.split()

            for scale_name, lo, hi, max_windows in MULTISCALE_SCALES:
                if scale_name == "phrase":
                    frags = _phrase_windows(words, lo, hi, max_windows, rng)
                else:
                    frags = _sentence_windows(sentences, lo, hi, max_windows, rng)
                for j, frag_text in enumerate(frags):
                    out_records.append({
                        "author": author, "book": book,
                        "chunk_id": f"{chunk_id}_{scale_name}_{j}",
                        "text": frag_text, "word_count": _word_count(frag_text),
                        "scale": scale_name,
                    })

            for j, frag_text in enumerate(_real_sentence_fragments(sentences, rng)):
                out_records.append({
                    "author": author, "book": book,
                    "chunk_id": f"{chunk_id}_sentence_{j}",
                    "text": frag_text, "word_count": _word_count(frag_text),
                    "scale": "sentence",
                })

    rng.shuffle(out_records)  # порядок в файле не должен коррелировать со scale/книгой

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_scale: dict[str, int] = {}
    for rec in out_records:
        by_scale[rec["scale"]] = by_scale.get(rec["scale"], 0) + 1

    print(f"Исходных чанков (scale=large): {n_source}")
    print(f"Итоговый multiscale-датасет: {len(out_records)} фрагментов -> {output_path}")
    for scale in ("large", "medium", "semi_small", "window", "sentence", "phrase"):
        if scale in by_scale:
            print(f"  {scale:12s}: {by_scale[scale]}")


# =============================================================================
# ПОСТРОЕНИЕ МОДЕЛИ
# =============================================================================

def _safe_inner_cv(y, requested: int = 3) -> int:
    """Внутренние cv (StackingClassifier, CalibratedClassifierCV) не могут
    запросить больше фолдов, чем самое малочисленное количество примеров
    какого-либо класса в данных. На больших корпусах это всегда просто
    вернёт `requested`; страхует только очень маленькие/несбалансированные
    датасеты."""
    if y is None or len(y) == 0:
        return 2
    counts = np.bincount(y)
    counts = counts[counts > 0]
    min_count = int(counts.min()) if len(counts) else 2
    # StackingClassifier само ещё раз делит train на `requested` частей перед
    # тем как каждый базовый классификатор увидит свою долю - закладываем
    # на это запас, чтобы вложенный CalibratedClassifierCV не запросил
    # больше фолдов, чем реально доступно примеров редкого класса.
    effective_min = max(1, min_count // 2)
    return max(2, min(requested, effective_min))


def build_pipeline(n_authors: int, inner_cv: int = 3) -> Pipeline:
    """Собирает финальный pipeline: FeatureUnion(char-ngram TF-IDF,
    word-ngram TF-IDF, стилометрия) -> стэкинг классификаторов."""

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        min_df=2, max_df=0.95, sublinear_tf=True,
        max_features=60_000,
    )
    word_tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        min_df=2, max_df=0.9, sublinear_tf=True,
        max_features=30_000, lowercase=True,
    )
    stylo = Pipeline([
        ("features", StylometricFeaturizer()),
        ("scale", StandardScaler()),
    ])

    features = FeatureUnion([
        ("char_tfidf", char_tfidf),
        ("word_tfidf", word_tfidf),
        ("stylometric", stylo),
    ])

    # Базовые классификаторы, каждый из которых - известный сильный вариант
    # для TF-IDF-признаков в задачах атрибуции авторства.
    base_estimators = [
        ("linsvc", CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", dual="auto"),
            method="sigmoid", cv=inner_cv)),
        ("logreg", LogisticRegression(
            C=2.0, max_iter=3000, class_weight="balanced")),
        ("sgd", CalibratedClassifierCV(
            SGDClassifier(loss="modified_huber", alpha=1e-5,
                           class_weight="balanced", max_iter=2000),
            method="sigmoid", cv=inner_cv)),
    ]
    final_estimator = LogisticRegression(max_iter=3000)

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
        cv=inner_cv,
    )

    pipeline = Pipeline([
        ("features", features),
        ("clf", stacking),
    ])
    return pipeline


# =============================================================================
# ОБУЧЕНИЕ + ОЦЕНКА КАЧЕСТВА (группировка по книге против утечки данных)
# =============================================================================

def train(data_path: Path, model_dir: Path, cv_folds: int = 5,
          test_size: float = 0.15, random_state: int = 42) -> None:
    print(f"Загрузка датасета: {data_path}")
    texts, authors, groups, scales = load_dataset(data_path)
    n = len(texts)
    if n == 0:
        print("Датасет пуст. Сначала запустите preprocess_corpus.py", file=sys.stderr)
        sys.exit(1)

    le = LabelEncoder()
    y = le.fit_transform(authors)
    n_authors = len(le.classes_)
    print(f"Загружено {n} чанков, {n_authors} авторов: {list(le.classes_)}")
    scale_counts: dict[str, int] = {}
    for s in scales:
        scale_counts[s] = scale_counts.get(s, 0) + 1
    if len(scale_counts) > 1:
        print("Распределение по масштабам фрагмента: "
              + ", ".join(f"{k}={v}" for k, v in sorted(scale_counts.items())))

    n_groups = len(set(groups))
    if n_groups < 2:
        print("Внимание: в датасете только одна книга/группа - "
              "документ-уровневая кросс-валидация невозможна.", file=sys.stderr)

    # ---- Финальный отложенный тест (группировка по книге, чтобы чанки
    # одной и той же книги не оказались одновременно в train и test;
    # стратификация по автору, чтобы даже авторы с малым числом книг
    # гарантированно попали и в train, и в test) ----
    n_splits_holdout = max(2, round(1 / test_size))
    n_splits_holdout = min(n_splits_holdout, n_groups) if n_groups >= 2 else 2
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits_holdout, shuffle=True,
                                     random_state=random_state)
        train_idx, test_idx = next(sgkf.split(texts, y, groups=groups))
    except ValueError:
        # На совсем маленьких/несбалансированных данных стратификация может
        # быть невозможна - откатываемся на обычный группированный сплит.
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                      random_state=random_state)
        train_idx, test_idx = next(splitter.split(texts, y, groups=groups))
    X_train = [texts[i] for i in train_idx]
    y_train = y[train_idx]
    g_train = [groups[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_test = y[test_idx]
    scales_test = [scales[i] for i in test_idx]

    print(f"Train: {len(X_train)} чанков | Test (отложенный, другие книги): {len(X_test)} чанков")

    # ---- Кросс-валидация на train (тоже группированная по книге) ----
    n_splits = min(cv_folds, len(set(g_train)))
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        cv_scores = []
        for fold, (tr_i, va_i) in enumerate(gkf.split(X_train, y_train, groups=g_train)):
            # безопасный запас: stacking сам ещё раз разбивает train на inner_cv
            # частей, поэтому берём дополнительный запас прочности на маленьких данных
            fold_cv = max(2, min(_safe_inner_cv(y_train[tr_i]), 3))
            try:
                pipe = build_pipeline(n_authors, inner_cv=fold_cv)
                pipe.fit([X_train[i] for i in tr_i], y_train[tr_i])
                preds = pipe.predict([X_train[i] for i in va_i])
            except ValueError as exc:
                print(f"  CV фолд {fold + 1}/{n_splits}: пропущен (недостаточно "
                      f"примеров класса в этой группировке) - {exc}")
                continue
            acc = accuracy_score(y_train[va_i], preds)
            f1 = f1_score(y_train[va_i], preds, average="macro")
            cv_scores.append((acc, f1))
            print(f"  CV фолд {fold + 1}/{n_splits}: accuracy={acc:.4f}  macro-F1={f1:.4f}")
        if cv_scores:
            accs, f1s = zip(*cv_scores)
            print(f"Кросс-валидация (группировка по книге): "
                  f"accuracy={np.mean(accs):.4f}±{np.std(accs):.4f}  "
                  f"macro-F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
        else:
            print("Все фолды кросс-валидации пропущены (слишком мало данных на "
                  "класс в разбиении) - переходим к финальному обучению.")
    else:
        print("Недостаточно книг для группированной кросс-валидации - "
              "пропускаем этот шаг и сразу обучаем финальную модель.")

    # ---- Финальное обучение на всём train, оценка на отложенном test ----
    print("Обучение финальной модели на всех train-данных...")
    try:
        final_pipeline = build_pipeline(n_authors, inner_cv=_safe_inner_cv(y_train))
        final_pipeline.fit(X_train, y_train)
    except ValueError:
        print("  Недостаточно данных для калиброванного стэкинга - "
              "понижаем внутреннюю кросс-валидацию до минимума (cv=2).")
        final_pipeline = build_pipeline(n_authors, inner_cv=2)
        final_pipeline.fit(X_train, y_train)

    if len(X_test):
        preds = final_pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", labels=np.arange(n_authors),
                       zero_division=0)
        print(f"\nОтложенный тест (книги, не виденные при обучении): "
              f"accuracy={acc:.4f}  macro-F1={f1:.4f}\n")
        all_labels = np.arange(n_authors)
        print(classification_report(y_test, preds, labels=all_labels,
                                     target_names=le.classes_, digits=4,
                                     zero_division=0))
        cm = confusion_matrix(y_test, preds, labels=all_labels)
        print("Confusion matrix (строки=истина, столбцы=предсказание):")
        print("            " + "  ".join(f"{c[:10]:>10}" for c in le.classes_))
        for cls_name, row in zip(le.classes_, cm):
            print(f"{cls_name[:10]:>10}  " + "  ".join(f"{v:>10d}" for v in row))
        missing = [le.classes_[i] for i in all_labels if i not in set(y_test)]
        if missing:
            print(f"\nВнимание: в отложенном тесте не оказалось ни одной книги "
                  f"авторов {missing} (маленькая группа при группированном "
                  f"разбиении по книгам) - для них метрики выше посчитаны как 0 "
                  f"по поддержке (support=0), это не ошибка модели, а особенность "
                  f"конкретного разбиения. Кросс-валидация выше эту группу всё "
                  f"равно покрывает.")

        # ---- Разбивка отложенного теста ПО МАСШТАБУ фрагмента ----
        # Это единственный объективный способ проверить, что дообучение на
        # multiscale-датасете действительно подняло качество именно на
        # коротких фрагментах (ради чего всё затевалось), а не просто
        # "размыло" модель в среднем. Общий accuracy/F1 выше это скрывает,
        # т.к. он усредняет по всем масштабам сразу, а больших фрагментов
        # ("large") в датасете меньше всего - на общий счёт они влияют слабо
        # даже если сильно просядут.
        by_scale_test: dict[str, int] = {}
        for s in scales_test:
            by_scale_test[s] = by_scale_test.get(s, 0) + 1
        scale_metrics = {}
        if len(by_scale_test) > 1:
            print("\nОтложенный тест в разбивке по масштабу фрагмента "
                  "(так видно, реально ли короткие тексты определяются лучше, "
                  "а длинные не просели):")
            y_test_arr = np.asarray(y_test)
            preds_arr = np.asarray(preds)
            scales_test_arr = np.asarray(scales_test)
            for scale_name in sorted(by_scale_test):
                mask = scales_test_arr == scale_name
                s_acc = accuracy_score(y_test_arr[mask], preds_arr[mask])
                s_f1 = f1_score(y_test_arr[mask], preds_arr[mask], average="macro",
                                 labels=np.arange(n_authors), zero_division=0)
                scale_metrics[scale_name] = {
                    "n": int(mask.sum()), "accuracy": float(s_acc), "macro_f1": float(s_f1),
                }
                print(f"  {scale_name:12s} (n={int(mask.sum()):5d}): "
                      f"accuracy={s_acc:.4f}  macro-F1={s_f1:.4f}")

    # ---- Дообучение на 100% данных для продакшн-модели ----
    print("\nДообучение финальной модели на 100% данных (для использования в проде)...")
    try:
        production_pipeline = build_pipeline(n_authors, inner_cv=_safe_inner_cv(y))
        production_pipeline.fit(texts, y)
    except ValueError:
        print("  Недостаточно данных для калиброванного стэкинга - "
              "понижаем внутреннюю кросс-валидацию до минимума (cv=2).")
        production_pipeline = build_pipeline(n_authors, inner_cv=2)
        production_pipeline.fit(texts, y)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(production_pipeline, model_dir / MODEL_FILE)
    joblib.dump(le, model_dir / LABELS_FILE)

    # ---- Детектор "вне распределения" (novelty) ----
    # ВАЖНО: это НЕ 6-й класс "ИИ" и не переобучение под конкретные LLM.
    # Модель атрибуции авторства (StackingClassifier) - closed-set: она
    # обязана нормализовать вероятность в 100% среди 5 известных авторов,
    # даже если реальный текст не похож ни на одного из них (см. discussion
    # в app.py). Детектор ниже решает другую, дополняющую задачу: "насколько
    # признаковый вектор этого текста вообще типичен для обучающего корпуса
    # (человеческая англоязычная художественная проза XIX века) как единого
    # целого?" - без привязки к тому, какой из пяти авторов ближе. Низкий
    # результат означает, что top-1 атрибуция недостоверна ПО ЛЮБОЙ причине:
    # ИИ-генерация, машинный перевод, нехудожественный текст, плагиат из
    # неизвестного источника и т.п. - детектор не привязан к стилю каких-то
    # конкретных LLM и не устаревает так, как устарел бы явный "класс ИИ".
    #
    # КАЛИБРОВКА ПОРОГА - КРИТИЧЕСКИ ВАЖНАЯ ЧАСТЬ. IsolationForest, оценённый
    # на тех же данных, на которых он обучался, систематически завышает
    # "нормальность" этих данных (лес буквально видел эти точки при
    # построении разбиений) - наивная калибровка на train-скорах даёт
    # обманчиво узкий диапазон и ложные срабатывания на ЛЮБОМ новом тексте,
    # включая настоящие работы человека. Поэтому перцентили (p01/p50) здесь
    # считаются на честных out-of-fold оценках (GroupKFold по книгам - как
    # и при оценке качества классификатора выше): каждый чанк оценивается
    # лесом, который его не видел. Итоговый "боевой" лес для продакшна при
    # этом всё равно обучается на 100% данных - для максимального охвата.
    print("\nОбучение и калибровка детектора 'вне распределения' (novelty)...")
    feat_matrix = production_pipeline.named_steps["features"].transform(texts)
    n_components = min(OOD_SVD_COMPONENTS, feat_matrix.shape[1] - 1, max(feat_matrix.shape[0] - 1, 1))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced_all = svd.fit_transform(feat_matrix)

    n_splits_ood = min(5, n_groups) if n_groups >= 2 else 0
    oof_scores = np.full(len(texts), np.nan)
    if n_splits_ood >= 2:
        gkf_ood = GroupKFold(n_splits=n_splits_ood)
        for fold_i, (tr_i, va_i) in enumerate(gkf_ood.split(reduced_all, y, groups=groups)):
            iso_fold = IsolationForest(n_estimators=300, contamination=OOD_CONTAMINATION,
                                        random_state=random_state, n_jobs=-1)
            iso_fold.fit(reduced_all[tr_i])
            oof_scores[va_i] = iso_fold.score_samples(reduced_all[va_i])
            print(f"  novelty-калибровка, фолд {fold_i + 1}/{n_splits_ood}: готово")

    valid_mask = ~np.isnan(oof_scores)
    if valid_mask.sum() >= 20:
        p01, p50 = (float(v) for v in np.percentile(oof_scores[valid_mask], [1, 50]))
    else:
        # Слишком мало данных/групп для честного out-of-fold - откатываемся
        # на in-sample калибровку с явным предупреждением; НЕ для продакшна
        # с малыми корпусами без ручной перепроверки порога.
        print("  ВНИМАНИЕ: недостаточно данных для честной out-of-fold "
              "калибровки novelty (< 20 валидных оценок) - используется "
              "оптимистичная in-sample калибровка, порог требует ручной "
              "проверки перед использованием в проде.")
        iso_probe = IsolationForest(n_estimators=300, contamination=OOD_CONTAMINATION,
                                     random_state=random_state, n_jobs=-1).fit(reduced_all)
        p01, p50 = (float(v) for v in np.percentile(iso_probe.score_samples(reduced_all), [1, 50]))

    # Финальный "боевой" лес - на 100% данных, для максимального охвата в проде.
    iso = IsolationForest(n_estimators=300, contamination=OOD_CONTAMINATION,
                           random_state=random_state, n_jobs=-1)
    iso.fit(reduced_all)

    joblib.dump({"svd": svd, "iso": iso, "p01": p01, "p50": p50}, model_dir / OOD_FILE)
    print(f"Novelty-детектор сохранён: {model_dir / OOD_FILE} "
          f"(honest out-of-fold p01={p01:.4f}, p50={p50:.4f}, компонент SVD={n_components}, "
          f"валидных oof-оценок={int(valid_mask.sum())}/{len(texts)})")

    meta = {
        "authors": list(le.classes_),
        "n_chunks_trained_on": n,
        "held_out_accuracy": float(acc) if len(X_test) else None,
        "held_out_macro_f1": float(f1) if len(X_test) else None,
        "held_out_metrics_by_scale": scale_metrics if len(X_test) else None,
        "scale_distribution_trained_on": scale_counts,
        "target_words_per_chunk": "1450-1550 (см. preprocess_corpus.py)",
        "ood_detector": "IsolationForest поверх TruncatedSVD(features), см. ood_novelty.joblib",
    }
    (model_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"\nМодель сохранена в: {model_dir / MODEL_FILE}")
    print(f"Метаданные: {model_dir / META_FILE}")


# =============================================================================
# ИНФЕРЕНС НА НОВОМ ТЕКСТЕ
# =============================================================================

def load_model(model_dir: Path):
    # safe_joblib_load, не joblib.load напрямую - см. safe_load.py,
    # SECURITY_AND_HARDENING_TODO.md, п.24 (защита на будущее: не даёт
    # выйти за пределы model_dir, если этот путь когда-нибудь начнёт
    # собираться из непроверенного пользовательского ввода).
    pipeline = safe_joblib_load(model_dir / MODEL_FILE, allowed_dir=model_dir)
    le = safe_joblib_load(model_dir / LABELS_FILE, allowed_dir=model_dir)
    return pipeline, le


def load_ood_detector(model_dir: Path) -> dict | None:
    """Возвращает детектор новизны (svd + isolation forest + перцентили
    train-корпуса) или None, если модель обучена до появления этой фичи и
    ood_novelty.joblib отсутствует - в этом случае весь код выше по стеку
    должен продолжать работать, просто без сигнала новизны (обратная
    совместимость со старыми артефактами модели)."""
    path = model_dir / OOD_FILE
    if not path.exists():
        return None
    return safe_joblib_load(path, allowed_dir=model_dir)


def novelty_pct_scores(chunk_texts: list[str], pipeline, ood: dict) -> np.ndarray:
    """Для каждого чанка считает 0-100%: насколько его признаковый вектор
    типичен для обучающего корпуса пяти авторов В ЦЕЛОМ (не для конкретного
    автора). Это НЕ вероятность и не альтернатива style_score - это
    независимая проверка достоверности самой стилевой атрибуции. Низкий %
    означает: top-1 автор из ranking, каким бы высоким ни был его
    относительный процент, не стоит доверять - текст лежит вне того, на чём
    вообще обучалась модель."""
    feat_matrix = pipeline.named_steps["features"].transform(chunk_texts)
    reduced = ood["svd"].transform(feat_matrix)
    raw = ood["iso"].score_samples(reduced)  # выше = типичнее для train-корпуса
    p01, p50 = ood["p01"], ood["p50"]
    span = max(p50 - p01, 1e-6)
    # Линейная шкала: p01 обучающей выборки -> ~0%, p50 -> ~70%; специально
    # не растягиваем до 100% на p50, чтобы даже "типичный" человеческий
    # чанк не выглядел как абсолютная гарантия - клип сверху всё равно есть.
    pct = 70.0 * (raw - p01) / span
    return np.clip(pct, 0.0, 100.0)


def predict_author(text: str, model_dir: Path,
                    chunk_cfg: prep.ChunkingConfig | None = None) -> dict:
    """Определяет автора произвольного (неизвестного) текста.

    Текст прогоняется через ТОТ ЖЕ пайплайн очистки и нарезки на окна
    ~1500 слов, что и обучающие данные, затем модель предсказывает
    вероятности по каждому окну, и они агрегируются (усреднение
    логарифмов вероятностей = геометрическое среднее) в единый вердикт
    по всему тексту - это надёжнее, чем решение по одному окну.
    """
    pipeline, le = load_model(model_dir)
    ood = load_ood_detector(model_dir)
    cfg = chunk_cfg or prep.ChunkingConfig()

    cleaned = prep.clean_raw_text(text)
    paragraphs = prep.build_paragraph_index(cleaned)
    if not paragraphs:
        raise ValueError("После очистки текст оказался пустым - нечего анализировать.")

    chunks = prep.chunk_paragraphs(paragraphs, cfg)
    chunk_texts = [c["text"] for c in chunks]

    proba = pipeline.predict_proba(chunk_texts)  # (n_chunks, n_authors)
    log_proba = np.log(np.clip(proba, 1e-12, 1.0))
    mean_log_proba = log_proba.mean(axis=0)
    agg_proba = np.exp(mean_log_proba)
    agg_proba = agg_proba / agg_proba.sum()

    order = np.argsort(-agg_proba)
    ranked = [
        {"author": le.classes_[i], "probability": round(float(agg_proba[i]), 4)}
        for i in order
    ]

    chunk_novelty = novelty_pct_scores(chunk_texts, pipeline, ood) if ood is not None else None

    per_chunk = []
    for i, ct in enumerate(chunk_texts):
        p = proba[i]
        top = int(np.argmax(p))
        entry = {
            "chunk_id": i,
            "word_count": chunks[i]["word_count"],
            "predicted_author": le.classes_[top],
            "confidence": round(float(p[top]), 4),
        }
        if chunk_novelty is not None:
            entry["novelty_pct"] = round(float(chunk_novelty[i]), 1)
        per_chunk.append(entry)

    novelty_pct = float(np.mean(chunk_novelty)) if chunk_novelty is not None else None

    return {
        "predicted_author": ranked[0]["author"],
        "confidence": ranked[0]["probability"],
        "ranking": ranked,
        "n_chunks_analyzed": len(chunk_texts),
        "per_chunk_predictions": per_chunk,
        # None означает "модель обучена без детектора новизны" (старые
        # артефакты) - вызывающий код должен трактовать это как "сигнал
        # недоступен", а НЕ как "текст типичен".
        "novelty_pct": round(novelty_pct, 1) if novelty_pct is not None else None,
    }


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Обучить модель на dataset.jsonl")
    p_train.add_argument("--data", type=Path, required=True,
                          help="Путь к dataset.jsonl (результат preprocess_corpus.py "
                               "или build-multiscale)")
    p_train.add_argument("--model-dir", type=Path, default=Path("./model"))
    p_train.add_argument("--cv-folds", type=int, default=5)
    p_train.add_argument("--test-size", type=float, default=0.15)

    p_multi = sub.add_parser(
        "build-multiscale",
        help="Достроить dataset.jsonl фрагментами разной длины (medium/"
             "semi_small/window/sentence/phrase) для устойчивой посегментной "
             "подсветки стиля - см. комментарий к build_multiscale_dataset()")
    p_multi.add_argument("--data", type=Path,
                          default=Path("data_processed/dataset.jsonl"),
                          help="Исходный dataset.jsonl (по умолчанию "
                               "data_processed/dataset.jsonl)")
    p_multi.add_argument("--output", type=Path,
                          default=Path("data_processed/dataset_multiscale.jsonl"),
                          help="Куда сохранить расширенный датасет")

    p_pred = sub.add_parser("predict", help="Определить автора нового текста")
    p_pred.add_argument("--model-dir", type=Path, default=Path("./model"))
    group = p_pred.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Путь к .txt файлу неизвестного текста")
    group.add_argument("--text", type=str, help="Текст напрямую строкой")
    p_pred.add_argument("--json", action="store_true",
                         help="Вывести полный результат в формате JSON")

    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.command == "train":
        train(args.data, args.model_dir, cv_folds=args.cv_folds,
              test_size=args.test_size)

    elif args.command == "build-multiscale":
        build_multiscale_dataset(args.data, args.output)
        print("\nГотово. Дальше обучите модель на новом датасете отдельной командой:")
        print(f"    python train_model.py train --data {args.output} "
              f"--model-dir ./model_multiscale")
        print("(в отдельную папку, а не в ./model, чтобы можно было сравнить "
              "held_out_metrics_by_scale старой и новой модели перед переключением)")

    elif args.command == "predict":
        raw_text = args.input.read_text(encoding="utf-8", errors="replace") \
            if args.input else args.text
        result = predict_author(raw_text, args.model_dir)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"\nПредсказанный автор: {result['predicted_author']} "
                  f"(уверенность {result['confidence']*100:.1f}%)")
            print(f"Проанализировано окон: {result['n_chunks_analyzed']}")
            if result["novelty_pct"] is not None:
                print(f"Типичность для обучающего корпуса (novelty): "
                      f"{result['novelty_pct']:.1f}% "
                      f"{'-- ВНЕ РАСПРЕДЕЛЕНИЯ, атрибуции ниже доверять нельзя' if result['novelty_pct'] < 35 else ''}")
            print("\nПолное распределение вероятностей по авторам:")
            for r in result["ranking"]:
                bar = "#" * int(r["probability"] * 40)
                print(f"  {r['author']:<20} {r['probability']*100:6.2f}%  {bar}")


if __name__ == "__main__":
    main()