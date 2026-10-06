#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author_verification.py
=====================================================================
PROFILE-BASED (OPEN-SET) AUTHORSHIP VERIFICATION - ядро нового Style
Check, заменяющего closed-set классификацию "к какому из 5 известных
авторов ближе текст" на реальный продуктовый сценарий SHYNDYQ:

    У КОНКРЕТНОГО студента есть несколько предыдущих работ.
    Из них строится его индивидуальный Style Profile.
    Новая работа сравнивается С ЭТИМ профилем, а не с чужими 5 классами.

ПОЧЕМУ ЭТО НЕ ТО ЖЕ САМОЕ, ЧТО train_model.py
-------------------------------------------------------------------
train_model.py решает ЗАКРЫТУЮ задачу классификации: pipeline.predict()
обязан вернуть один из N=5 фиксированных классов (даже если текст не
похож ни на одного из них - вероятности всё равно нормализуются в 100%
между известными классами). Это в принципе не может ответить на вопрос
"похож ли ЭТОТ студент на самого себя?", потому что студент никогда не
был одним из этих 5 классов.

Здесь же автор (кандидат) не выбирается из фиксированного списка - у
него в принципе может не быть класса вообще. Мы проверяем ГИПОТЕЗУ
("это тот же автор, что писал профильные тексты?"), а не выбираем
ближайший из известных ярлыков. Это классическая постановка Authorship
VERIFICATION (в противовес Attribution) - open-set, hypothesis-testing.

ЧТО ПЕРЕИСПОЛЬЗУЕТСЯ ИЗ СУЩЕСТВУЮЩЕГО ПРОЕКТА (см. п.13 тех.задания)
-------------------------------------------------------------------
  - train_model.StylometricFeaturizer / extract_stylometric_features /
    FUNCTION_WORDS - весь существующий набор явных стилометрических
    признаков (длина/вариативность предложений, TTR, hapax, пунктуация,
    доля диалога, частоты служебных слов и т.д.) используется как есть,
    без переписывания - именно это и есть "структурные" + "function
    words" + "punctuation" + "lexical diversity" признаки профиля.
  - preprocess_corpus.shingles/jaccard - для мягкой (не блокирующей)
    проверки на почти-дубликаты между профилем и новым текстом, точно
    та же техника, что preprocess_corpus.py уже использует для поиска
    дублей в корпусе при препроцессинге.
  - char_wb / word TF-IDF в том же духе, что и train_model.build_pipeline
    (char_wb 2-5 грамм + word 1-2 грамм) - тот же тип признаков, что уже
    доказал себя как основной сигнал атрибуции авторства в проекте,
    только теперь это ОТДЕЛЬНЫЙ, самостоятельно переобучаемый энкодер
    (см. train_verifier.py), не завязанный на 5 фиксированных классов.

АРХИТЕКТУРА (см. также README.md)
-------------------------------------------------------------------
  StyleEncoder           текст -> вектор признаков (char/word TF-IDF +
                          стилометрия). Общий, переиспользуемый, НЕ знает
                          ничего об "авторах" как классах.

  build_author_profile   [текст, текст, ...] -> AuthorProfile (устойчивое
                          агрегированное представление КОНКРЕТНОГО автора:
                          mean/std по всем признаковым пространствам +
                          метаданные качества профиля).

  compare_text_to_profile  (новый_текст, профиль) -> результат сравнения
                          (score/verdict/confidence/evidence) - НЕ
                          классификация, а проверка гипотезы совпадения.

  PairwiseVerifier        необязательная обученная надстройка (Logistic
                          Regression поверх нескольких сходств -
                          char/word/stylo), которая обучается на ПАРАХ
                          (текст, профиль) -> same/different (см.
                          train_verifier.py.). Без неё compare_text_to_profile
                          всё равно работает - на понятной ручной формуле
                          взвешенного сходства (см. _heuristic_combine).
                          Верификатор НИГДЕ не хранит и не использует
                          identity/label конкретных обучающих авторов -
                          только обезличенные признаки сходства (см.
                          SIMILARITY_FEATURE_NAMES), поэтому
                          он по построению применим к любому кандидату,
                          включая тех, кого не было в обучении (open-set).
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import warnings
import collections
from dataclasses import dataclass, field
from pathlib import Path

import joblib

from safe_load import safe_joblib_load
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import preprocess_corpus as prep
import train_model as tm

warnings.filterwarnings("ignore", category=UserWarning)

ENCODER_FILE = "style_encoder.joblib"
VERIFIER_FILE = "pairwise_verifier.joblib"
META_FILE = "verifier_meta.json"

# Признаки сходства, которые видит PairwiseVerifier - см. также
# _EXPERIMENT_FEATURES_NOTE ниже про то, откуда взялся именно этот набор.
# ВАЖНО: "lq"/"lp"/"ln" (сырая длина запроса/профиля) НЕ входят в этот
# список, хотя и вычисляются в similarity_features_from_encoded() (см. там)
# и используются как МНОЖИТЕЛИ во взаимодействиях ниже - см.
# _EXPERIMENT_FEATURES_NOTE про то, почему включение их НАПРЯМУЮ в модель
# было проверено и отклонено (катастрофическая потеря open-set обобщения).
SIMILARITY_FEATURE_NAMES = [
    "sim_char", "sim_word", "stylo_cosine", "stylo_delta_sim",
    "fw_cos", "fw_delta", "pu_cos", "pu_delta",
    "sim_char_x_lq", "sim_char_x_ln",
    "sim_word_x_lq", "sim_word_x_ln",
    "stylo_delta_sim_x_lq", "stylo_delta_sim_x_ln",
    "fw_delta_x_lq", "fw_delta_x_ln",
    "fw_cos_x_lq", "fw_cos_x_ln",
]

# Исходные 4 - единственные, которые видит ручная (нетренированная)
# эвристика _heuristic_combine(), см. HEURISTIC_WEIGHTS ниже - расширение
# SIMILARITY_FEATURE_NAMES выше её не затрагивает намеренно (веса для
# новых признаков никто вручную не калибровал, в отличие от этих 4).
_HEURISTIC_FEATURE_NAMES = ["sim_char", "sim_word", "stylo_cosine", "stylo_delta_sim"]

# ---- experiments/verifier_experiments.py + verifier_diagnose_openset.py:
# откуда взялся именно этот набор, и что было ОТКЛОНЕНО -------------------
# Первый проход (experiments/verifier_experiments.py, 3 сида книжных
# разбиений) показал: добавление lq/lp/ln (лог-длина запроса/суммарной
# длины и числа текстов профиля) КАК ПРЯМЫХ признаков + их взаимодействий с
# существующими 4 сходствами дало устойчивый прирост на смешанном
# известные+unseen тестовом пуле (ROC-AUC 0.734 -> 0.766, TPR@FAR=2%
# 0.201 -> 0.237). Полная переобучка ЭТОГО набора (train_verifier.py) на
# ПРОДАКШН-протоколе дала на первый взгляд ещё лучший результат на
# известных авторах (validation ROC-AUC 0.918 -> 0.957) - НО катастрофически
# провалила п.C evaluate.py (open-set автор, никогда не видimg классификатором):
# ROC-AUC 0.923 -> 0.638, TPR@FAR=2% 0.6 -> 0.0. Раздельная диагностика
# (experiments/verifier_diagnose_openset.py, та же C_open_set-методология)
# показала: причина - ИМЕННО сырые lp/ln как отдельные признаки. Все
# обучающие пары в train_verifier.py строятся с профилем из 2-6 книг
# (~1500 слов) - т.е. lp/ln почти не варьируются на обучении, и логрегрессия
# выучивает по ним узкую, не обобщающуюся корреляцию, которая полностью
# ломается на open-set профиле другого размера (в п.C - до 60 текстов).
# Убрав lp/ln как ПРЯМЫЕ признаки (оставив их только внутри произведений
# sim_x_lq/sim_x_ln), open-set ROC-AUC вернулся к 0.993 - т.е. тот же
# уровень, что и у исходных 4 признаков, при этом сохранив весь измеримый
# прирост на известных авторах. Также проверялись text-distortion
# (Stamatatos 2017) и частоты служебных слов по фиксированному словарю
# отдельно от fw_cos/fw_delta выше - ни то, ни другое не дало прироста
# ПОВЕРХ этого набора (разница в пределах шума между сидами), поэтому в
# продукт не вошли - самый простой набор, показавший весь прирост, без
# open-set регрессии.
# ВАЖНО: это измерено на 4 англоязычных авторах художественной прозы
# (Gutenberg) с MarkTwain как held-out - НЕ на студенческих эссе. Прежде
# чем полагаться на это в проде, ПОВТОРИТЕ ту же диагностику (C_open_set)
# на реальных студенческих данных - см. SECURITY_AND_HARDENING_TODO.md, п.11.

# Веса ручной (baseline) формулы, когда обученного PairwiseVerifier нет -
# см. _heuristic_combine(). Сумма = 1.0.
HEURISTIC_WEIGHTS = {
    "sim_char": 0.35,
    "sim_word": 0.25,
    "stylo_cosine": 0.15,
    "stylo_delta_sim": 0.25,
}

# Список служебных слов (fw_cos/fw_delta) - переиспользуем существующий в
# проекте список (train_model.FUNCTION_WORDS), не заводим второй (см.
# докстринг модуля выше про переиспользование существующих признаков).
_FUNCTION_WORDS = tm.FUNCTION_WORDS
# Пунктуация (pu_cos/pu_delta) - фиксированный, небольшой набор символов,
# частота которых на 1000 символов текста - классический, устойчивый к
# теме стилометрический сигнал (Burrows и др.).
_PUNCT_CHARS = [",", ";", ":", "!", "?", ".", "-", "\u2014",
                '"', "'", "(", ")", "\u201c", "\u201d", "\u2019"]


def _function_word_freqs(text: str) -> np.ndarray:
    tokens = re.findall(r"[a-z']+", text.lower())
    n = max(1, len(tokens))
    counts = collections.Counter(tokens)
    return np.array([counts[w] / n for w in _FUNCTION_WORDS])


def _punct_freqs(text: str) -> np.ndarray:
    n_chars = max(1, len(text))
    return np.array([text.count(p) / n_chars * 1000.0 for p in _PUNCT_CHARS])


class ProfileLeakageError(ValueError):
    """Поднимается, когда new_text - это буквально тот же документ, что уже
    участвовал в построении профиля кандидата. Сравнение текста с профилем,
    построенным ИЗ НЕГО ЖЕ, тривиально даёт "совпадение" и ничего не
    проверяет - поэтому это ошибка использования API, а не легитимный
    (пусть и уверенный) результат верификации."""


# =============================================================================
# УТИЛИТЫ: нормализация/хэш текста для защиты от утечки и дублей
# =============================================================================

def _normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def text_fingerprint(text: str) -> str:
    """Стабильный хэш ТОЧНОГО содержимого текста (после тривиальной
    нормализации пробелов/регистра) - используется, чтобы гарантированно
    запретить один и тот же документ одновременно в профиле и в new_text
    (см. ProfileLeakageError) и чтобы не размножать профиль дублями."""
    return hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()


def near_duplicate_ratio(text_a: str, text_b: str, k: int = 50) -> float:
    """Мягкая (не блокирующая) проверка на почти-дубликат - переиспользует
    ту же shingle/Jaccard-технику, что preprocess_corpus.find_duplicates
    использует при чистке корпуса. В отличие от text_fingerprint, ловит
    "тот же текст с парой правок", а не только байт-в-байт совпадение."""
    a, b = prep.shingles(text_a, k=k), prep.shingles(text_b, k=k)
    if not a or not b:
        return 0.0
    return prep.jaccard(a, b)


# =============================================================================
# ПРОСТАЯ ЭВРИСТИКА ЯЗЫКА/ПИСЬМЕННОСТИ
# =============================================================================

_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_script(text: str) -> dict:
    """НЕ настоящее определение языка (ru vs kk не различить только по
    алфавиту - оба кириллические) - только грубая проверка письменности,
    достаточная, чтобы понять "это вообще английский текст, для которого
    обучены char/word n-граммы, или нет". См. config.CYRILLIC_RATIO_NON_ENGLISH
    и README про честные ограничения ru/kk поддержки."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return {"cyrillic_ratio": 0.0, "latin_ratio": 0.0, "likely_non_english": False}
    n = len(letters)
    cyr = sum(1 for c in letters if _CYRILLIC_RE.match(c))
    lat = sum(1 for c in letters if _LATIN_RE.match(c))
    cyr_ratio = cyr / n
    lat_ratio = lat / n
    import config
    return {
        "cyrillic_ratio": round(cyr_ratio, 4),
        "latin_ratio": round(lat_ratio, 4),
        "likely_non_english": cyr_ratio >= config.CYRILLIC_RATIO_NON_ENGLISH,
    }


# =============================================================================
# STYLE ENCODER - переиспользуемое, ОБЩЕЕ (не завязанное на 5 авторов)
# отображение текст -> признаки
# =============================================================================

class StyleEncoder:
    """char/word TF-IDF (в том же духе, что train_model.build_pipeline) +
    существующий StylometricFeaturizer проекта. В отличие от
    author_style_pipeline.joblib, это НЕ классификатор - здесь нет и не
    может быть слоя "5 классов". fit() один раз на достаточно большом и
    разнообразном фоновом корпусе (см. train_verifier.py), дальше просто
    переиспользуется как проекция в фиксированное признаковое
    пространство для ЛЮБОГО текста/автора, увиденного или нет."""

    def __init__(self, char_max_features: int = 20_000, word_max_features: int = 15_000):
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5),
            min_df=2, max_df=0.95, sublinear_tf=True,
            max_features=char_max_features,
        )
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            min_df=2, max_df=0.9, sublinear_tf=True,
            max_features=word_max_features, lowercase=True,
        )
        # Переиспользуем существующий трансформер стилометрии проекта как есть.
        self.stylo_featurizer = tm.StylometricFeaturizer()
        self.stylo_scaler = StandardScaler()
        # fw/pu z-нормировка - заполняется в fit() (см. там). None до fit() -
        # так же, как остальные *_vectorizer до fit() не умеют .transform().
        # ВАЖНО: энкодеры, сохранённые ДО этой правки (без fw_mu/pu_mu),
        # несовместимы с новым similarity_features_from_encoded() - при
        # обновлении кода нужно переобучить style_encoder.joblib заново
        # (train_verifier.py), как и при любом другом изменении признакового
        # пространства энкодера.
        self.fw_mu = self.fw_sd = self.pu_mu = self.pu_sd = None
        self.is_fitted = False
        self.n_fit_texts = 0

    def fit(self, texts: list[str]) -> "StyleEncoder":
        self.char_vectorizer.fit(texts)
        self.word_vectorizer.fit(texts)
        self.stylo_featurizer.fit(texts)
        stylo_raw = self.stylo_featurizer.transform(texts)
        self.stylo_scaler.fit(stylo_raw)
        # ---- fw/pu z-нормировка (см. SIMILARITY_FEATURE_NAMES/fw_cos и т.д.) ----
        # mean/std служебных-слов и пунктуации-частот СЧИТАЮТСЯ НА TRAIN, как
        # и весь остальной энкодер - иначе тестовые тексты неявно "подглядывали"
        # бы в свою собственную статистику через нормировку (data leakage).
        fw_raw = np.vstack([_function_word_freqs(t) for t in texts])
        pu_raw = np.vstack([_punct_freqs(t) for t in texts])
        self.fw_mu = fw_raw.mean(axis=0)
        self.fw_sd = fw_raw.std(axis=0) + 1e-6
        self.pu_mu = pu_raw.mean(axis=0)
        self.pu_sd = pu_raw.std(axis=0) + 1e-6
        self.is_fitted = True
        self.n_fit_texts = len(texts)
        return self

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "StyleEncoder не обучен - вызовите fit() или загрузите "
                "готовый энкодер (load_style_encoder / train_verifier.py)."
            )

    def transform_char(self, texts: list[str]) -> sparse.csr_matrix:
        self._check_fitted()
        return self.char_vectorizer.transform(texts)

    def transform_word(self, texts: list[str]) -> sparse.csr_matrix:
        self._check_fitted()
        return self.word_vectorizer.transform(texts)

    def transform_stylo_raw(self, texts: list[str]) -> np.ndarray:
        self._check_fitted()
        return self.stylo_featurizer.transform(texts)

    def transform_stylo_scaled(self, texts: list[str]) -> np.ndarray:
        return self.stylo_scaler.transform(self.transform_stylo_raw(texts))

    @property
    def stylo_feature_names(self) -> list[str]:
        return list(self.stylo_featurizer.feature_names_)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path, allowed_dir: Path | None = None) -> "StyleEncoder":
        # safe_joblib_load, не joblib.load напрямую, когда вызывающий код
        # передаёт allowed_dir - см. safe_load.py, SECURITY_AND_HARDENING_TODO.md,
        # п.24 (защита на будущее против пути вне ожидаемой директории моделей).
        if allowed_dir is not None:
            return safe_joblib_load(path, allowed_dir=allowed_dir)
        return joblib.load(path)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _mean_dense(mat: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(mat.mean(axis=0)).ravel()


# =============================================================================
# AUTHOR PROFILE
# =============================================================================

@dataclass
class AuthorProfile:
    candidate_author_id: str
    n_texts: int
    n_texts_deduplicated: int
    total_words: int
    per_text_word_counts: list[int]
    text_fingerprints: set[str]
    raw_texts: list[str]             # дедуплицированные исходники - только для near-dup проверки
    char_mean: np.ndarray
    word_mean: np.ndarray
    stylo_mean: np.ndarray          # СЫРОЕ (нешкалированное) пространство
    stylo_std: np.ndarray           # std профиля по каждому признаку (для справки/legacy)
    stylo_median: np.ndarray        # медиана - устойчива к выбросам при малом n_texts
    stylo_mad: np.ndarray           # median absolute deviation - устойчивая замена std
    stylo_feature_names: list[str]
    fw_z: np.ndarray                # средний z-score частот служебных слов по профилю
    pu_z: np.ndarray                # средний z-score частот пунктуации по профилю
    consistency: float              # среднее попарное сходство собственных текстов профиля (0..1)
    quality: dict = field(default_factory=dict)
    language: dict = field(default_factory=dict)
    duplicates_removed: int = 0

    def to_public_dict(self) -> dict:
        """Компактное представление для API/логов - БЕЗ сырых векторов
        признаков (они внутренние и ничего не говорят пользователю)."""
        return {
            "candidate_author_id": self.candidate_author_id,
            "number_of_profile_texts": self.n_texts_deduplicated,
            "total_words": self.total_words,
            "per_text_word_counts": self.per_text_word_counts,
            "consistency": round(self.consistency, 3),
            "duplicates_removed": self.duplicates_removed,
            "quality": self.quality,
            "language": self.language,
        }


def _min_words_check(total_words: int, n_texts: int) -> dict:
    import config
    reasons = []
    if n_texts < config.PROFILE_MIN_TEXTS:
        reasons.append(
            f"недостаточно предыдущих работ в профиле: {n_texts} "
            f"(минимум {config.PROFILE_MIN_TEXTS})"
        )
    if total_words < config.PROFILE_MIN_TOTAL_WORDS:
        reasons.append(
            f"суммарно слишком мало текста в профиле: {total_words} слов "
            f"(минимум {config.PROFILE_MIN_TOTAL_WORDS})"
        )
    return {"passes": not reasons, "reasons": reasons}


def assess_profile_quality(n_texts: int, total_words: int,
                            per_text_word_counts: list[int],
                            consistency: float) -> dict:
    """Отдельная, явная оценка качества Style Profile (п.8 задания) - НЕ
    зависит от того, какой текст мы потом будем с ним сравнивать. Именно
    от этой оценки зависит, может ли verdict вообще быть MATCH/MISMATCH,
    или он принудительно понижается до UNCERTAIN (см. compare_text_to_profile)."""
    basic = _min_words_check(total_words, n_texts)
    short_texts = sum(1 for w in per_text_word_counts if w < 150)

    reasons = list(basic["reasons"])
    if short_texts and short_texts == n_texts:
        reasons.append("все работы в профиле очень короткие (<150 слов)")
    if consistency < 0.15 and n_texts >= 2:
        reasons.append(
            f"низкая внутренняя согласованность профиля (consistency={consistency:.2f}) "
            f"- предыдущие работы стилистически сильно расходятся между собой"
        )

    if not basic["passes"]:
        level = "insufficient"
    elif reasons:
        level = "limited"
    elif n_texts >= 5 and total_words >= 4000:
        level = "good"
    else:
        level = "adequate"

    return {
        "level": level,               # insufficient | limited | adequate | good
        "n_texts": n_texts,
        "total_words": total_words,
        "consistency": round(consistency, 3),
        "reasons": reasons,
    }


def build_author_profile(texts: list[str], encoder: StyleEncoder,
                          candidate_author_id: str = "unknown") -> AuthorProfile:
    """Строит устойчивый Style Profile кандидата ИЗ ЕГО (и только его)
    предыдущих работ. Ничего из другого автора сюда попасть не может -
    функция принимает только тексты ОДНОГО кандидата за раз (никакого
    смешивания авторов - см. п.1 задания)."""
    if not texts:
        raise ValueError("build_author_profile: пустой список текстов - профиль строить не из чего.")

    # --- дедупликация: один и тот же документ не должен учитываться дважды
    # (п.14 "repeated same document -> blocked from both profile and test") ---
    seen: dict[str, str] = {}
    duplicates_removed = 0
    for t in texts:
        fp = text_fingerprint(t)
        if fp in seen:
            duplicates_removed += 1
            continue
        seen[fp] = t
    dedup_texts = list(seen.values())
    fingerprints = set(seen.keys())

    word_counts = [len(t.split()) for t in dedup_texts]
    total_words = sum(word_counts)

    char_mat = encoder.transform_char(dedup_texts)
    word_mat = encoder.transform_word(dedup_texts)
    stylo_raw = encoder.transform_stylo_raw(dedup_texts)

    char_mean = _mean_dense(char_mat)
    word_mean = _mean_dense(word_mat)
    stylo_mean = stylo_raw.mean(axis=0)
    # ddof=0 - профиль иногда строится всего из 1 текста (std=0 в этом
    # случае корректно и обрабатывается ниже через эпсилон в compare_*).
    stylo_std = stylo_raw.std(axis=0)
    # Медиана/MAD - устойчивая (robust) альтернатива mean/std, особенно
    # важная при МАЛЫХ профилях (1-3 текста), где среднее и std считаются
    # по считанным точкам и один нетипичный текст может сильно их сдвинуть.
    # MAD = median(|x_i - median(x)|); множитель 1.4826 - стандартная
    # поправка, приводящая MAD к масштабу std для нормального распределения
    # (иначе MAD систематически занижает разброс по сравнению со std).
    stylo_median = np.median(stylo_raw, axis=0)
    stylo_mad = 1.4826 * np.median(np.abs(stylo_raw - stylo_median), axis=0)

    # --- внутренняя согласованность профиля: среднее попарное косинусное
    # сходство char-векторов собственных текстов кандидата (если текст один -
    # согласованность неопределена, считаем нейтральной 1.0, т.к. не с чем
    # сравнивать, а "мало текстов" уже отдельно ловится в quality-проверке) ---
    if len(dedup_texts) >= 2:
        char_dense = char_mat.toarray()
        sims = []
        for i in range(len(char_dense)):
            for j in range(i + 1, len(char_dense)):
                sims.append(_cosine(char_dense[i], char_dense[j]))
        consistency = float(np.mean(sims)) if sims else 1.0
    else:
        consistency = 1.0

    quality = assess_profile_quality(len(dedup_texts), total_words, word_counts, consistency)

    # ---- fw_z/pu_z: средний ПО ТЕКСТАМ ПРОФИЛЯ z-score частот служебных
    # слов/пунктуации (см. SIMILARITY_FEATURE_NAMES) - усредняем ПОСЛЕ
    # z-нормировки каждого текста отдельно (а не нормируем средние сырые
    # частоты), т.к. это устойчивее при разной длине текстов в профиле.
    fw_z_per_text = [(_function_word_freqs(t) - encoder.fw_mu) / encoder.fw_sd for t in dedup_texts]
    pu_z_per_text = [(_punct_freqs(t) - encoder.pu_mu) / encoder.pu_sd for t in dedup_texts]
    fw_z = np.mean(fw_z_per_text, axis=0)
    pu_z = np.mean(pu_z_per_text, axis=0)

    languages = [detect_script(t) for t in dedup_texts]
    non_english_share = float(np.mean([lg["likely_non_english"] for lg in languages])) if languages else 0.0

    return AuthorProfile(
        candidate_author_id=candidate_author_id,
        n_texts=len(texts),
        n_texts_deduplicated=len(dedup_texts),
        total_words=total_words,
        per_text_word_counts=word_counts,
        text_fingerprints=fingerprints,
        raw_texts=dedup_texts,
        char_mean=char_mean,
        word_mean=word_mean,
        stylo_mean=stylo_mean,
        stylo_std=stylo_std,
        stylo_median=stylo_median,
        stylo_mad=stylo_mad,
        stylo_feature_names=encoder.stylo_feature_names,
        fw_z=fw_z,
        pu_z=pu_z,
        consistency=consistency,
        quality=quality,
        language={"non_english_share": round(non_english_share, 3)},
        duplicates_removed=duplicates_removed,
    )


# =============================================================================
# СХОДСТВО текст <-> профиль (сырые, обезличенные признаки для верификатора)
# =============================================================================

def _stylo_delta_similarity(stylo_vec: np.ndarray, profile: AuthorProfile) -> float:
    """Burrows' Delta-подобное расстояние: средняя |z-оценка| нового текста
    относительно СОБСТВЕННОГО распределения профиля (а не глобального
    корпуса) - именно так классический Delta и определяется в стилометрии
    (Burrows, 2002), только вместо частот отдельных слов здесь используется
    уже существующий в проекте набор стилометрических признаков.

    Нормировка - МЕДИАНА/MAD, а не среднее/std: при малых профилях (1-3
    текста, самый слабый случай по D-разделу evaluate.py) среднее/std
    считаются по считанным точкам и легко искажаются одним нетипичным
    текстом, а медиана/MAD устойчивы к этому по построению. Когда MAD==0
    (типично при n_texts=1, где медиана совпадает с единственным
    наблюдением) - откатываемся на std, а если и он вырожден, на 1.0
    (нейтральная нормировка).

    Возвращает сходство (не расстояние): 1/(1+delta), т.е. 1.0 при полном
    совпадении, убывает к 0 при большом расхождении."""
    scale = np.where(profile.stylo_mad > 1e-6, profile.stylo_mad,
                      np.where(profile.stylo_std > 1e-6, profile.stylo_std, 1.0))
    z = np.abs(stylo_vec - profile.stylo_median) / scale
    delta = float(np.mean(z))
    return 1.0 / (1.0 + delta)


def similarity_features(new_text: str, profile: AuthorProfile,
                         encoder: StyleEncoder) -> dict:
    """Считает обезличенные признаки сходства текст<->профиль (см.
    SIMILARITY_FEATURE_NAMES) - ИМЕННО они (а не
    сырые векторы) идут дальше в PairwiseVerifier/эвристику. Компактность
    важна: это позволяет обучать верификатор на очень небольшом числе
    интерпретируемых измерений, что резко снижает риск переобучиться под
    конкретных 5 обучающих авторов (см. докстринг PairwiseVerifier).

    Тонкий враппер над encode_text()+similarity_features_from_encoded() -
    удобен, когда текст сравнивается только с ОДНИМ профилем (обычный
    продуктовый сценарий, см. compare_text_to_profile). Для сравнения
    ОДНОГО текста с НЕСКОЛЬКИМИ профилями (массовая оценка в evaluate.py)
    используйте encode_text() один раз и дальше
    similarity_features_from_encoded() - иначе дорогой char n-gram TF-IDF
    трансформ будет пересчитываться заново на каждую пару."""
    return similarity_features_from_encoded(encode_text(new_text, encoder), profile, encoder)


def encode_text(text: str, encoder: StyleEncoder) -> dict:
    """Кодирует текст ОДИН раз (самая дорогая часть - char n-gram TF-IDF)
    в промежуточное представление, пригодное для сравнения с ЛЮБЫМ числом
    профилей без повторного трансформа. См. similarity_features_from_encoded()."""
    char_vec = _mean_dense(encoder.transform_char([text]))
    word_vec = _mean_dense(encoder.transform_word([text]))
    stylo_vec = encoder.transform_stylo_raw([text])[0]
    stylo_scaled = encoder.stylo_scaler.transform(stylo_vec.reshape(1, -1))[0]
    fw_z = (_function_word_freqs(text) - encoder.fw_mu) / encoder.fw_sd
    pu_z = (_punct_freqs(text) - encoder.pu_mu) / encoder.pu_sd
    return {"char": char_vec, "word": word_vec, "stylo_raw": stylo_vec,
            "stylo_scaled": stylo_scaled, "fw_z": fw_z, "pu_z": pu_z,
            "n_words": len(text.split())}


def similarity_features_from_encoded(encoded: dict, profile: AuthorProfile,
                                      encoder: StyleEncoder) -> dict:
    sim_char = _cosine(encoded["char"], profile.char_mean)
    sim_word = _cosine(encoded["word"], profile.word_mean)

    stylo_scaled_profile_mean = encoder.stylo_scaler.transform(profile.stylo_mean.reshape(1, -1))[0]
    stylo_cosine = (_cosine(encoded["stylo_scaled"], stylo_scaled_profile_mean) + 1.0) / 2.0

    stylo_delta_sim = _stylo_delta_similarity(encoded["stylo_raw"], profile)

    # ---- fw/pu: те же две формы сравнения, что и для стилометрии выше -
    # косинус (общая форма профиля частот) и delta-подобная средняя |z|
    # разность (насколько КАЖДОЕ слово/символ типично для профиля) - см.
    # docstring выше про experiments/verifier_experiments.py.
    fw_cos = (_cosine(encoded["fw_z"], profile.fw_z) + 1.0) / 2.0
    fw_delta = 1.0 / (1.0 + float(np.mean(np.abs(encoded["fw_z"] - profile.fw_z))))
    pu_cos = (_cosine(encoded["pu_z"], profile.pu_z) + 1.0) / 2.0
    pu_delta = 1.0 / (1.0 + float(np.mean(np.abs(encoded["pu_z"] - profile.pu_z))))

    # ---- длина: log1p/log сглаживают распределение (слов может быть от
    # десятков до тысяч) - логрегрессия видит устойчивый, примерно линейный
    # диапазон вместо перекошенного хвоста сырых чисел слов.
    lq = math.log1p(encoded.get("n_words", 0))
    lp = math.log1p(profile.total_words)
    ln = math.log(max(1, profile.n_texts_deduplicated))

    feats = {
        "sim_char": sim_char, "sim_word": sim_word,
        "stylo_cosine": stylo_cosine, "stylo_delta_sim": stylo_delta_sim,
        "fw_cos": fw_cos, "fw_delta": fw_delta, "pu_cos": pu_cos, "pu_delta": pu_delta,
        "lq": lq, "lp": lp, "ln": ln,
    }
    # ---- взаимодействия: "насколько СИЛЬНО значение сходства должно влиять
    # на итоговый score" МЕНЯЕТСЯ в зависимости от того, сколько у нас слов
    # текста (lq) и слов/текстов профиля (ln) - короткий текст/маленький
    # профиль даёт зашумлённое сходство, которому логрегрессия, обученная на
    # РАЗНЫХ по размеру парах, учится доверять меньше именно через эти
    # произведения (без явного if/else по длине в коде - веса подбираются
    # обучением). Это и есть измеренный источник прироста (см. докстринг
    # SIMILARITY_FEATURE_NAMES выше) - интеракции важнее самих fw/pu.
    for base in ("sim_char", "sim_word", "stylo_delta_sim", "fw_delta", "fw_cos"):
        feats[f"{base}_x_lq"] = feats[base] * lq
        feats[f"{base}_x_ln"] = feats[base] * ln
    return feats


def _heuristic_combine(feats: dict) -> float:
    score = sum(HEURISTIC_WEIGHTS[k] * feats[k] for k in _HEURISTIC_FEATURE_NAMES)
    return float(np.clip(score, 0.0, 1.0))


def impostors_similarity(encoded: dict, profile: "AuthorProfile",
                          impostor_profiles: "list[AuthorProfile]",
                          n_iterations: int = 30, subsample_frac: float = 0.5,
                          rng: "random.Random | None" = None) -> float:
    """Метод самозванцев (Impostors method, Koppel & Winter 2014) -
    устойчивая ДОБАВКА к similarity_features(), особенно полезная именно
    там, где обычное сходство наименее надёжно - при МАЛЕНЬКИХ профилях
    (см. раздел D в evaluate.py: AUC=0.68 при n_texts=1 против 0.91 при
    n_texts=8). Идея: вместо ОДНОГО сравнения text<->profile на всём
    признаковом пространстве, делаем n_iterations сравнений на случайных
    ПОДВЫБОРКАХ признаков (subsample_frac от размерности char-пространства)
    против каждый раз СЛУЧАЙНО выбранного "самозванца" - профиля другого
    автора. score = доля итераций, где текст оказался ближе к НАСТОЯЩЕМУ
    профилю, чем к самозванцу. Многократное усреднение по случайным
    подпространствам сглаживает шум единственного (или нескольких)
    текстов в маленьком профиле - там, где обычный mean-вектор профиля
    ещё нестабилен, голосование по многим случайным проекциям уже
    устойчиво отличает "похоже" от "не похоже".

    Требует пул из ≥1 профилей-самозванцев. Он есть "бесплатно" везде, где
    уже есть несколько кандидатов: в train_verifier.py/evaluate.py - это
    другие известные авторы, в app.py - профили ДРУГИХ студентов по тому
    же предмету. Если пул пуст - возвращает нейтральные 0.5 (сигнала нет,
    ни в плюс, ни в минус), а не 0 или 1, чтобы не искажать score."""
    if not impostor_profiles:
        return 0.5
    rng = rng or random.Random()
    char_vec = encoded["char"]
    n_dims = len(char_vec)
    k = max(1, int(n_dims * subsample_frac))
    wins = 0
    for _ in range(n_iterations):
        idx = rng.sample(range(n_dims), k)
        impostor = rng.choice(impostor_profiles)
        sim_true = _cosine(char_vec[idx], profile.char_mean[idx])
        sim_impostor = _cosine(char_vec[idx], impostor.char_mean[idx])
        if sim_true > sim_impostor:
            wins += 1
    return wins / n_iterations


def score_similarity(feats: dict, verifier: "PairwiseVerifier | None" = None) -> float:
    """Единая точка входа text<->profile сходство -> итоговый score - и
    compare_text_to_profile, и evaluate.py используют именно эту функцию,
    чтобы формула комбинирования никогда не расходилась между продуктовым
    кодом и отчётом об оценке качества."""
    if verifier is not None and verifier.is_fitted:
        return verifier.predict_proba_same(feats)
    return _heuristic_combine(feats)


# =============================================================================
# PAIRWISE VERIFIER (обучаемая, но необязательная надстройка)
# =============================================================================

class PairwiseVerifier:
    """Логистическая регрессия (со StandardScaler - см. ниже) поверх
    SIMILARITY_FEATURE_NAMES из similarity_features(). Обучается на ПАРАХ
    (см. train_verifier.py): (текст_А, профиль_А) -> same, (текст_А,
    профиль_B) -> different. КРИТИЧЕСКИ ВАЖНО: этот класс
    не хранит и не видит ничего про то, КАКИЕ именно авторы были в
    обучающих парах - только числа сходства на пару. Поэтому на
    инференсе он одинаково применим к любому кандидату, независимо от
    того, встречался ли автор (или "похожий" на него) в обучающих данных -
    это и есть open-set свойство по построению, а не только по
    декларации.

    StandardScaler ОБЯЗАТЕЛЕН (в отличие от исходной версии с 4 признаками,
    которые все и так были в диапазоне ~[0,1]): признаки длины (lq/lp/ln) и
    особенно взаимодействия (sim_x_lq и т.п.) имеют совсем другой масштаб -
    без масштабирования логрегрессия либо игнорировала бы их, либо
    численно нестабильно подстраивалась бы только под них."""

    def __init__(self, C: float = 0.5):
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=5000),
        )
        self.is_fitted = False

    @staticmethod
    def _to_matrix(feats_list: list[dict]) -> np.ndarray:
        return np.asarray([[f[name] for name in SIMILARITY_FEATURE_NAMES] for f in feats_list])

    def fit(self, feats_list: list[dict], labels: list[int]) -> "PairwiseVerifier":
        X = self._to_matrix(feats_list)
        y = np.asarray(labels)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba_same(self, feats: dict) -> float:
        if not self.is_fitted:
            raise RuntimeError("PairwiseVerifier не обучен.")
        x = self._to_matrix([feats])
        return float(self.model.predict_proba(x)[0, 1])

    def predict_proba_same_batch(self, feats_list: list[dict]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PairwiseVerifier не обучен.")
        x = self._to_matrix(feats_list)
        return self.model.predict_proba(x)[:, 1]

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path, allowed_dir: Path | None = None) -> "PairwiseVerifier":
        if allowed_dir is not None:
            return safe_joblib_load(path, allowed_dir=allowed_dir)
        return joblib.load(path)


# =============================================================================
# COMPARE: текст <-> профиль -> итоговый результат верификации
# =============================================================================

def _length_band(n_words: int) -> str:
    if n_words >= 1500:
        return ">1500"
    if n_words >= 800:
        return "800-1500"
    if n_words >= 400:
        return "400-800"
    if n_words >= 200:
        return "200-400"
    return "<200"


# ---- Жёсткая abstain-политика по длине текста (см. SECURITY_AND_HARDENING_TODO.md,
# п.11 "Open-set verifier слаб на коротких текстах") -------------------------
# Измеренные цифры (evaluation_report.json, раздел E): ROC-AUC 0.69-0.75 и
# TPR@FAR=2% всего 12-31% на текстах короче 800 слов - т.е. верификатор на
# таких текстах существенно менее надёжен, чем на длинных, а его выдача до
# этой правки просто получала более низкий "confidence"-ярлык рядом с тем же
# MATCH/MISMATCH и числовым процентом - что легко проигнорировать (см.
# формулировку TODO буквально). Ниже - два РАЗНЫХ по жёсткости порога:
#   - < HARD_ABSTAIN_MIN_WORDS: verdict принудительно UNCERTAIN ("NO VERDICT"
#     в терминах TODO) - MATCH/MISMATCH на таком тексте вообще не выдаются,
#     независимо от того, насколько уверенно выглядит сырое сходство.
#   - < VERY_LOW_CONFIDENCE_MAX_WORDS: verdict остаётся обычным (MATCH/
#     MISMATCH/UNCERTAIN по порогам), но confidence принудительно "very_low" -
#     жёстко, а не как один из нескольких сигналов в _confidence_label(),
#     который мог быть перекрыт высокой уверенностью по марже/качеству
#     профиля.
HARD_ABSTAIN_MIN_WORDS = 200
VERY_LOW_CONFIDENCE_MAX_WORDS = 800


def _confidence_label(profile_quality_level: str, score: float,
                       threshold_match: float, threshold_mismatch: float,
                       length_band: str, by_length_metrics: dict | None) -> str:
    if profile_quality_level == "insufficient":
        return "low"
    # На сколько раскрыт "зазор" вокруг решения - чем ближе к порогу, тем ниже уверенность.
    if score >= threshold_match or score <= threshold_mismatch:
        margin_conf = "high"
    else:
        margin_conf = "low"
    # Честная по-масштабная надёжность (см. train_verifier.py/evaluate.py,
    # заполняющих VERIFIER_META["by_length"]) - аналог analysis.confidence_for_word_count,
    # но для верификатора, а не для старой 5-классовой модели.
    length_conf = None
    if by_length_metrics and length_band in by_length_metrics:
        auc = by_length_metrics[length_band].get("roc_auc")
        if auc is not None:
            length_conf = "high" if auc >= 0.85 else "medium" if auc >= 0.70 else "low"

    levels = {"high": 3, "medium": 2, "low": 1}
    candidates = [margin_conf] + ([length_conf] if length_conf else [])
    if profile_quality_level == "limited":
        candidates.append("medium")
    worst = min(candidates, key=lambda c: levels[c])
    return worst


def compare_text_to_profile(new_text: str, profile: AuthorProfile,
                             encoder: StyleEncoder,
                             verifier: PairwiseVerifier | None = None,
                             threshold_match: float = 0.70,
                             threshold_mismatch: float = 0.35,
                             by_length_metrics: dict | None = None,
                             impostor_pool: "list[AuthorProfile] | None" = None) -> dict:
    """ГЛАВНАЯ функция нового Style Check (см. INPUT/OUTPUT в тех.задании).

    ВАЖНО: НИКОГДА не делает predict([new_text]) -> один_из_N_известных_авторов.
    Всегда: score = сравнение(new_text, profile_КОНКРЕТНОГО_кандидата).

    impostor_pool (необязательно) - профили ДРУГИХ кандидатов (см.
    impostors_similarity()) для метода самозванцев - экспериментальная
    добавка, включаемая ТОЛЬКО когда пул реально передан (по умолчанию
    выключена, полностью обратно совместима). Вес её вклада АДАПТИВНО
    зависит от размера профиля: чем меньше предыдущих работ у кандидата
    (там, где обычное сходство наименее надёжно - см. evaluate.py, раздел
    D), тем больше веса получает голосование по случайным подпространствам
    признаков; для уже большого, устойчивого профиля вклад почти нулевой,
    т.к. там обычный score и так надёжен."""
    if not new_text or not new_text.strip():
        raise ValueError("compare_text_to_profile: new_text пуст.")

    fp = text_fingerprint(new_text)
    if fp in profile.text_fingerprints:
        raise ProfileLeakageError(
            "new_text - это тот же документ, что уже используется в Style "
            "Profile кандидата. Сравнение текста с профилем, построенным "
            "из него самого, недопустимо (см. п.4 'leakage prevention')."
        )

    n_words = len(new_text.split())
    length_band = _length_band(n_words)

    # Мягкая проверка на почти-дубликат (не блокирует, но снижает доверие -
    # напр. студент прислал ту же работу с парой правок под другим файлом).
    # Ограничиваем стоимость на очень больших профилях (Jaccard по shingles
    # для каждого текста профиля линеен по его длине).
    near_dup_max = 0.0
    for prev_text in profile.raw_texts[:50]:
        near_dup_max = max(near_dup_max, near_duplicate_ratio(new_text, prev_text))

    language = detect_script(new_text)

    encoded = encode_text(new_text, encoder)
    feats = similarity_features_from_encoded(encoded, profile, encoder)
    score = score_similarity(feats, verifier)
    method = "pairwise_verifier" if (verifier is not None and verifier.is_fitted) else "heuristic_weighted_similarity"

    impostor_sim = None
    impostor_blend_weight = 0.0
    if impostor_pool:
        impostor_sim = impostors_similarity(encoded, profile, impostor_pool)
        # Вес убывает с размером профиля: 0.45 при n_texts=1 -> ~0.05 при n_texts>=8.
        impostor_blend_weight = float(np.clip(0.5 - 0.05 * profile.n_texts_deduplicated, 0.05, 0.45))
        score = (1.0 - impostor_blend_weight) * score + impostor_blend_weight * impostor_sim
        method += "+impostors"

    quality_level = profile.quality.get("level", "insufficient")
    gating_reasons = []

    if n_words < HARD_ABSTAIN_MIN_WORDS:
        verdict = "UNCERTAIN"
        gating_reasons.append(
            f"проверяемый текст короче {HARD_ABSTAIN_MIN_WORDS} слов ({n_words}) - "
            f"жёсткий abstain по длине (см. evaluation_report.json, раздел E: "
            f"верификатор ненадёжен на текстах такой длины) - MATCH/MISMATCH не "
            f"выдаются независимо от значения сырого сходства"
        )
    elif quality_level == "insufficient":
        verdict = "UNCERTAIN"
        gating_reasons.append("профиль автора не соответствует минимальным требованиям (см. profile_quality)")
    elif language["likely_non_english"] or profile.language.get("non_english_share", 0) > 0.5:
        verdict = "UNCERTAIN"
        gating_reasons.append(
            "текст (или профиль) похож на не-английский по письменности - "
            "текущая модель обучена только на англоязычной прозе и не может "
            "давать содержательную оценку для ru/kk текста (см. README, "
            "раздел 'Ограничения')"
        )
    elif score >= threshold_match:
        verdict = "MATCH"
    elif score <= threshold_mismatch:
        verdict = "MISMATCH"
    else:
        verdict = "UNCERTAIN"

    confidence = _confidence_label(quality_level, score, threshold_match,
                                    threshold_mismatch, length_band, by_length_metrics)
    if n_words < VERY_LOW_CONFIDENCE_MAX_WORDS:
        # Жёсткая нижняя граница - не может быть перекрыта высокой уверенностью
        # по марже вокруг порога или по качеству профиля (см. докстринг
        # HARD_ABSTAIN_MIN_WORDS/VERY_LOW_CONFIDENCE_MAX_WORDS выше).
        confidence = "very_low"

    explanation = _build_explanation(verdict, score, quality_level, gating_reasons,
                                      n_words, length_band, profile, confidence)

    return {
        "candidate_author_id": profile.candidate_author_id,
        "style_similarity_score": round(score, 4),
        "style_similarity_percent": round(score * 100, 1),
        "verdict": verdict,
        "confidence": confidence,
        "threshold": {"match_min": threshold_match, "mismatch_max": threshold_mismatch,
                      "method": method},
        "number_of_profile_texts": profile.n_texts_deduplicated,
        "profile_quality": profile.quality,
        "evidence": {
            **{k: round(v, 4) for k, v in feats.items()},
            "profile_consistency": round(profile.consistency, 3),
            "new_text_word_count": n_words,
            "length_band": length_band,
            "language": language,
            "near_duplicate_jaccard_max": near_dup_max,
            "impostor_similarity": round(impostor_sim, 4) if impostor_sim is not None else None,
            "impostor_blend_weight": round(impostor_blend_weight, 3) if impostor_pool else None,
        },
        "explanation": explanation,
    }


def _build_explanation(verdict: str, score: float, quality_level: str,
                        gating_reasons: list[str], n_words: int,
                        length_band: str, profile: AuthorProfile,
                        confidence: str = "") -> str:
    pct = round(score * 100, 1)
    if gating_reasons:
        return (
            f"Оценка не может считаться доказательной: {'; '.join(gating_reasons)}. "
            f"Сырое сходство составило {pct}%, но verdict принудительно "
            f"понижен до UNCERTAIN, а не показан как MATCH/MISMATCH."
        )
    base = (f"Сходство новой работы со Style Profile кандидата (построен по "
            f"{profile.n_texts_deduplicated} предыдущим работам, "
            f"{profile.total_words} слов) составило {pct}%. "
            f"Длина проверяемого текста: {n_words} слов ({length_band}).")
    if confidence == "very_low":
        base += (
            f" ВНИМАНИЕ: текст короче {VERY_LOW_CONFIDENCE_MAX_WORDS} слов - "
            f"на текстах такой длины верификатор существенно менее надёжен "
            f"(см. evaluation_report.json, раздел E); не опирайтесь на один "
            f"процент выше как на достаточное основание, даже при verdict "
            f"MATCH/MISMATCH."
        )
    if verdict == "MATCH":
        return base + " Это соответствует уровню MATCH - стиль текста статистически похож на профиль."
    if verdict == "MISMATCH":
        return base + " Это ниже порога MISMATCH - стиль текста статистически отличается от профиля."
    return base + " Значение попадает в промежуточную зону между MATCH и MISMATCH - однозначного вывода нет."


# =============================================================================
# МЕТРИКИ КАЧЕСТВА ВЕРИФИКАТОРА (FAR/FRR/EER/TPR@FAR/AUC) - п.7 задания.
# Намеренно НЕ используем accuracy как основную метрику (см. докстринг
# ниже и README) - это open-set verification задача с сильным дисбалансом
# цены ошибки (ложное обвинение дороже пропуска), поэтому основной
# критерий - контролируемый FAR при максимально возможном TPR.
# =============================================================================

def far_frr_at_threshold(pos_scores: np.ndarray, neg_scores: np.ndarray,
                          threshold: float) -> tuple[float, float]:
    """FAR = доля НЕГАТИВНЫХ (разные авторы) пар, ошибочно принятых за
    совпадение (score >= threshold). FRR = доля ПОЗИТИВНЫХ (тот же автор)
    пар, ошибочно отклонённых (score < threshold)."""
    pos_scores = np.asarray(pos_scores)
    neg_scores = np.asarray(neg_scores)
    far = float(np.mean(neg_scores >= threshold)) if len(neg_scores) else float("nan")
    frr = float(np.mean(pos_scores < threshold)) if len(pos_scores) else float("nan")
    return far, frr


def eer(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate - точка, где FAR≈FRR. Возвращает (eer, порог)."""
    all_scores = np.unique(np.concatenate([np.asarray(pos_scores), np.asarray(neg_scores)]))
    best_gap, best_eer, best_thr = None, None, None
    for thr in all_scores:
        far, frr = far_frr_at_threshold(pos_scores, neg_scores, thr)
        gap = abs(far - frr)
        if best_gap is None or gap < best_gap:
            best_gap, best_eer, best_thr = gap, (far + frr) / 2.0, float(thr)
    return float(best_eer), float(best_thr)


def calibrate_threshold_for_far(neg_scores: np.ndarray, target_far: float) -> float:
    """Порог, при котором доля негативных пар со score >= порог не
    превышает target_far - считается ТОЛЬКО на validation-скорах (п.7
    задания: "рассчитывай threshold только на validation set"), никогда на
    train и никогда на итоговом test/held-out наборе, который используется
    для честного репортинга метрик."""
    neg = np.sort(np.asarray(neg_scores))
    if len(neg) == 0:
        return 0.5
    idx = int(np.ceil((1.0 - target_far) * len(neg)))
    idx = min(max(idx, 0), len(neg) - 1)
    return float(neg[idx])


def calibrate_threshold_for_low_genuine_percentile(pos_scores: np.ndarray,
                                                     percentile: float) -> float:
    """Порог MISMATCH - процентиль распределения ПОЗИТИВНЫХ (genuine) скоров
    на validation. Ниже этого порога попадает только небольшая, заранее
    заданная доля настоящих совпадений (например 5%) - значит, всё, что
    ниже, статистически уже похоже на "явно другой автор", а не просто
    "не дотянул до MATCH"."""
    pos = np.asarray(pos_scores)
    if len(pos) == 0:
        return 0.3
    return float(np.percentile(pos, percentile))


def tpr_at_far(pos_scores: np.ndarray, neg_scores: np.ndarray, target_far: float) -> float:
    thr = calibrate_threshold_for_far(neg_scores, target_far)
    _, frr = far_frr_at_threshold(pos_scores, neg_scores, thr)
    return 1.0 - frr


def roc_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    s = np.concatenate([np.asarray(pos_scores), np.asarray(neg_scores)])
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def pr_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    s = np.concatenate([np.asarray(pos_scores), np.asarray(neg_scores)])
    if len(set(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def full_metrics_report(pos_scores: np.ndarray, neg_scores: np.ndarray,
                         target_far: float, low_genuine_percentile: float = 5.0) -> dict:
    """Единая сводка метрик для одного среза оценки (п.7 + F. в evaluate.py) -
    HE accuracy, а полный набор FAR/FRR/EER/TPR@FAR/ROC-AUC/PR-AUC."""
    pos_scores = np.asarray(pos_scores)
    neg_scores = np.asarray(neg_scores)
    eer_value, eer_threshold = eer(pos_scores, neg_scores) if len(pos_scores) and len(neg_scores) else (float("nan"), float("nan"))
    thr_match = calibrate_threshold_for_far(neg_scores, target_far) if len(neg_scores) else float("nan")
    thr_mismatch = calibrate_threshold_for_low_genuine_percentile(pos_scores, low_genuine_percentile) if len(pos_scores) else float("nan")
    far_at_match, frr_at_match = far_frr_at_threshold(pos_scores, neg_scores, thr_match) if len(pos_scores) and len(neg_scores) else (float("nan"), float("nan"))
    return {
        "n_positive": int(len(pos_scores)),
        "n_negative": int(len(neg_scores)),
        "mean_positive_score": float(np.mean(pos_scores)) if len(pos_scores) else None,
        "mean_negative_score": float(np.mean(neg_scores)) if len(neg_scores) else None,
        "roc_auc": roc_auc(pos_scores, neg_scores),
        "pr_auc": pr_auc(pos_scores, neg_scores),
        "eer": eer_value,
        "eer_threshold": eer_threshold,
        f"tpr_at_far_{target_far}": tpr_at_far(pos_scores, neg_scores, target_far) if len(pos_scores) and len(neg_scores) else float("nan"),
        "threshold_match": thr_match,
        "threshold_mismatch": thr_mismatch,
        "far_at_match_threshold": far_at_match,
        "frr_at_match_threshold": frr_at_match,
    }


# =============================================================================
# ЗАГРУЗКА/СОХРАНЕНИЕ ГОТОВОГО ВЕРИФИКАТОРА (для app.py / api_analyze.py)
# =============================================================================

def load_verifier_artifacts(verifier_dir: Path) -> tuple[StyleEncoder | None, PairwiseVerifier | None, dict]:
    """Возвращает (encoder, verifier, meta) или (None, None, {}), если
    train_verifier.py ещё не запускался - вызывающий код (app.py) должен
    в этом случае откатиться на эвристическую формулу (verifier=None
    полностью поддерживается compare_text_to_profile) или явно сообщить,
    что профильная проверка пока недоступна."""
    import json
    enc_path = verifier_dir / ENCODER_FILE
    ver_path = verifier_dir / VERIFIER_FILE
    meta_path = verifier_dir / META_FILE
    if not enc_path.exists():
        return None, None, {}
    encoder = StyleEncoder.load(enc_path, allowed_dir=verifier_dir)
    verifier = PairwiseVerifier.load(ver_path, allowed_dir=verifier_dir) if ver_path.exists() else None
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    return encoder, verifier, meta
