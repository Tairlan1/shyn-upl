#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_features.py
=====================================================================
Sklearn-совместимый экстрактор признаков для классификатора
"человек / ИИ" (ai_detector.py). Помимо сырых TF-IDF признаков,
подмешивает те же поверхностные сигналы, что раньше считались вручную
в ai_heuristics.py (лексикон, плотность дискурсивных маркеров,
"перечисление из трёх", однородность длины предложений) - но теперь их
веса выучиваются логистической регрессией на размеченных примерах, а
не подбираются вручную.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import ai_heuristics
import preprocess_corpus as prep


class SurfaceFeaturizer(BaseEstimator, TransformerMixin):
    """Извлекает вектор числовых поверхностных признаков из текста."""

    FEATURE_NAMES = [
        "lexicon_signal", "discourse_signal", "list_of_three_signal",
        "uniformity_signal", "paragraph_uniformity_signal",
        "structural_markup_signal", "avg_sentence_len", "std_sentence_len",
        "avg_word_len", "type_token_ratio", "comma_per_100w",
        "semicolon_per_100w", "contraction_per_100w", "first_person_per_100w",
        "em_dash_per_100w",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [self._features(text) for text in X]
        return np.array(rows, dtype=float)

    @staticmethod
    def _features(text: str) -> list[float]:
        words = [w for w in text.split() if any(c.isalpha() for c in w)]
        n_words = max(1, len(words))
        sentences = prep.split_sentences(text)
        sent_lens = [len(s.split()) for s in sentences if s.strip()] or [0]

        ah = ai_heuristics.score_fragment(text)

        avg_sent_len = float(np.mean(sent_lens))
        std_sent_len = float(np.std(sent_lens))
        avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0

        lower_words = [w.lower().strip(".,;:!?\"'()") for w in words]
        ttr = len(set(lower_words)) / n_words

        commas = text.count(",")
        semis = text.count(";")
        contractions = len(re.findall(r"\b\w+'(?:t|re|ve|ll|d|s|m)\b", text, re.IGNORECASE))
        first_person = len(re.findall(r"\b(I|me|my|mine|we|us|our)\b", text))
        # Тире (em dash, "—"/"--") - задокументированный кросс-моделный
        # "tell" LLM-текста, не завязанный на конкретный лексикон.
        em_dashes = text.count("—") + len(re.findall(r"(?<!-)--(?!-)", text))

        return [
            ah["lexicon_signal"],
            ah["discourse_signal"],
            ah["list_of_three_signal"],
            ah["uniformity_signal"],
            ah["paragraph_uniformity_signal"],
            ah["structural_markup_signal"],
            avg_sent_len,
            std_sent_len,
            avg_word_len,
            ttr,
            100.0 * commas / n_words,
            100.0 * semis / n_words,
            100.0 * contractions / n_words,
            100.0 * first_person / n_words,
            100.0 * em_dashes / n_words,
        ]


class StructuralOnlyFeaturizer(SurfaceFeaturizer):
    """SurfaceFeaturizer БЕЗ лексикона/дискурс-маркеров (lexicon_signal,
    discourse_signal - см. ai_heuristics.py, явно задокументированы там
    как САМЫЕ модель-специфичные из шести эвристических сигналов).
    Оставляет только структурные/ритмические признаки (list-of-three,
    burstiness предложений/абзацев, markup, пунктуация, длины слов) -
    используется в train_ai_detector.py для честной проверки того,
    насколько работа модели держится именно на переносимых между LLM
    сигналах, а не на словарном "почерке" одной конкретной модели (см.
    compute_feature_group_ablation)."""
    DROP = {"lexicon_signal", "discourse_signal"}

    def transform(self, X):
        full = super().transform(X)
        keep_idx = [i for i, n in enumerate(self.FEATURE_NAMES) if n not in self.DROP]
        return full[:, keep_idx]
