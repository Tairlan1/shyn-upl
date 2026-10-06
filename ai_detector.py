#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_detector.py
=====================================================================
Инференс обученного классификатора "человек / ИИ" (см.
train_ai_detector.py для методологии и честной оценки ограничений).
Загружает model_ai_detector/ai_detector_pipeline.joblib один раз и
переиспользует для всех последующих вызовов.

Публичный интерфейс совместим с ai_heuristics.score_fragment(text) -
оба возвращают dict с ключом "ai_score" (float 0..1) - что позволяет
app.py использовать обученную модель как основной источник, а
ai_heuristics как резерв/источник подсвечиваемых маркеров.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from safe_load import safe_joblib_load

_PIPELINE = None
_META = None
MIN_WORDS_FOR_MODEL = 40


def _model_dir(project_dir: Path) -> Path:
    return project_dir / "model_ai_detector"


def is_available(project_dir: Path) -> bool:
    return (_model_dir(project_dir) / "ai_detector_pipeline.joblib").exists()


def load(project_dir: Path):
    global _PIPELINE, _META
    if _PIPELINE is not None:
        return _PIPELINE

    import json
    model_dir = _model_dir(project_dir)
    pipeline_path = model_dir / "ai_detector_pipeline.joblib"
    meta_path = model_dir / "ai_detector_meta.json"
    # safe_joblib_load, не joblib.load напрямую - см. safe_load.py,
    # SECURITY_AND_HARDENING_TODO.md, п.24.
    _PIPELINE = safe_joblib_load(pipeline_path, allowed_dir=model_dir)
    if meta_path.exists():
        _META = json.loads(meta_path.read_text(encoding="utf-8"))
    return _PIPELINE


def get_meta(project_dir: Path) -> dict | None:
    load(project_dir)
    return _META


def score_fragment(text: str, project_dir: Path) -> dict | None:
    """Возвращает {"ai_score": float, "source": "trained_classifier"} либо
    None, если модель недоступна или фрагмент слишком короткий (в таком
    случае вызывающий код app.py откатывается на ai_heuristics)."""
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if len(words) < MIN_WORDS_FOR_MODEL:
        return None
    if not is_available(project_dir):
        return None

    pipeline = load(project_dir)
    proba = pipeline.predict_proba([text])[0]
    # класс 1 = "AI" (см. train_ai_detector.py: y = 1 для AI, 0 для человека)
    ai_proba = float(proba[1])

    return {"ai_score": ai_proba, "source": "trained_classifier"}
