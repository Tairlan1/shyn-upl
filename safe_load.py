#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
safe_load.py
=====================================================================
См. SECURITY_AND_HARDENING_TODO.md, п.24:

    "joblib.load() — исполняемый артефакт (pickle). Пока модели только
    локальные — не риск. Станет риском, если появится model registry/
    remote update/admin upload. Нужно: держать в уме при добавлении
    любого механизма загрузки моделей извне; не давать такой
    функциональности без sandboxing."

joblib.load() - это, по сути, pickle.load() под капотом: файл может
содержать произвольный код, который выполнится в момент загрузки (через
__reduce__ у любого объекта в пикле), а не только "данные модели". Сегодня
все joblib.load() в проекте (train_model.py, ai_detector.py,
author_verification.py) грузят файлы ТОЛЬКО из локальных директорий
(./model, ./model_multiscale, ./model_ai_detector, ./model_verifier),
заданных доверенным оператором (CLI-флаг/config.py) - НЕ файлы, которые
загрузил через веб-форму студент/учитель. Поэтому это НЕ уязвимость
сегодня (см. текст TODO выше - согласны с этой оценкой).

Эта функция - защита "на будущее, по умолчанию": если КОГДА-НИБУДЬ
появится механизм вроде "учитель загружает свою модель" или "подтянуть
модель из S3/registry по имени, пришедшему в запросе", safe_joblib_load()
не даст прочитать файл ВНЕ ожидаемой директории моделей (например,
../../etc/passwd или абсолютный путь, полученный из непроверенного
пользовательского ввода) - без этого добавление такой функциональности
было бы на один шаг ближе к произвольному выполнению кода при загрузке.

Это НЕ песочница/sandboxing для содержимого самого файла (см. текст TODO -
"не давать такой функциональности без sandboxing" - т.е. если/когда
появится реальная загрузка моделей извне, одной этой проверки пути
НЕДОСТАТОЧНО - нужна отдельная задача на sandboxing самого процесса
десериализации, например через отдельный непривилегированный
процесс/контейнер). Здесь закрывается только один конкретный, дешёвый в
реализации вектор (обход директории/абсолютный путь), а не общий риск
произвольного pickle.
"""

from __future__ import annotations

from pathlib import Path

import joblib


class UntrustedModelPathError(ValueError):
    """Путь к joblib-артефакту лежит вне ожидаемой директории моделей -
    см. докстринг модуля выше."""


def safe_joblib_load(path: Path, allowed_dir: Path):
    """joblib.load(path), но только если path физически лежит ВНУТРИ
    allowed_dir (после resolve() - т.е. с учётом ../ и симлинков) - иначе
    UntrustedModelPathError до вызова joblib.load(), а не после."""
    resolved_path = Path(path).resolve()
    resolved_allowed_dir = Path(allowed_dir).resolve()
    if resolved_allowed_dir not in resolved_path.parents and resolved_path != resolved_allowed_dir:
        raise UntrustedModelPathError(
            f"Отказ загружать модель: {resolved_path} лежит вне ожидаемой "
            f"директории моделей {resolved_allowed_dir} (см. safe_load.py, "
            f"SECURITY_AND_HARDENING_TODO.md п.24)."
        )
    return joblib.load(resolved_path)
