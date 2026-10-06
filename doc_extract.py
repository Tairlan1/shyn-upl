#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
doc_extract.py
=====================================================================
Извлечение обычного текста из загруженного студентом файла (.docx,
.pdf, .txt) с сохранением разбиения на абзацы (пустая строка между
абзацами) - это важно для стилометрических признаков модели
(длина абзаца, доля диалога и т.д. считаются по абзацам).

БЕЗОПАСНОСТЬ (см. SECURITY_AND_HARDENING_TODO.md)
-------------------------------------------------------------------
п.7 - тип файла определяется НЕ по расширению из имени файла (его может
задать кто угодно), а по фактическим байтам содержимого (magic
bytes/сигнатура формата), ДО передачи в pdfplumber/python-docx. Файл,
названный "эссе.pdf", но не являющийся настоящим PDF, отклоняется с
понятной ошибкой, а не уходит прямо в парсер - см. UnsupportedFileError.

п.6 - жёсткие лимиты на размер входа (страницы PDF, абзацы DOCX, итоговое
число слов), чтобы один загруженный файл не мог положить процесс анализа
на неограниченное время/память. Это НЕ замена MAX_CONTENT_LENGTH на
уровне Flask (см. app.py/api_analyze.py - тот лимит нужен ДО того, как
байты вообще дойдут досюда), а вторая, независимая линия защиты - на
случай, если файл небольшой по размеру байт, но "раздутый" по структуре
(например PDF с тысячами почти пустых страниц).
"""

from __future__ import annotations

import io
import zipfile

# ---- Лимиты (см. докстринг выше, п.6) --------------------------------
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB - независимая проверка на
                                       # случай, если вызывающий код не
                                       # выставил Flask MAX_CONTENT_LENGTH
MAX_PDF_PAGES = 300
MAX_DOCX_PARAGRAPHS = 50_000
MAX_EXTRACTED_WORDS = 60_000  # с большим запасом относительно самой
                                # длинной ожидаемой студенческой работы
                                # (~1500-2000 слов) - защита от DoS, а не
                                # реалистичный лимит на курсовую работу


class UnsupportedFileError(ValueError):
    """Расширение файла не совпадает с его реальным содержимым (magic
    bytes), либо файл превышает допустимые лимиты размера/структуры -
    см. SECURITY_AND_HARDENING_TODO.md, пп. 6 и 7."""


def _sniff_kind(raw_bytes: bytes) -> str:
    """Определяет РЕАЛЬНЫЙ формат файла по первым байтам содержимого -
    не доверяя расширению из имени файла (его подделать тривиально)."""
    if raw_bytes.startswith(b"%PDF-"):
        return "pdf"
    if raw_bytes[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        # Сигнатура ZIP - DOCX (OOXML) это ZIP-контейнер с определённой
        # внутренней структурой. Проверяем не только "это вообще ZIP", но
        # и что внутри действительно word/-содержимое - иначе произвольный
        # .zip, переименованный в .docx, прошёл бы дальше в python-docx.
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                names = set(zf.namelist())
        except (zipfile.BadZipFile, OSError):
            return "unknown"
        if "[Content_Types].xml" in names and any(n.startswith("word/") for n in names):
            return "docx"
        return "unknown"
    return "text"  # нет распознанной бинарной сигнатуры - трактуем как
                    # обычный текст; финальная проверка - что он вообще
                    # декодируется (см. extract_text)


def extract_text(filename: str, raw_bytes: bytes) -> str:
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise UnsupportedFileError(
            f"Файл слишком большой ({len(raw_bytes) / (1024*1024):.1f} MB, "
            f"максимум {MAX_UPLOAD_BYTES // (1024*1024)} MB)."
        )

    name = (filename or "").lower()
    declared = (
        "docx" if name.endswith(".docx") else
        "pdf" if name.endswith(".pdf") else
        "text"
    )
    actual = _sniff_kind(raw_bytes)

    if declared in ("docx", "pdf") and actual != declared:
        expected_label = "DOCX" if declared == "docx" else "PDF"
        raise UnsupportedFileError(
            f"Файл называется .{declared}, но по фактическому содержимому "
            f"это не {expected_label}-документ (расширение имени файла не "
            f"является гарантией его реального формата). Загрузите настоящий "
            f"файл в заявленном формате или вставьте текст напрямую."
        )

    if declared == "docx":
        text = _extract_docx(raw_bytes)
    elif declared == "pdf":
        text = _extract_pdf(raw_bytes)
    else:
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("utf-8", errors="replace")

    n_words = len(text.split())
    if n_words > MAX_EXTRACTED_WORDS:
        raise UnsupportedFileError(
            f"В файле обнаружено {n_words} слов - это больше допустимого "
            f"лимита ({MAX_EXTRACTED_WORDS}). Загрузите работу меньшего "
            f"объёма."
        )
    return text


def _extract_docx(raw_bytes: bytes) -> str:
    import docx  # python-docx

    document = docx.Document(io.BytesIO(raw_bytes))
    if len(document.paragraphs) > MAX_DOCX_PARAGRAPHS:
        raise UnsupportedFileError(
            f"Документ содержит {len(document.paragraphs)} абзацев - это "
            f"больше допустимого лимита ({MAX_DOCX_PARAGRAPHS})."
        )
    paragraphs = [p.text.strip() for p in document.paragraphs]
    paragraphs = [p for p in paragraphs if p]
    return "\n\n".join(paragraphs)


def _extract_pdf(raw_bytes: bytes) -> str:
    import pdfplumber

    pages_text = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        if len(pdf.pages) > MAX_PDF_PAGES:
            raise UnsupportedFileError(
                f"PDF содержит {len(pdf.pages)} страниц - это больше "
                f"допустимого лимита ({MAX_PDF_PAGES})."
            )
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text.strip())
    # pdfplumber обычно сохраняет одиночные переводы строк внутри абзаца
    # и не всегда ставит пустую строку между абзацами - нормализуем так,
    # чтобы соседние строки одного абзаца схлопывались, а разрывы страниц
    # трактовались как границы абзацев.
    normalized = []
    for page_text in pages_text:
        lines = [ln.strip() for ln in page_text.split("\n")]
        lines = [ln for ln in lines if ln]
        normalized.append(" ".join(lines))
    return "\n\n".join(normalized)
