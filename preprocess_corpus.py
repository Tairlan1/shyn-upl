#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_corpus.py
=====================================================================
Готовит корпус художественных текстов (Project Gutenberg .txt) к
обучению модели атрибуции авторства.

Делает для каждого файла в /data/<Author>/*.txt:

  1. Надёжное чтение файла (авто-детект кодировки).
  2. Удаление служебных данных Project Gutenberg (шапка/подвал,
     лицензия, "Produced by", транскрайберские пометки).
  3. Удаление оглавления (Contents/Table of Contents), Appendix,
     Index, Glossary, Footnotes/Notes, если они являются "хвостом"
     книги, а не художественным текстом.
  4. Удаление сносок, вставок переводчика/редактора
     ("[Footnote ...]", "[Translator's note: ...]" и т.п.),
     нумерации сносок вида [12].
  5. Починку типографских артефактов OCR/печати: перенос слова по
     слогам на конце строки ("charac-\nter" -> "character"),
     склейку строк одного абзаца, нормализацию кавычек/тире/пробелов.
  6. Контроль качества: детектор "мусорных" OCR-строк, детектор
     не-английского текста, детектор дублей/почти-дублей между
     файлами (в том числе внутри одного автора).
  7. Разбиение вычищенного текста на смысловые окна ~1500 слов
     (1450-1550, мягкий потолок 1650 в крайних случаях), не разрывая
     предложения и абзацы, с приоритетом на границы глав и с явным
     избеганием начала/конца окна репликой диалога.
  8. Экспорт готового датасета в JSONL + подробные отчёты о качестве.

Использование:
    python preprocess_corpus.py \
        --input-dir /data \
        --output-dir /data_processed \
        --target-words 1500 --min-words 1450 --max-words 1550

Результат в output-dir:
    dataset.jsonl            - основной ML-датасет (по одному чанку в строке)
    quality_report.json      - метрики качества по каждой книге
    duplicates_report.json   - найденные дубликаты/почти-дубликаты
    preprocessing.log        - подробный лог обработки
    chunks/<Author>/*.txt    - опционально, чанки как отдельные файлы
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Необязательные зависимости: скрипт обязан работать даже без них.
# ---------------------------------------------------------------------------
try:
    import chardet  # более точный детект кодировки
    _HAS_CHARDET = True
except ImportError:
    _HAS_CHARDET = False

try:
    import nltk  # noqa
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    try:
        nltk.data.find("tokenizers/punkt")
        _HAS_NLTK_PUNKT = True
    except LookupError:
        try:
            nltk.data.find("tokenizers/punkt_tab")
            _HAS_NLTK_PUNKT = True
        except LookupError:
            _HAS_NLTK_PUNKT = False
except ImportError:
    _HAS_NLTK_PUNKT = False


LOG = logging.getLogger("preprocess_corpus")


# =============================================================================
# 1. ЧТЕНИЕ ФАЙЛОВ
# =============================================================================

def read_text_file(path: Path) -> str:
    """Читает текстовый файл, надёжно определяя кодировку."""
    raw = path.read_bytes()

    encodings_to_try: list[str] = []
    if _HAS_CHARDET:
        guess = chardet.detect(raw)
        if guess and guess.get("encoding"):
            encodings_to_try.append(guess["encoding"])
    encodings_to_try += ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

    seen = set()
    for enc in encodings_to_try:
        enc_l = enc.lower()
        if enc_l in seen:
            continue
        seen.add(enc_l)
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # Последний рубеж: latin-1 не бросает исключений никогда.
    return raw.decode("latin-1", errors="replace")


# =============================================================================
# 2. УДАЛЕНИЕ СЛУЖЕБНЫХ ДАННЫХ PROJECT GUTENBERG
# =============================================================================

_GUTENBERG_START_RE = re.compile(
    r"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*{3}",
    re.IGNORECASE,
)
_GUTENBERG_END_RE = re.compile(
    r"\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*{3}",
    re.IGNORECASE,
)
# Старый формат ебуков (до ~2000-х годов)
_GUTENBERG_OLD_END_RE = re.compile(
    r"^\s*End of (?:the )?Project Gutenberg('s)? Etext.*$",
    re.IGNORECASE | re.MULTILINE,
)
_SMALL_PRINT_RE = re.compile(
    r"\*END\*THE SMALL PRINT.*?(?=\n\n)", re.IGNORECASE | re.DOTALL
)

_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^\s*Produced by .*$", re.IGNORECASE),
    re.compile(r"^\s*Transcriber('|’)s Note[:s]*.*$", re.IGNORECASE),
    re.compile(r"^\s*\[Transcriber('|’)s Note.*?\]\s*$", re.IGNORECASE),
    re.compile(r"^\s*This (?:e[Bb]ook|etext) is for the use of anyone.*$", re.IGNORECASE),
    re.compile(r"^\s*Updated editions will replace.*$", re.IGNORECASE),
    re.compile(r"^\s*Creating the works from (?:public domain|print).*$", re.IGNORECASE),
    re.compile(r"^\s*www\.gutenberg\.(org|net).*$", re.IGNORECASE),
    re.compile(r"^\s*Title:\s.*$", re.IGNORECASE),
    re.compile(r"^\s*Author:\s.*$", re.IGNORECASE),
    re.compile(r"^\s*Release [Dd]ate:\s.*$", re.IGNORECASE),
    re.compile(r"^\s*Language:\s.*$", re.IGNORECASE),
    re.compile(r"^\s*Character set encoding:\s.*$", re.IGNORECASE),
    re.compile(r"^\s*Credits?:\s.*$", re.IGNORECASE),
    re.compile(r"^\s*eBook No\.?:.*$", re.IGNORECASE),
    re.compile(r"^\s*Most recently updated:.*$", re.IGNORECASE),
    re.compile(r"^\s*Project Gutenberg('|’)s .*, by .*$", re.IGNORECASE),
]


def strip_gutenberg_boilerplate(text: str) -> str:
    """Отрезает лицензионную шапку/подвал Project Gutenberg."""
    start_m = _GUTENBERG_START_RE.search(text)
    end_m = _GUTENBERG_END_RE.search(text)

    if start_m and end_m and end_m.start() > start_m.end():
        text = text[start_m.end():end_m.start()]
    elif start_m:
        text = text[start_m.end():]
        # старый подвал без "***"
        old_end = _GUTENBERG_OLD_END_RE.search(text)
        if old_end:
            text = text[: old_end.start()]
    else:
        # Файл без стандартных маркеров - попробуем старый small-print формат
        text = _SMALL_PRINT_RE.sub("", text)
        old_end = _GUTENBERG_OLD_END_RE.search(text)
        if old_end:
            text = text[: old_end.start()]

    lines = text.split("\n")
    cleaned_lines = [
        ln for ln in lines
        if not any(p.match(ln) for p in _BOILERPLATE_LINE_PATTERNS)
    ]
    return "\n".join(cleaned_lines)


# =============================================================================
# 3. ОГЛАВЛЕНИЕ / APPENDIX / INDEX / GLOSSARY / FOOTNOTES-БЛОКИ
# =============================================================================

_CHAPTER_HEADING_RE = re.compile(
    r"^\s*(CHAPTER|Chapter|BOOK|Book|PART|Part|PROLOGUE|Prologue|"
    r"EPILOGUE|Epilogue|VOLUME|Volume)\b.{0,60}$"
)
_ROMAN_HEADING_RE = re.compile(r"^\s*[IVXLCDM]{1,8}\.?\s*$")
_BACKMATTER_HEADINGS = re.compile(
    r"^\s*(APPENDIX|GLOSSARY|INDEX|FOOTNOTES?|NOTES?|BIBLIOGRAPHY|"
    r"ACKNOWLEDGEMENTS?)\s*\.?\s*$",
    re.IGNORECASE,
)
_TOC_HEADING_RE = re.compile(
    r"^\s*(CONTENTS|TABLE OF CONTENTS)\s*\.?\s*$", re.IGNORECASE
)


_TOC_ENTRY_LEADER_RE = re.compile(r"\.{2,}|\s{2,}\d{1,4}\s*$")


def remove_table_of_contents(text: str) -> str:
    """Удаляет блок оглавления, если он обнаружен в начале книги.

    Идём вперёд от заголовка CONTENTS, пропуская пустые строки, пока не
    найдём первую настоящую "прозаическую" строку (длинную, это уже текст
    главы). Затем аккуратно возвращаемся назад максимум на один настоящий
    заголовок главы (не являющийся строкой-пунктом оглавления с точками-
    лидерами/номером страницы), чтобы не потерять сам заголовок главы.
    """
    lines = text.split("\n")
    toc_idx = None
    for i, ln in enumerate(lines[:2000]):  # оглавление всегда в начале файла
        if _TOC_HEADING_RE.match(ln):
            toc_idx = i
            break
    if toc_idx is None:
        return text

    prose_idx = None
    j = toc_idx + 1
    while j < len(lines):
        s = lines[j].strip()
        if not s:
            j += 1
            continue
        if len(s.split()) >= 20 or len(s) >= 150:
            prose_idx = j
            break
        j += 1

    if prose_idx is None:
        # Не нашли явного начала прозы - удаляем только саму строку CONTENTS,
        # остальное решит remove_backmatter/ручная проверка.
        return "\n".join(lines[:toc_idx] + lines[toc_idx + 1:])

    heading_start = prose_idx
    pulled_heading = False
    k = prose_idx - 1
    while k > toc_idx:
        s = lines[k].strip()
        if not s:
            heading_start = k
            k -= 1
            continue
        is_heading_like = bool(_CHAPTER_HEADING_RE.match(s) or _ROMAN_HEADING_RE.match(s))
        is_toc_entry = bool(_TOC_ENTRY_LEADER_RE.search(s))
        if not pulled_heading and is_heading_like and not is_toc_entry:
            heading_start = k
            pulled_heading = True
            k -= 1
            continue
        break

    return "\n".join(lines[:toc_idx] + lines[heading_start:])


def remove_backmatter(text: str) -> str:
    """Убирает Appendix/Index/Glossary/Footnotes/Notes, если это хвост книги
    (после этой метки в тексте больше не встречаются главы)."""
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if _BACKMATTER_HEADINGS.match(ln.strip()):
            remainder = "\n".join(lines[i + 1:])
            if not _CHAPTER_HEADING_RE.search(remainder):
                return "\n".join(lines[:i])
    return text


# =============================================================================
# 4. СНОСКИ, ПРИМЕЧАНИЯ РЕДАКТОРА/ПЕРЕВОДЧИКА
# =============================================================================

_INLINE_FOOTNOTE_MARK_RE = re.compile(r"\[\s*\d{1,3}\s*\]")
_FOOTNOTE_BLOCK_RE = re.compile(
    r"\[\s*Footnote[^\]]*\]", re.IGNORECASE | re.DOTALL
)
_EDITOR_TRANSLATOR_NOTE_RE = re.compile(
    r"\[[^\]]{0,300}?(translator|editor|transcriber)[^\]]{0,300}?\]",
    re.IGNORECASE | re.DOTALL,
)


def remove_notes_and_footnotes(text: str) -> str:
    text = _FOOTNOTE_BLOCK_RE.sub(" ", text)
    text = _EDITOR_TRANSLATOR_NOTE_RE.sub(" ", text)
    text = _INLINE_FOOTNOTE_MARK_RE.sub("", text)
    return text


# =============================================================================
# 5. ТИПОГРАФСКИЕ АРТЕФАКТЫ / НОРМАЛИЗАЦИЯ
# =============================================================================

_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*\d{1,4}\s*$")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def fix_typography(text: str) -> str:
    """Чинит перенос слов по слогам на конце строки и убирает висящие
    номера страниц, оставляя реальные абзацы нетронутыми."""
    text = unicodedata.normalize("NFC", text)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    lines = [ln for ln in text.split("\n") if not _PAGE_NUMBER_LINE_RE.match(ln)]
    text = "\n".join(lines)

    # нормализация кавычек/тире к единому виду (важно для детекта диалога
    # и для того, чтобы разные издания не создавали ложный стилевой сигнал)
    replacements = {
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
        "\u2018": "'", "\u2019": "'", "\u2032": "'",
        "\u2013": "-", "\u2014": "--", "\u2015": "--",
        "\ufeff": "", "\u00a0": " ",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)

    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def join_paragraph_lines(text: str) -> list[str]:
    """Разбивает текст на абзацы (по пустой строке) и склеивает внутренние
    переносы строк каждого абзаца в одну строку."""
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = []
    for p in raw_paragraphs:
        joined = " ".join(line.strip() for line in p.split("\n") if line.strip())
        joined = _MULTI_SPACE_RE.sub(" ", joined).strip()
        if joined:
            paragraphs.append(joined)
    return paragraphs


# =============================================================================
# 6. КОНТРОЛЬ КАЧЕСТВА: OCR, ЯЗЫК, ДУБЛИКАТЫ
# =============================================================================

_COMMON_ENGLISH_WORDS = set("""
the of and a to in is was he that it for you his with as had her not on at
by have be i this but from they we she or an will my one all would there
their what so up out if about who get which go me when make can like time
no just him know take people into year your good some could them see other
than then now look only come its over think also back after use two how our
work first well way even new want because any these give day most us
""".split())


def compute_ocr_quality_score(text: str) -> dict:
    """Эвристическая оценка «мусорности» текста (артефакты OCR/сканирования)."""
    tokens = re.findall(r"[A-Za-z']+", text)
    if not tokens:
        return {"ocr_score": 1.0, "suspicious_token_ratio": 1.0,
                "common_word_ratio": 0.0, "n_tokens": 0}

    n = len(tokens)
    suspicious = 0
    for t in tokens:
        if len(t) >= 4 and not re.search(r"[aeiouAEIOU]", t):
            suspicious += 1  # длинное слово без гласных - похоже на OCR-мусор
        elif re.search(r"[A-Z][a-z]*[A-Z]", t) and len(t) > 3:
            suspicious += 1  # рандомная смена регистра внутри слова
    suspicious_ratio = suspicious / n

    common_hits = sum(1 for t in tokens if t.lower() in _COMMON_ENGLISH_WORDS)
    common_ratio = common_hits / n

    non_letter_ratio = 1 - (len(re.findall(r"[A-Za-z]", text)) /
                            max(1, len(text)))

    # итоговый скор: 1.0 = отлично, 0.0 = очень плохо
    ocr_score = max(0.0, min(1.0,
        1.0 - 3 * suspicious_ratio - max(0, 0.25 - common_ratio) * 2
    ))
    return {
        "ocr_score": round(ocr_score, 4),
        "suspicious_token_ratio": round(suspicious_ratio, 4),
        "common_word_ratio": round(common_ratio, 4),
        "non_letter_ratio": round(non_letter_ratio, 4),
        "n_tokens": n,
    }


def shingles(text: str, k: int = 50) -> set:
    """k-словные шинглы для оценки почти-дублей (Jaccard)."""
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(0, len(words) - k + 1, k // 2 or 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# =============================================================================
# 7. РАЗБИЕНИЕ НА ПРЕДЛОЖЕНИЯ / СТРУКТУРНЫЕ ЕДИНИЦЫ
# =============================================================================

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "vs", "etc", "mssrs", "jr", "sr", "prof",
    "col", "gen", "capt", "lieut", "rev", "hon", "esq", "no", "vol", "pp",
    "cf", "i.e", "e.g", "a.m", "p.m", "u.s", "u.k",
}
# ВАЖНО (см. SECURITY_AND_HARDENING_TODO.md, раздел про русский/казахский):
# раньше эта регулярка требовала ЛАТИНСКУЮ заглавную букву ([A-Z]) сразу
# после знака конца предложения - для кириллического текста такая буква
# никогда не встречается, поэтому ВЕСЬ абзац (даже в сотни слов) склеивался
# в одно "предложение". Это не косметическая мелочь: avg_sentence_len -
# признак, на который напрямую опирается ai_features.SurfaceFeaturizer
# (см. train_ai_detector.py) - ошибочное значение в сотню с лишним слов
# полностью ломает признак и уводит классификатор в произвольную сторону.
# Кириллические заглавные (А-ЯЁ + казахские Ә Ғ Қ Ң Ө Ұ Ү Һ І) добавлены
# явно, а не через \w/\p{Lu} (в stdlib re они недоступны без модуля regex).
_SENT_SPLIT_RE = re.compile(
    r'(?<=[.!?])(?=["\')\]]*\s+[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ"\'\u201c])'
)

# Доля кириллических букв, начиная с которой считаем текст "не английским"
# для ЦЕЛЕЙ ВЫБОРА СПЛИТТЕРА (см. split_sentences ниже) - то же пороговое
# значение и та же логика, что в config.CYRILLIC_RATIO_NON_ENGLISH /
# author_verification.detect_script, НЕ импортируется отсюда напрямую (во
# избежание цикла preprocess_corpus -> author_verification -> preprocess_corpus),
# поэтому минимальная проверка продублирована здесь - если поменяете порог
# в config.py, поменяйте и здесь.
_CYRILLIC_RE_LOCAL = re.compile(r"[\u0400-\u04FF]")
_CYRILLIC_RATIO_THRESHOLD = 0.15


def _looks_cyrillic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    cyr = sum(1 for c in letters if _CYRILLIC_RE_LOCAL.match(c))
    return (cyr / len(letters)) >= _CYRILLIC_RATIO_THRESHOLD


def split_sentences(text: str) -> list[str]:
    """Разбивает абзац на предложения. Использует NLTK punkt, если он
    установлен и содержит нужные данные, иначе — устойчивый regex-сплиттер,
    который не режет по сокращениям (Mr., Dr., St. и т.д.).

    ВАЖНО: для кириллического текста (см. _looks_cyrillic) NLTK punkt
    ПРИНЦИПИАЛЬНО НЕ используется, даже если установлен - nltk.sent_tokenize
    без явного языка грузит английскую модель (её пунктуационные/
    аббревиатурные эвристики не валидировались для русского/казахского),
    а готовой предобученной punkt-модели для ru/kk в стандартной поставке
    NLTK нет вообще. Вместо того чтобы молча применять непроверенную для
    этого языка модель, используем regex-фолбэк ниже (после правки выше он
    хотя бы корректно видит кириллические заглавные буквы как границу
    предложения) - это более предсказуемо и честно, чем статистическая
    модель, которая никогда не тестировалась на этом языке."""
    if _HAS_NLTK_PUNKT and not _looks_cyrillic(text):
        try:
            return [s.strip() for s in _nltk_sent_tokenize(text) if s.strip()]
        except Exception:
            pass

    # regex-фолбэк
    candidates = _SENT_SPLIT_RE.split(text)
    sentences: list[str] = []
    buf = ""
    for piece in candidates:
        buf = (buf + " " + piece).strip() if buf else piece
        last_word = re.findall(r"([A-Za-z]+)\.\s*$", buf)
        if last_word and last_word[-1].lower() in _ABBREVIATIONS:
            continue  # это было сокращение, не конец предложения - продолжаем копить
        sentences.append(buf)
        buf = ""
    if buf:
        sentences.append(buf)
    return [s for s in sentences if s.strip()]


def is_dialogue_paragraph(paragraph: str) -> bool:
    s = paragraph.strip()
    return bool(s) and s[0] in ('"', "'", "\u201c", "\u2018")


def is_dialogue_ending(paragraph: str) -> bool:
    s = paragraph.strip()
    return bool(s) and s[-1] in ('"', "'", "\u201d", "\u2019")


# =============================================================================
# СТРУКТУРА ДАННЫХ: АБЗАЦ / ГЛАВА
# =============================================================================

@dataclass
class Paragraph:
    text: str
    word_count: int
    is_dialogue_start: bool
    is_dialogue_end: bool
    chapter_id: int
    is_chapter_boundary: bool = False  # True для первого абзаца новой главы


def build_paragraph_index(text: str) -> list[Paragraph]:
    """Строит абзацы с разметкой глав (по эвристике заголовков глав)."""
    paragraphs_raw = join_paragraph_lines(text)
    result: list[Paragraph] = []
    chapter_id = 0
    for p in paragraphs_raw:
        is_heading = bool(_CHAPTER_HEADING_RE.match(p)) and len(p.split()) <= 10
        if is_heading:
            chapter_id += 1
            continue  # заголовок главы не идёт в текст как отдельный абзац
        wc = len(p.split())
        if wc == 0:
            continue
        para = Paragraph(
            text=p,
            word_count=wc,
            is_dialogue_start=is_dialogue_paragraph(p),
            is_dialogue_end=is_dialogue_ending(p),
            chapter_id=chapter_id,
        )
        if result and result[-1].chapter_id != para.chapter_id:
            para.is_chapter_boundary = True
        elif not result:
            para.is_chapter_boundary = True
        result.append(para)
    return result


# =============================================================================
# 8. ЧАНКИНГ (ОКНА ~1500 СЛОВ)
# =============================================================================

@dataclass
class ChunkingConfig:
    target_words: int = 1500
    min_words: int = 1450
    max_words: int = 1550
    hard_max_words: int = 1700       # крайний случай, если нет хорошей точки разреза
    min_final_chunk: int = 600       # слишком маленький последний чанк лучше слить с предыдущим


def _score_cut(acc_words: int, cfg: ChunkingConfig,
               next_para: Paragraph | None, last_para: Paragraph,
               is_chapter_start: bool) -> float:
    """Чем выше скор, тем лучше точка разреза после текущего абзаца."""
    score = 0.0
    score -= abs(acc_words - cfg.target_words) / 50.0
    if is_chapter_start:
        score += 5.0
    if last_para.is_dialogue_end:
        score -= 2.0  # не хотим обрывать окно на реплике диалога
    if next_para is not None and next_para.is_dialogue_start:
        score -= 2.0  # не хотим начинать окно с реплики диалога
    return score


def chunk_paragraphs(paragraphs: list[Paragraph],
                      cfg: ChunkingConfig) -> list[dict]:
    """Жадно собирает абзацы в окна ~target_words слов, предпочитая границы
    глав и избегая начала/конца окна на диалоге."""
    chunks: list[dict] = []
    i = 0
    n = len(paragraphs)
    while i < n:
        acc: list[Paragraph] = []
        acc_words = 0
        best_cut = None  # индекс (включительно) последнего абзаца лучшего разреза
        best_score = float("-inf")
        j = i
        # Единственный абзац крупнее hard_max - разрежем по предложениям.
        while j < n:
            p = paragraphs[j]
            if not acc and p.word_count > cfg.hard_max_words:
                sub_chunks = _split_giant_paragraph(p, cfg)
                for sc in sub_chunks:
                    chunks.append(sc)
                j += 1
                i = j
                acc = []
                acc_words = 0
                best_cut = None
                best_score = float("-inf")
                continue

            acc.append(p)
            acc_words += p.word_count

            if cfg.min_words <= acc_words <= cfg.max_words:
                next_para = paragraphs[j + 1] if j + 1 < n else None
                sc = _score_cut(acc_words, cfg, next_para, p,
                                 is_chapter_start=(next_para.is_chapter_boundary
                                                   if next_para else True))
                if sc > best_score:
                    best_score = sc
                    best_cut = j

            if acc_words > cfg.max_words:
                if best_cut is not None:
                    break
                if acc_words <= cfg.hard_max_words:
                    # ещё можно поискать разрез чуть дальше цели
                    if acc_words > cfg.hard_max_words - 50:
                        best_cut = j
                        break
                else:
                    best_cut = j
                    break
            j += 1
        else:
            # дошли до конца книги, не найдя формального разреза
            if acc:
                best_cut = j - 1

        if best_cut is None:
            best_cut = j if j < n else n - 1

        chunk_paras = paragraphs[i:best_cut + 1]
        if chunk_paras:
            chunks.append(_make_chunk_record(chunk_paras))
        i = best_cut + 1

    # Слияние слишком маленького последнего чанка книги с предыдущим
    if len(chunks) >= 2 and chunks[-1]["word_count"] < cfg.min_final_chunk:
        last = chunks.pop()
        chunks[-1]["text"] = chunks[-1]["text"] + "\n\n" + last["text"]
        chunks[-1]["word_count"] += last["word_count"]
        chunks[-1]["ends_with_dialogue"] = last["ends_with_dialogue"]

    return chunks


def _split_giant_paragraph(p: Paragraph, cfg: ChunkingConfig) -> list[dict]:
    """Редкий крайний случай: один абзац длиннее hard_max слов.
    Разбиваем по предложениям (не по словам), чтобы не резать посреди фразы."""
    sentences = split_sentences(p.text)
    out = []
    buf, buf_words = [], 0
    for s in sentences:
        wc = len(s.split())
        if buf_words + wc > cfg.max_words and buf_words >= cfg.min_words:
            out.append(_make_chunk_record([Paragraph(
                text=" ".join(buf), word_count=buf_words,
                is_dialogue_start=is_dialogue_paragraph(buf[0]),
                is_dialogue_end=is_dialogue_ending(buf[-1]),
                chapter_id=p.chapter_id, is_chapter_boundary=False)]))
            buf, buf_words = [], 0
        buf.append(s)
        buf_words += wc
    if buf:
        out.append(_make_chunk_record([Paragraph(
            text=" ".join(buf), word_count=buf_words,
            is_dialogue_start=is_dialogue_paragraph(buf[0]),
            is_dialogue_end=is_dialogue_ending(buf[-1]),
            chapter_id=p.chapter_id, is_chapter_boundary=False)]))
    return out


def _make_chunk_record(chunk_paras: list[Paragraph]) -> dict:
    text = "\n\n".join(p.text for p in chunk_paras)
    return {
        "text": text,
        "word_count": sum(p.word_count for p in chunk_paras),
        "n_paragraphs": len(chunk_paras),
        "starts_with_dialogue": chunk_paras[0].is_dialogue_start,
        "ends_with_dialogue": chunk_paras[-1].is_dialogue_end,
        "chapter_span": (chunk_paras[0].chapter_id, chunk_paras[-1].chapter_id),
        "starts_new_chapter": chunk_paras[0].is_chapter_boundary,
    }


# =============================================================================
# ПОЛНЫЙ ПАЙПЛАЙН ОБРАБОТКИ ОДНОЙ КНИГИ
# =============================================================================

def clean_raw_text(raw_text: str) -> str:
    """Применяет весь пайплайн очистки к сырому тексту книги."""
    text = strip_gutenberg_boilerplate(raw_text)
    text = remove_table_of_contents(text)
    text = remove_backmatter(text)
    text = remove_notes_and_footnotes(text)
    text = fix_typography(text)
    return text


def process_book(path: Path, author: str, cfg: ChunkingConfig) -> dict:
    """Полный цикл обработки одной книги: чтение -> очистка -> контроль
    качества -> чанкинг. Возвращает словарь с чанками и метриками."""
    raw = read_text_file(path)
    cleaned = clean_raw_text(raw)
    paragraphs = build_paragraph_index(cleaned)

    total_words = sum(p.word_count for p in paragraphs)
    quality = compute_ocr_quality_score(cleaned)
    quality["total_words"] = total_words
    quality["n_paragraphs"] = len(paragraphs)
    quality["n_chapters_detected"] = len({p.chapter_id for p in paragraphs}) if paragraphs else 0
    quality["shingles"] = shingles(cleaned)

    warnings = []
    if total_words < 3000:
        warnings.append(f"Очень короткий текст после очистки ({total_words} слов) - "
                         f"проверьте, не срезано ли слишком много служебных данных.")
    if quality["ocr_score"] < 0.6:
        warnings.append(f"Низкий OCR-скор ({quality['ocr_score']}) - похоже на "
                         f"артефакты сканирования, желательна ручная проверка.")
    if quality["common_word_ratio"] < 0.15:
        warnings.append("Низкая доля частотных английских слов - возможно, текст "
                         "не на английском или сильно повреждён.")

    chunks = chunk_paragraphs(paragraphs, cfg) if paragraphs else []

    return {
        "author": author,
        "book": path.stem,
        "path": str(path),
        "quality": {k: v for k, v in quality.items() if k != "shingles"},
        "warnings": warnings,
        "chunks": chunks,
    }


# =============================================================================
# ДЕДУБЛИКАЦИЯ НА УРОВНЕ КОРПУСА
# =============================================================================

def find_duplicates(book_results: list[dict], shingle_sets: dict,
                     threshold: float = 0.6) -> list[dict]:
    dups = []
    keys = list(shingle_sets.keys())
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            sim = jaccard(shingle_sets[keys[a]], shingle_sets[keys[b]])
            if sim >= threshold:
                dups.append({"book_a": keys[a], "book_b": keys[b],
                             "similarity": round(sim, 4)})
    return dups


# =============================================================================
# ОСНОВНОЙ ПРОГОН ПО КОРПУСУ
# =============================================================================

def run_pipeline(input_dir: Path, output_dir: Path, cfg: ChunkingConfig,
                  dup_threshold: float = 0.6, write_chunk_files: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "preprocessing.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )
    LOG.info("NLTK punkt доступен: %s | chardet доступен: %s",
             _HAS_NLTK_PUNKT, _HAS_CHARDET)

    author_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))])  # _demo_samples = test-only, never train
    if not author_dirs:
        LOG.error("В %s не найдено ни одной папки автора.", input_dir)
        return

    all_book_results: list[dict] = []

    for author_dir in author_dirs:
        author = author_dir.name
        txt_files = sorted(author_dir.glob("*.txt"))
        LOG.info("Автор %-20s -> %d файлов", author, len(txt_files))
        for txt_path in txt_files:
            LOG.info("  Обработка: %s", txt_path.name)
            try:
                result = process_book(txt_path, author, cfg)
            except Exception as exc:  # не роняем весь прогон из-за одного файла
                LOG.exception("  Ошибка при обработке %s: %s", txt_path, exc)
                continue
            for w in result["warnings"]:
                LOG.warning("    [%s/%s] %s", author, txt_path.stem, w)
            all_book_results.append(result)

    # для дедупликации строим шинглы по полному (склеенному из чанков) тексту книги
    shingle_map: dict[str, set] = {}
    for r in all_book_results:
        full_text = " ".join(c["text"] for c in r["chunks"])
        shingle_map[f"{r['author']}/{r['book']}"] = shingles(full_text)

    duplicates = find_duplicates(all_book_results, shingle_map, dup_threshold)
    if duplicates:
        LOG.warning("Найдено %d пар подозрительно похожих книг (см. duplicates_report.json)",
                    len(duplicates))
        for d in duplicates:
            LOG.warning("  %s <-> %s (similarity=%.3f)", d["book_a"], d["book_b"], d["similarity"])

    # ---- запись датасета ----
    dataset_path = output_dir / "dataset.jsonl"
    n_chunks_total = 0
    with dataset_path.open("w", encoding="utf-8") as f:
        for r in all_book_results:
            for idx, chunk in enumerate(r["chunks"]):
                record = {
                    "author": r["author"],
                    "book": r["book"],
                    "chunk_id": idx,
                    "text": chunk["text"],
                    "word_count": chunk["word_count"],
                    "n_paragraphs": chunk["n_paragraphs"],
                    "starts_with_dialogue": chunk["starts_with_dialogue"],
                    "ends_with_dialogue": chunk["ends_with_dialogue"],
                    "starts_new_chapter": chunk["starts_new_chapter"],
                    "chapter_span": chunk["chapter_span"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_chunks_total += 1

                if write_chunk_files:
                    chunk_dir = output_dir / "chunks" / r["author"]
                    chunk_dir.mkdir(parents=True, exist_ok=True)
                    fname = chunk_dir / f"{r['book']}__chunk_{idx:04d}.txt"
                    fname.write_text(chunk["text"], encoding="utf-8")

    # ---- отчёт о качестве ----
    quality_report = {
        f"{r['author']}/{r['book']}": {**r["quality"], "warnings": r["warnings"],
                                        "n_chunks": len(r["chunks"])}
        for r in all_book_results
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "duplicates_report.json").write_text(
        json.dumps(duplicates, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- сводка ----
    per_author_chunks: dict[str, int] = {}
    for r in all_book_results:
        per_author_chunks[r["author"]] = per_author_chunks.get(r["author"], 0) + len(r["chunks"])

    LOG.info("=" * 70)
    LOG.info("ГОТОВО. Книг обработано: %d | Чанков создано: %d",
              len(all_book_results), n_chunks_total)
    for author, cnt in sorted(per_author_chunks.items()):
        LOG.info("  %-20s : %d чанков", author, cnt)
    LOG.info("Датасет: %s", dataset_path)
    LOG.info("Отчёт о качестве: %s", output_dir / "quality_report.json")
    LOG.info("Отчёт о дублях:    %s", output_dir / "duplicates_report.json")


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=Path("/data"),
                     help="Папка с подпапками авторов (по умолчанию /data)")
    ap.add_argument("--output-dir", type=Path, default=Path("/data_processed"),
                     help="Куда сохранить датасет и отчёты")
    ap.add_argument("--target-words", type=int, default=1500)
    ap.add_argument("--min-words", type=int, default=1450)
    ap.add_argument("--max-words", type=int, default=1550)
    ap.add_argument("--dup-threshold", type=float, default=0.6,
                     help="Порог Jaccard-схожести для флага дубликата (0-1)")
    ap.add_argument("--write-chunk-files", action="store_true",
                     help="Дополнительно сохранить каждый чанк отдельным .txt")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = ChunkingConfig(target_words=args.target_words,
                          min_words=args.min_words,
                          max_words=args.max_words)
    run_pipeline(args.input_dir, args.output_dir, cfg,
                 dup_threshold=args.dup_threshold,
                 write_chunk_files=args.write_chunk_files)


if __name__ == "__main__":
    main()