#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api_analyze.py
=====================================================================
Настоящий REST-эндпоинт для интеграции с UniPlatform (или любым другим
внешним фронтендом). В отличие от app.py (который рендерит HTML-шаблоны),
здесь — только JSON, и анализ идёт по-абзацно, чтобы можно было
подсветить в интерфейсе, какие именно куски текста совпадают со стилем
студента/автора, а какие похожи на ИИ-генерацию (в том числе внутри
ОДНОГО файла, где текст намеренно смешан).

Ничего не предвычислено и не захардкожено: каждый вызов /api/v1/analyze
реально прогоняет присланный текст через
  - model/author_style_pipeline.joblib   (стиль)
  - model_ai_detector/ai_detector_pipeline.joblib  (ИИ-детектор,
    с откатом на ai_heuristics.py для слишком коротких абзацев)

Запуск:
    pip install flask-cors --break-system-packages   # если ещё не стоит
    python3 api_analyze.py
Слушает на 127.0.0.1:5001. CORS ограничен одним origin'ом
(SHYNDYQ_API_ALLOWED_ORIGIN, по умолчанию http://localhost:5173 — дефолтный
порт Vite для локальной разработки).

АВТОРИЗАЦИЯ (см. SECURITY_AND_HARDENING_TODO.md, п.1)
---------------------------------------------------------------------
/api/v1/analyze, /api/v1/verify, /api/v1/authors и /api/v1/candidate-texts
требуют заголовок `X-API-Key`, сверяемый со значением(ями) переменной
окружения SHYNDYQ_API_KEY (можно перечислить несколько через запятую —
для ротации ключей без даунтайма). /api/v1/health не защищён (нужен
балансировщику/оркестратору для healthcheck без секрета).

Это НЕ полноценный auth для конечных пользователей — предполагается, что
ключ знает только доверенный gateway/бэкенд UniPlatform, а не браузер
студента напрямую (иначе ключ утечёт через devtools).

previous_texts БОЛЬШЕ НЕ принимается в теле /api/v1/verify (см.
candidate_store.py) — раньше клиент мог прислать произвольный текст как
"предыдущую работу" произвольного candidate_author_id прямо внутри
запроса на верификацию, без отдельного следа в логах. Теперь тексты
профиля регистрируются ОТДЕЛЬНЫМ вызовом (POST /api/v1/candidate-texts,
с собственной записью source/registered_by), и /verify строит профиль
только из уже зарегистрированного. Это не устраняет необходимость
доверять самому вызывающему (ключ по-прежнему доказывает лишь "это
доверенный gateway", а не "этот конкретный текст — точно этого
кандидата") — см. подробности в докстринге candidate_store.py.

Если сервис запущен с --host, отличным от 127.0.0.1/localhost, и
SHYNDYQ_API_KEY не задан — процесс отказывается стартовать (см. __main__),
чтобы нельзя было случайно выставить незащищённый API наружу.
"""

from __future__ import annotations

import hmac
import os
import sys
import json
from pathlib import Path

from flask import Flask, jsonify, request

import ai_detector
import ai_heuristics
import author_verification as av
import candidate_store
import config
import doc_extract
import preprocess_corpus as prep
import train_model as tm

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "model"  # переопределяется флагом --model-dir при запуске

# author_style_pipeline.joblib был сохранён при запуске train_model.py
# напрямую (модуль __main__), поэтому кастомные трансформеры должны быть
# видны под именем __main__.* при анлоаде — иначе joblib.load падает.
sys.modules["__main__"].StylometricFeaturizer = tm.StylometricFeaturizer
sys.modules["__main__"].DenseTransformer = tm.DenseTransformer

app = Flask(__name__)

# --- CORS: один конкретный origin, не "*" (см. докстринг файла) -----------
_ALLOWED_ORIGIN = os.environ.get("SHYNDYQ_API_ALLOWED_ORIGIN", "http://localhost:5173")

try:
    from flask_cors import CORS
    CORS(app, origins=[_ALLOWED_ORIGIN], allow_headers=["X-API-Key", "Content-Type"])
except ImportError:
    @app.after_request
    def _add_cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = _ALLOWED_ORIGIN
        resp.headers["Access-Control-Allow-Headers"] = "X-API-Key, Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return resp

# --- Авторизация: X-API-Key на /analyze, /verify, /authors ----------------
# Несколько ключей через запятую - удобно для ротации (выдать новый,
# подождать, пока все клиенты перейдут, убрать старый) без даунтайма.
def _load_api_keys() -> frozenset[str]:
    raw = os.environ.get("SHYNDYQ_API_KEY", "")
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


API_KEYS: frozenset[str] = _load_api_keys()

if not API_KEYS:
    # Под `python api_analyze.py --host 0.0.0.0` без ключа процесс вообще
    # откажется стартовать - см. check_startup_security()/__main__ ниже.
    # Но под gunicorn (см. Dockerfile/docker-compose.yml - `gunicorn
    # api_analyze:app`) модуль просто ИМПОРТИРУЕТСЯ, __main__ не
    # выполняется, и та проверка не сработает - единственная реальная
    # защита в этом случае - `${SHYNDYQ_API_KEY:?...}` в docker-compose.yml,
    # которая не даст контейнеру вообще стартовать без переменной
    # окружения. Если кто-то запустит `gunicorn api_analyze:app` НАПРЯМУЮ,
    # в обход docker-compose - этого предупреждения будет недостаточно,
    # поэтому оно есть здесь как минимум явный сигнал в логах.
    print("ВНИМАНИЕ: SHYNDYQ_API_KEY не задан - /api/v1/analyze и /api/v1/verify "
          "не защищены авторизацией. Под gunicorn это НЕ проверяется автоматически "
          "(см. комментарий выше) - убедитесь, что ключ задан перед публикацией "
          "наружу.")

# Префиксы путей, которые требуют ключ. /api/v1/health сюда намеренно НЕ
# входит - см. докстринг файла.
_PROTECTED_PATH_PREFIXES = ("/api/v1/analyze", "/api/v1/verify", "/api/v1/authors",
                            "/api/v1/candidate-texts")

# ---- Хранилище доверенных текстов профиля (см. candidate_store.py, п.1) ---
CANDIDATE_DB_PATH = Path(os.environ.get("SHYNDYQ_CANDIDATE_DB_PATH",
                                          str(PROJECT_DIR / "candidate_texts.db")))
CANDIDATE_STORE = candidate_store.CandidateTextStore(CANDIDATE_DB_PATH)
MIN_IMPOSTOR_SOURCE_CANDIDATES = 1  # ниже этого - impostor_pool просто пуст, не ошибка


@app.before_request
def _require_api_key():
    if request.method == "OPTIONS":
        return None  # CORS preflight - браузер не присылает кастомные заголовки
    if not any(request.path.startswith(p) for p in _PROTECTED_PATH_PREFIXES):
        return None
    if not API_KEYS:
        # Ключ не сконфигурирован. Мы дошли до этой точки только если сервис
        # был явно поднят на 127.0.0.1/localhost (иначе __main__ отказался
        # бы стартовать) - т.е. это осознанный выбор для локальной
        # разработки, поэтому пропускаем без ошибки.
        return None
    provided = request.headers.get("X-API-Key", "")
    if not any(hmac.compare_digest(provided, key) for key in API_KEYS):
        return jsonify({
            "error": "Unauthorized: отсутствует или неверен заголовок X-API-Key."
        }), 401
    return None

MIN_WORDS_FOR_STYLE = 25  # значение по умолчанию для "старой" (только 1500 слов)
                           # модели - см. _load_style_model(), где оно
                           # автоматически понижается для multiscale-модели.

# Пороги СИНХРОНИЗИРОВАНЫ с app.py (BAND_RED_MAX/GREEN_MIN, AI_BAND_RED_MAX/
# GREEN_MIN) - раздельно для стиля и для ИИ, т.к. у ложного обвинения в ИИ
# цена ошибки намного выше, чем у заниженного style%. Меняя эти числа,
# обязательно меняйте и соответствующие в app.py/config - иначе Flask-версия
# (report.html) и это демо (UniPlatform) снова разойдутся в вердиктах.
STYLE_RED_MAX = 0.33
STYLE_GREEN_MIN = 0.80
AI_RED_MAX = 0.25
AI_GREEN_MIN = 0.90

_PIPELINE = None
_LABEL_ENCODER = None
_MODEL_META_CACHE: dict | None = None

VERIFIER_DIR = PROJECT_DIR / "model_verifier"  # см. train_verifier.py
_STYLE_ENCODER = None
_PAIRWISE_VERIFIER = None
_VERIFIER_META: dict = {}
_VERIFIER_LOAD_ATTEMPTED = False


def _load_verifier():
    """Ленивая (один раз за процесс) загрузка profile-based верификатора -
    см. author_verification.py/train_verifier.py. В отличие от
    _load_style_model() (обязательной для запуска этого API), отсутствие
    model_verifier/ НЕ является ошибкой - /api/v1/verify в этом случае
    просто отвечает понятной 503, а /api/v1/analyze продолжает работать
    как раньше (п.13 задания - обратная совместимость)."""
    global _STYLE_ENCODER, _PAIRWISE_VERIFIER, _VERIFIER_META, _VERIFIER_LOAD_ATTEMPTED
    if not _VERIFIER_LOAD_ATTEMPTED:
        _STYLE_ENCODER, _PAIRWISE_VERIFIER, _VERIFIER_META = av.load_verifier_artifacts(VERIFIER_DIR)
        _VERIFIER_LOAD_ATTEMPTED = True
    return _STYLE_ENCODER, _PAIRWISE_VERIFIER, _VERIFIER_META

# (нижняя_граница_слов, ключ_масштаба_в_model_meta.json, понятная подпись)
# ЗЕРКАЛО app.py::_WORD_COUNT_BANDS - держите в синхроне, иначе Flask-версия
# (report_simple.html) и это API (UniPlatform) дадут разные формулировки
# доверия для текста одной и той же длины.
_WORD_COUNT_BANDS = [
    (900, "large", "полноценная работа (900+ слов)"),
    (600, "medium", "объёмный текст (600-900 слов)"),
    (250, "semi_small", "средний по объёму текст (250-600 слов)"),
    (60, "window", "короткий текст (60-250 слов)"),
    (0, "phrase", "очень короткий текст/фраза (менее 60 слов)"),
]


def _load_model_meta() -> dict:
    global _MODEL_META_CACHE
    if _MODEL_META_CACHE is None:
        meta_path = MODEL_DIR / "model_meta.json"
        try:
            _MODEL_META_CACHE = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _MODEL_META_CACHE = {}
    return _MODEL_META_CACHE


def confidence_for_word_count(n_words: int) -> dict:
    """Честная, непрофессионалу понятная оценка того, насколько можно
    доверять style%/highlighting ИМЕННО для текста такой длины - на основе
    held_out_metrics_by_scale активной модели (см. train_model.py
    build-multiscale и compare_models.py), а не общей точности "в среднем".
    Идентична app.py::confidence_for_word_count - см. её докстринг."""
    meta = _load_model_meta()
    by_scale = meta.get("held_out_metrics_by_scale")

    band_label = next(label for lo, _, label in _WORD_COUNT_BANDS if n_words >= lo)
    scale_key = next(key for lo, key, _ in _WORD_COUNT_BANDS if n_words >= lo)

    if not by_scale or scale_key not in by_scale:
        if n_words >= 900:
            return {"bandLabel": band_label, "level": "known",
                    "accuracyPct": round(meta.get("held_out_accuracy", 0) * 100, 1),
                    "text": "Модель проверялась именно на текстах такой длины - "
                            "оценке для такого объёма можно доверять."}
        return {"bandLabel": band_label, "level": "unknown", "accuracyPct": None,
                "text": "Эта модель обучена и проверена только на объёмных текстах "
                        "(около 1500 слов) - для более коротких текстов её "
                        "достоверность отдельно не измерялась, относитесь к "
                        "результату с осторожностью."}

    acc = round(by_scale[scale_key]["accuracy"] * 100, 1)
    if acc >= 90:
        level, text = "high", "Модель проверялась на текстах такой длины и почти всегда угадывает автора верно."
    elif acc >= 70:
        level, text = "good", "Модель проверялась на текстах такой длины и в большинстве случаев угадывает автора верно."
    elif acc >= 50:
        level, text = "moderate", "Для текста такой длины модель ошибается заметно чаще - относитесь к результату как к ориентиру, а не к доказательству."
    else:
        level, text = "low", "Для текста такой длины модель часто ошибается - на этот результат не стоит полагаться серьёзно, нужен более длинный текст для надёжной оценки."
    return {"bandLabel": band_label, "level": level, "accuracyPct": acc, "text": text}


def _load_style_model():
    global _PIPELINE, _LABEL_ENCODER, MIN_WORDS_FOR_STYLE
    if _PIPELINE is None:
        _PIPELINE, _LABEL_ENCODER = tm.load_model(MODEL_DIR)

        # Порог "текст слишком короткий для стилометрии" был откалиброван
        # под старую модель (обучена только на ~1500-словных чанках - при
        # 25 словах она НИКОГДА не видела ничего похожего на обучении, но
        # 25 всё равно было условной цифрой "хоть что-то", не измеренной).
        # Для multiscale-модели (см. train_model.py build-multiscale и
        # compare_models.py) есть честные held-out метрики по масштабам в
        # model_meta.json - "phrase" (10-25 слов) даёт ~60%+ accuracy на
        # НЕВИДЕННЫХ данных (при случайности 20% на 5 авторов), поэтому для
        # такой модели порог можно и нужно понизить, иначе преимущество
        # переобучения не используется на практике.
        meta_path = MODEL_DIR / "model_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("held_out_metrics_by_scale"):
                    MIN_WORDS_FOR_STYLE = 10
                    print(f"Обнаружена multiscale-модель ({MODEL_DIR}) - "
                          f"MIN_WORDS_FOR_STYLE понижен до {MIN_WORDS_FOR_STYLE} "
                          f"(см. held_out_metrics_by_scale в {meta_path.name}).")
            except (json.JSONDecodeError, OSError):
                pass
    return _PIPELINE, _LABEL_ENCODER


def _split_paragraphs(text: str) -> list[str]:
    cleaned = prep.clean_raw_text(text)
    paragraphs = [p.strip() for p in cleaned.split("\n\n")]
    return [p for p in paragraphs if p]


def _style_tier(style_score: float) -> str:
    if style_score <= STYLE_RED_MAX:
        return "red"
    if style_score >= STYLE_GREEN_MIN:
        return "green"
    return "yellow"


def _ai_tier(ai_score: float) -> str:
    """Семантика цвета обратная относительно стиля: низкий AI-score - хорошо
    (зелёный), высокий - тревожно (красный). Пороги намеренно асимметричны
    (red начинается только с 0.90) - см. комментарий у AI_GREEN_MIN выше."""
    if ai_score >= AI_GREEN_MIN:
        return "red"
    if ai_score <= AI_RED_MAX:
        return "green"
    return "yellow"


def _analyze_paragraph(paragraph: str, expected_author: str, pipeline, le,
                       ai_language_supported: bool = True) -> dict:
    words = [w for w in paragraph.split() if any(c.isalpha() for c in w)]
    word_count = len(words)

    entry = {
        "text": paragraph,
        "wordCount": word_count,
        "styleScore": None,
        "styleTier": None,  # 'red' | 'yellow' | 'green' | None (слишком короткий фрагмент)
        "topAuthor": None,
        "topAuthorProb": None,
        "aiScore": None,
        "aiTier": None,
        "aiSource": None,
    }

    # --- ИИ-детектор (работает от ~40 слов, иначе эвристика-резерв) ---
    # Гейт по языку (см. analyze() и SECURITY_AND_HARDENING_TODO.md, раздел
    # про русский/казахский): и обученная модель, и эвристика-резерв
    # обучены/откалиброваны только на английском. На русском/казахском
    # обученная модель давала P(AI) 89-98% на подлинном человеческом
    # тексте. aiScore=None (НЕ 0.0 - ноль читался бы клиентом как "точно
    # человек") + явный aiTier="unsupported_language".
    if not ai_language_supported:
        entry["aiScore"] = None
        entry["aiTier"] = "unsupported_language"
        entry["aiSource"] = "unsupported_language"
    else:
        ai_result = ai_detector.score_fragment(paragraph, PROJECT_DIR)
        if ai_result is not None:
            entry["aiScore"] = round(ai_result["ai_score"], 4)
            entry["aiSource"] = ai_result["source"]
        else:
            h = ai_heuristics.score_fragment(paragraph)
            entry["aiScore"] = round(h.get("ai_score", 0.0), 4)
            entry["aiSource"] = "heuristics_fallback"
        entry["aiTier"] = _ai_tier(entry["aiScore"])

    # --- Стилометрия (нужна более длинная выборка, иначе шумно) ---
    if word_count >= MIN_WORDS_FOR_STYLE:
        proba = pipeline.predict_proba([paragraph])[0]
        top_idx = int(proba.argmax())
        entry["topAuthor"] = le.classes_[top_idx]
        entry["topAuthorProb"] = round(float(proba[top_idx]), 4)
        if expected_author in le.classes_:
            exp_idx = list(le.classes_).index(expected_author)
            entry["styleScore"] = round(float(proba[exp_idx]), 4)
            entry["styleTier"] = _style_tier(entry["styleScore"])

    return entry


@app.route("/api/v1/health")
def health():
    encoder, _, _ = _load_verifier()
    return jsonify({"status": "ok", "model_dir": str(MODEL_DIR),
                     "verifier_available": encoder is not None})


@app.route("/api/v1/authors")
def authors():
    _, le = _load_style_model()
    return jsonify({"authors": list(le.classes_)})


@app.route("/api/v1/analyze", methods=["POST"])
def analyze():
    """
    Принимает multipart/form-data:
      - file: файл (.txt/.docx/.pdf) ИЛИ
      - text: сырой текст строкой
      - expected_author: ключ автора, с которым сверяем стиль (напр. "MarkTwain")
    Возвращает: агрегированный style%/ai% по документу + разбор по абзацам
    для подсветки в интерфейсе.

    КОНТРАКТ (дополнение, см. раздел про русский/казахский в
    SECURITY_AND_HARDENING_TODO.md): если текст определён как не-английский
    (кириллица, см. author_verification.detect_script), AI Detection НЕ
    выполняется: docAiScore и aiScore абзацев = null, docAiTier/aiTier =
    "unsupported_language", aiLanguageSupported = false. Клиент ДОЛЖЕН
    обрабатывать null и это значение tier - показывать "не проверено", а не
    "чисто" (ни в коем случае не приводить null к 0).
    """
    pipeline, le = _load_style_model()

    expected_author = request.form.get("expected_author", "")
    raw_text = request.form.get("text", "")

    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        raw_text = doc_extract.extract_text(f.filename, f.read())

    if not raw_text or not raw_text.strip():
        return jsonify({"error": "Пустой текст — нечего анализировать."}), 400

    paragraphs = _split_paragraphs(raw_text)
    if not paragraphs:
        return jsonify({"error": "После очистки текста не осталось содержимого."}), 400

    language = av.detect_script(raw_text)
    ai_language_supported = not language["likely_non_english"]

    analyzed = [_analyze_paragraph(p, expected_author, pipeline, le, ai_language_supported)
                for p in paragraphs]

    scored = [a for a in analyzed if a["styleScore"] is not None]
    scored_ai = [a for a in analyzed if a["aiScore"] is not None]

    doc_style_score = round(sum(a["styleScore"] for a in scored) / len(scored), 4) if scored else None
    if not ai_language_supported:
        doc_ai_score = None  # не измерено - см. гейт по языку выше
    else:
        doc_ai_score = round(sum(a["aiScore"] for a in scored_ai) / len(scored_ai), 4) if scored_ai else 0.0

    doc_style_tier = _style_tier(doc_style_score) if doc_style_score is not None else None
    doc_ai_tier = _ai_tier(doc_ai_score) if doc_ai_score is not None else "unsupported_language"

    # ВАЖНО (изменено в рамках рефакторинга на profile-based verification,
    # см. author_verification.py/README): раньше здесь style_reliable
    # зависело от doc_ai_tier - т.е. AI Detection мог "спрятать" style%.
    # Это прямо запрещено новой архитектурой: "AI Detection должен
    # оставаться отдельным независимым модулем" - см. analysis.style_verdict_text
    # и tests/test_gating.py::test_ai_red_does_not_hide_style_verdict, где то
    # же самое зафиксировано для app.py. Эта лёгкая API не использует
    # novelty/OOD-детектор (в отличие от analysis.analyze_text), поэтому
    # style_reliable здесь просто всегда True - число style% никогда не
    # скрывается из-за результата ДРУГОГО модуля.
    style_reliable = True

    flagged_paragraphs = sum(
        1 for a in analyzed
        if a["aiTier"] == "red" or (a["styleTier"] == "red")
    )

    n_words = len(raw_text.split())

    return jsonify({
        "expectedAuthor": expected_author,
        "docStyleScore": doc_style_score,
        "docStyleTier": doc_style_tier,
        "docAiScore": doc_ai_score,
        "docAiTier": doc_ai_tier,
        "aiLanguageSupported": ai_language_supported,
        "language": language,
        "styleReliable": style_reliable,
        "confidence": confidence_for_word_count(n_words),
        "wordCount": n_words,
        "totalParagraphs": len(analyzed),
        "flaggedParagraphs": flagged_paragraphs,
        "paragraphs": analyzed,
        "modelInfo": {
            "authors": list(le.classes_),
            "source": "author_style_pipeline.joblib + ai_detector_pipeline.joblib (реальный инференс, без предрасчёта)",
        },
    })


@app.route("/api/v1/candidate-texts", methods=["POST"])
def register_candidate_text():
    """Регистрирует ОДИН текст как доверенное свидетельство профиля
    конкретного кандидата - единственный способ, которым previous_texts
    вообще попадают в систему (см. candidate_store.py, п.1). Это ОТДЕЛЬНЫЙ,
    явный вызов - не часть /verify - поэтому у каждой регистрации есть свой
    след (source/registered_by/added_at), который можно аудировать отдельно
    от результатов самой верификации.

    Принимает JSON: {"candidate_author_id": "...", "text": "...",
                      "source": "..." (необязательно, для аудита)}
    Возвращает: {"candidate_author_id": ..., "stored_texts_count": N}
    """
    payload = request.get_json(silent=True) or {}
    candidate_author_id = str(payload.get("candidate_author_id", "")).strip()
    text = payload.get("text", "")
    source = payload.get("source")

    if not candidate_author_id:
        return jsonify({"error": "candidate_author_id обязателен."}), 400
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "text пуст или не строка."}), 400

    # registered_by - какой X-API-Key (по первым символам, не целиком - не
    # логируем секрет полностью) сделал эту регистрацию, если ключей
    # несколько (см. ротация ключей в api_analyze.py) - полезно при разборе
    # инцидента, если один из нескольких доверенных ключей окажется
    # скомпрометирован.
    used_key = request.headers.get("X-API-Key", "")
    registered_by = f"key:{used_key[:6]}..." if used_key else "no-key-configured"

    CANDIDATE_STORE.add_text(candidate_author_id, text, source=source, registered_by=registered_by)
    return jsonify({
        "candidate_author_id": candidate_author_id,
        "stored_texts_count": CANDIDATE_STORE.count(candidate_author_id),
    })


@app.route("/api/v1/candidate-texts/<candidate_author_id>")
def candidate_text_count(candidate_author_id: str):
    """Только количество - НЕ отдаёт сам текст обратно по чтению (см.
    докстринг файла) - это диагностический эндпоинт ("сколько уже
    зарегистрировано"), а не способ прочитать чужой профиль."""
    return jsonify({
        "candidate_author_id": candidate_author_id,
        "stored_texts_count": CANDIDATE_STORE.count(candidate_author_id),
    })


@app.route("/api/v1/verify", methods=["POST"])
def verify():
    """
    Profile-based (open-set) authorship verification - см.
    author_verification.py и README ("Как это работает"). В отличие от
    /api/v1/analyze (закрытая классификация среди 5 эталонных авторов),
    здесь кандидат - это ЛЮБОЙ автор/студент, а не один из фиксированного
    списка; система лишь проверяет, похож ли new_text на профиль,
    построенный из ЗАРАНЕЕ ЗАРЕГИСТРИРОВАННЫХ текстов этого кандидата.

    ВАЖНО (см. SECURITY_AND_HARDENING_TODO.md, п.1): previous_texts
    больше НЕ принимается в теле этого запроса - профиль строится
    исключительно из того, что уже лежит в candidate_store.py под этим
    candidate_author_id (см. POST /api/v1/candidate-texts). Раньше
    previous_texts приходил прямо здесь на каждый вызов - это позволяло
    подставить произвольный текст как "предыдущую работу" произвольного
    кандидата и проверить new_text против самодельного профиля. Импостор-
    пул (метод самозванцев) точно так же больше не принимается от клиента
    (impostor_texts_by_candidate) - строится автоматически из ДРУГИХ
    кандидатов, уже зарегистрированных в том же хранилище (см.
    candidate_store.sample_impostor_texts).

    Принимает JSON:
      {
        "candidate_author_id": "student-123",   # произвольная строка
        "new_text": "..."                       # работа, которую нужно проверить
      }

    Возвращает результат compare_text_to_profile() (style_similarity_score/
    percent, verdict MATCH/UNCERTAIN/MISMATCH, confidence, threshold,
    number_of_profile_texts, profile_quality, evidence, explanation) -
    см. INPUT/OUTPUT контракт в тех.задании на рефакторинг.
    """
    encoder, verifier, meta = _load_verifier()
    if encoder is None:
        return jsonify({
            "error": "Profile-based verifier не обучен на сервере - запустите "
                     "train_verifier.py --model-dir model_verifier."
        }), 503

    payload = request.get_json(silent=True) or {}
    candidate_author_id = str(payload.get("candidate_author_id", "unknown"))
    new_text = payload.get("new_text", "")

    if not new_text or not new_text.strip():
        return jsonify({"error": "new_text пуст."}), 400

    previous_texts = CANDIDATE_STORE.get_texts(candidate_author_id)
    if not previous_texts:
        return jsonify({
            "error": f"Для candidate_author_id={candidate_author_id!r} не "
                     f"зарегистрировано ни одного текста. Сначала вызовите "
                     f"POST /api/v1/candidate-texts, чтобы зарегистрировать "
                     f"доверенные предыдущие работы этого кандидата."
        }), 400

    impostor_payload = CANDIDATE_STORE.sample_impostor_texts(candidate_author_id)
    impostor_pool = None
    if impostor_payload:
        impostor_pool = [
            av.build_author_profile(texts, encoder, candidate_author_id=str(other_id))
            for other_id, texts in impostor_payload.items()
        ]

    try:
        profile = av.build_author_profile(previous_texts, encoder, candidate_author_id=candidate_author_id)
        result = av.compare_text_to_profile(
            new_text, profile, encoder, verifier=verifier,
            threshold_match=meta.get("threshold_match", config.VERIFIER_DEFAULT_THRESHOLD_MATCH),
            threshold_mismatch=meta.get("threshold_mismatch", config.VERIFIER_DEFAULT_THRESHOLD_MISMATCH),
            impostor_pool=impostor_pool,
        )
    except av.ProfileLeakageError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result)


def check_startup_security(host: str, api_keys: frozenset[str]) -> str | None:
    """Возвращает текст ошибки, если конфигурация небезопасна для запуска
    (host не-localhost без ключа), иначе None. Вынесено из __main__ в
    отдельную функцию, чтобы это правило было юнит-тестируемым без
    реального запуска процесса/сети - см. tests/test_api_auth.py."""
    if host not in ("127.0.0.1", "localhost") and not api_keys:
        return (
            f"ОШИБКА: --host {host} публикует API за пределы локальной "
            "машины, но SHYNDYQ_API_KEY не задан - /api/v1/analyze и "
            "/api/v1/verify оказались бы доступны без авторизации кому "
            "угодно (см. SECURITY_AND_HARDENING_TODO.md, п.1). "
            "Задайте переменную окружения SHYNDYQ_API_KEY (например: "
            "python3 -c \"import secrets; print(secrets.token_hex(32))\") "
            "или запустите на 127.0.0.1 за прокси, который сам добавляет "
            "авторизацию."
        )
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                     help="папка с author_style_pipeline.joblib "
                          "(например ./model_multiscale для новой модели)")
    ap.add_argument("--host", default="127.0.0.1",
                     help="0.0.0.0 нужен при запуске в Docker/на удалённом "
                          "сервере, иначе API будет недоступен снаружи контейнера")
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()

    _startup_error = check_startup_security(args.host, API_KEYS)
    if _startup_error:
        print(_startup_error, file=sys.stderr)
        sys.exit(1)

    MODEL_DIR = args.model_dir
    print(f"Загружаю модель из {MODEL_DIR} ...")
    _load_style_model()
    print(f"Модель загружена. Запускаю API на http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
