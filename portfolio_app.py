#!/usr/bin/env python3
"""Shyndyq portfolio UI - STATELESS. Nothing is stored on the server: no database, no cookies,
no sessions. The browser keeps the portfolio in sessionStorage (cleared when the tab closes)
and sends the texts with each check; the server builds the profile, answers, and forgets.
Run: gunicorn portfolio_app:app --workers 1 --threads 4   (or: python portfolio_app.py)
api_analyze.py is only used for its model loaders/thresholds; its API routes are not served."""
import collections
import os
import random
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

# serverless hosts have a read-only disk except /tmp; api_analyze creates a (unused here) sqlite file on import
os.environ.setdefault("SHYNDYQ_CANDIDATE_DB_PATH", "/tmp/candidate_texts.db")
import api_analyze as api
import author_verification as av
import doc_extract
import overlap as ov
import train_verifier as tv

ROOT = Path(__file__).parent
api.MODEL_DIR = ROOT / "model_multiscale"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 2**20  # Vercel rejects bigger requests (4.5 MB) before they reach us
MAX_WORKS, MAX_WORK_CHARS, MAX_TOTAL = 30, 400_000, 2_500_000
_CORPUS, _IMP = {}, {}  # public reference-writer data only (never user data)


@app.before_request
def csrf():
    if request.method != "GET" and request.headers.get("X-Requested-With") != "shyn":
        return jsonify(error="Bad request."), 400


_HITS = collections.defaultdict(collections.deque)  # ip -> recent request times (rate limit only, no content)
ALLOWED = (".txt", ".docx", ".pdf")


def limited(kind, n, window=600):
    ip = (request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "?") + kind
    now, q = time.time(), _HITS[ip]
    while q and now - q[0] > window:
        q.popleft()
    if len(_HITS) > 5000:
        for k in [k for k, v in _HITS.items() if not v]:
            del _HITS[k]
    if len(q) >= n:
        return True
    q.append(now)
    return False


def err(msg, code, k=None):
    return jsonify(error=msg, k=k), code  # k = message key the UI translates (ru/en/kk)


@app.errorhandler(413)
def _413(e):
    return err("That file is too large. Please use files under 4 MB in total.", 413, "e_big")


@app.errorhandler(404)
def _404(e):
    return err("We couldn't find that page.", 404)


@app.errorhandler(405)
def _405(e):
    return err("That action isn't available.", 405)


@app.errorhandler(Exception)
def _500(e):
    if hasattr(e, "code") and isinstance(e.code, int) and e.code < 500:
        return err("Something about that request wasn't right. Please try again.", e.code)
    app.logger.error("unhandled error: %s", type(e).__name__)  # class only - never log user text
    return err("Something went wrong on our side. Please try again in a moment.", 500)


@app.after_request
def no_store(r):
    r.headers["Cache-Control"] = "no-store"
    return r


def corpus():
    if not _CORPUS:
        _CORPUS.update(tv.load_corpus_by_author_book(ROOT / "data_processed" / "dataset.jsonl"))
    return _CORPUS


def _split(w):
    allc = [t for b in sorted(corpus()[w]) for t in corpus()[w][b]]
    random.Random(w).shuffle(allc)
    return allc[:21], allc[21:]  # profile chunks, unseen sample chunks


@app.get("/")
def index():
    return render_template("portfolio.html")


@app.get("/p/api/writers")
def writers():
    return jsonify(writers=sorted(corpus()))


@app.post("/p/api/extract")
def extract():
    if limited("x", 60):
        return err("You're going a little fast. Please wait a few minutes and try again.", 429, "fast")
    works, errs = [], []
    for f in request.files.getlist("files")[:MAX_WORKS]:
        name = f.filename or "file"
        if not name.lower().endswith(ALLOWED):
            errs.append({"k": "e_unsup", "n": name})
            continue
        try:
            text = doc_extract.extract_text(name, f.read())[:MAX_WORK_CHARS]
        except Exception:
            errs.append({"k": "e_read", "n": name})
            continue
        if not text.strip():
            errs.append({"k": "e_read", "n": name})
            continue
        works.append({"title": name[:80], "text": text})
    return jsonify(works=works, errors=errs)

_STATS, _OV = {}, {}


def overlap_index():
    if not _OV:  # prebuilt by build_overlap_index.py; rebuilt in memory (slow) only if missing/outdated
        try:
            _OV["x"] = ov.load(ROOT / "data_processed" / "overlap_index.npz")
        except Exception:
            H, C, src = ov.build(corpus())
            _OV["x"] = (H, C, src)
    return _OV["x"]


@app.get("/p/api/corpus")
def corpus_info():  # what the model was trained on: authors > books > number of excerpts / words
    if not _STATS:
        _STATS["a"] = [{"name": a, "books": [{"name": b, "chunks": len(ch), "words": sum(len(t.split()) for t in ch)}
                                              for b, ch in sorted(bk.items())]} for a, bk in sorted(corpus().items())]
    return jsonify(authors=_STATS["a"])


@app.get("/p/api/chunk")
def chunk():
    a, b, i = request.args.get("a", ""), request.args.get("b", ""), request.args.get("i", 0, type=int)
    ch = corpus().get(a, {}).get(b)
    if not ch or not 0 <= i < len(ch):
        return err("Unknown excerpt.", 404)
    return jsonify(text=ch[i], n=len(ch))


@app.post("/p/api/overlap")
def overlap():  # is this text (or part of it) already in the training data?
    if limited("o", 40):
        return err("You're going a little fast. Please wait a few minutes and try again.", 429, "fast")
    text = str((request.get_json(silent=True) or {}).get("text", ""))[:MAX_WORK_CHARS]
    r = ov.query(text, *overlap_index()) if len(text.split()) >= 20 else None
    if r is None:
        return err("Please add at least 20 words.", 400, "e_s20")
    return jsonify(**r, wordCount=len(text.split()))


@app.get("/p/api/preset/<writer>")
def preset(writer):
    if writer not in corpus():
        return jsonify(error="Unknown writer."), 404
    return jsonify(works=_split(writer)[0])


@app.get("/p/api/sample/<writer>")
def sample(writer):
    if writer not in corpus():
        return jsonify(error="Unknown writer."), 404
    return jsonify(text=random.choice(_split(writer)[1]))  # full-length, never in the preset profile


def impostors(enc, skip):
    if not _IMP:
        for w, books in corpus().items():
            texts = [t for b in books.values() for t in b]
            _IMP[w] = av.build_author_profile(random.Random(w).sample(texts, min(4, len(texts))),
                                              enc, candidate_author_id=w)
    return [p for w, p in _IMP.items() if w != skip]


@app.post("/p/api/check")
def check():
    if limited("c", 25):
        return err("You're going a little fast. Please wait a few minutes and try again.", 429, "fast")
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", ""))
    works = [str(t)[:MAX_WORK_CHARS] for t in (body.get("works") or [])][:MAX_WORKS]
    if not text.strip():
        return err("Paste or upload the text you want to check.", 400, "e_notext")
    if not works or sum(map(len, works)) > MAX_TOTAL:
        return err("Your portfolio is empty or too large. Add a few earlier works (up to about 2.5 million characters in total).", 400, "e_noport")
    enc, ver, meta = api._load_verifier()
    if enc is None:
        return err("The checker is temporarily unavailable. Please try again later.", 503, "busy")
    try:
        prof = av.build_author_profile(works, enc, candidate_author_id="anonymous")
        res = av.compare_text_to_profile(
            text, prof, enc, verifier=ver,
            threshold_match=meta.get("threshold_match", api.config.VERIFIER_DEFAULT_THRESHOLD_MATCH),
            threshold_mismatch=meta.get("threshold_mismatch", api.config.VERIFIER_DEFAULT_THRESHOLD_MISMATCH),
            impostor_pool=impostors(enc, body.get("preset")))
    except av.ProfileLeakageError:
        return err("This text is already one of the works in your portfolio. Paste a new text to check.", 400, "leak")
    except ValueError:
        return err("We couldn't analyze this text. Please check that it's English prose of at least 50 words.", 400, "noan")
    pipe, le = api._load_style_model()
    ok = not av.detect_script(text)["likely_non_english"]
    paras = [api._analyze_paragraph(p, "", pipe, le, ok) for p in api._split_paragraphs(text)[:80]]
    scored = [p for p in paras if p["aiScore"] is not None]
    ai = round(sum(p["aiScore"] for p in scored) / len(scored), 4) if ok and scored else None
    words = text.split()
    wins = [w for w in (" ".join(words[i:i + 1000]) for i in range(0, len(words), 1000)) if len(w.split()) >= 10]
    wr = None
    if ok and wins:
        pr = np.mean([pipe.predict_proba([w])[0] for w in wins], axis=0)
        wr = {str(le.classes_[i]): float(pr[i]) for i in range(len(pr))}
    return jsonify(  # only plain facts - all wording is written in the UI, in English
        verify={"verdict": res["verdict"], "style_similarity_percent": res["style_similarity_percent"]},
        wordCount=len(words), englishOnly=not ok,
        ai={"score": ai, "paragraphs": [{"text": p["text"][:220], "words": p["wordCount"],
                                         "score": p["aiScore"], "tier": p["aiTier"]} for p in paras]},
        writers=wr)

@app.post("/p/api/ai")
def ai_only():  # standalone AI-likeness check: no author profile or portfolio needed
    if limited("a", 40):
        return err("You're going a little fast. Please wait a few minutes and try again.", 429, "fast")
    text = str((request.get_json(silent=True) or {}).get("text", ""))[:MAX_WORK_CHARS]
    if len(text.split()) < 50:
        return err("We couldn't analyze this text. Please check that it's English prose of at least 50 words.", 400, "noan")
    pipe, le = api._load_style_model()
    ok = not av.detect_script(text)["likely_non_english"]
    paras = [api._analyze_paragraph(p, "", pipe, le, ok) for p in api._split_paragraphs(text)[:80]]
    sc = [p for p in paras if p["aiScore"] is not None]
    tw = sum(p["wordCount"] for p in sc)
    score = round(sum(p["aiScore"] * p["wordCount"] for p in sc) / tw, 4) if ok and sc and tw else None  # word-weighted mean
    return jsonify(wordCount=len(text.split()), englishOnly=not ok, score=score,
                   paragraphs=[{"text": p["text"][:300], "words": p["wordCount"], "score": p["aiScore"], "tier": p["aiTier"]} for p in paras])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002)
