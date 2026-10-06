# Shyn-upl (public web UI, stateless)
No database, no cookies, no sessions. The portfolio lives in the visitor's browser (sessionStorage, cleared when the
tab closes) and is sent with each check; the server builds the profile, replies and forgets.
Deploy: any host that runs a Docker container (Render, Railway, Fly.io, Cloud Run, HF Spaces). No env vars needed.
Do NOT enable request-body logging on the host. Netlify/Vercel are not suitable (Python + ML libs exceed limits).
Local: pip install -r requirements.txt && python portfolio_app.py  -> http://127.0.0.1:5002

Notes: the UI is dark-only and available in English, Russian and Kazakh (strings live in the `D` table in templates/portfolio.html;
the server returns message keys, never translated text). The analysis models still understand English texts only.
Two separate modes: "5 classic authors" (browse the model's training data, then check a text: 1) overlap with the training
data, 2) originality = style match + AI-likeness) and "My portfolio" (user's own works).
Overlap search uses data_processed/overlap_index.npz (word 8-gram shingles, built by `python build_overlap_index.py`);
rebuild it whenever data_processed/dataset.jsonl changes. Light per-IP rate limit (25 checks / 10 min) is kept in memory only.
AI-likeness is labelled experimental: the detector was trained on a very small dataset.
