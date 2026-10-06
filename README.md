# Shyn-upl (public web UI, stateless)
No database, no cookies, no sessions. The portfolio lives in the visitor's browser (sessionStorage, cleared when the
tab closes) and is sent with each check; the server builds the profile, replies and forgets.
Deploy: any host that runs a Docker container (Render, Railway, Fly.io, Cloud Run, HF Spaces). No env vars needed.
Do NOT enable request-body logging on the host. Netlify/Vercel are not suitable (Python + ML libs exceed limits).
Local: pip install -r requirements.txt && python portfolio_app.py  -> http://127.0.0.1:5002

Notes: all user-facing messages are written in English in templates/portfolio.html (the model's internal Russian
explanations are never sent to the browser). Light per-IP rate limit (25 checks / 10 min) is kept in memory only.
AI-likeness is labelled experimental: the detector was trained on a very small dataset.
