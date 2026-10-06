#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
candidate_store.py
=====================================================================
См. SECURITY_AND_HARDENING_TODO.md, п.1 (вторая, ранее не закрытая
половина): "previous_texts не должен приходить от клиента вообще -
только student_id/submission_id, а backend сам достаёт доверенные
тексты."

ПРОБЛЕМА, КОТОРУЮ ЭТО ЗАКРЫВАЕТ
-------------------------------------------------------------------
До этой правки `/api/v1/verify` строил Style Profile кандидата ИЗ ТОГО,
ЧТО ПРИШЛО В ТЕЛЕ ЗАПРОСА (`previous_texts`) - на каждый вызов заново, без
какой-либо проверки, что эти тексты действительно принадлежат
`candidate_author_id`. Любой, кто знает API-ключ шлюза (см.
api_analyze.py, п.1, первая половина - уже закрыта), мог отправить
ЛЮБОЙ текст как "предыдущую работу" ЛЮБОГО кандидата и получить
MATCH/MISMATCH против профиля, который сам же и придумал в этом запросе -
т.е. верификация проверяла бы не "похож ли new_text на РЕАЛЬНОГО
candidate_author_id", а "похож ли new_text на текст, который вызывающий
только что подставил под этим именем".

РЕШЕНИЕ
-------------------------------------------------------------------
Тексты профиля теперь регистрируются ОТДЕЛЬНЫМ, явным вызовом
(`POST /api/v1/candidate-texts`, см. api_analyze.py) - каждая регистрация
записывается (кто/когда/через какой source) и хранится здесь, в
персистентном SQLite-хранилище. `/api/v1/verify` строит профиль ТОЛЬКО из
того, что уже лежит в этом хранилище под данным candidate_author_id -
никогда из текущего тела запроса. Это не устраняет требование доверия к
самому вызывающему (gateway, знающий SHYNDYQ_API_KEY, по-прежнему может
зарегистрировать произвольный текст под произвольным candidate_author_id -
X-API-Key доказывает "это доверенный gateway", а не "это конкретно текст
именно этого студента") - но это МЕНЯЕТ модель атаки: подмена текста
профиля теперь требует ОТДЕЛЬНОГО, аудируемого вызова регистрации (видного
в audit-логе, см. registered_by/source ниже), а не может быть тихо
провёрнута заодно с самим запросом на верификацию, результат которого
никто отдельно не проверяет.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


class CandidateTextStore:
    """candidate_author_id -> список ранее зарегистрированных текстов
    (append-only - нет метода update/delete: если текст был ошибочно
    зарегистрирован не под тем кандидатом, правильный ответ - явно
    зарегистрировать корректный текст под правильным candidate_author_id,
    а не молча переписать историю задним числом)."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidate_texts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_author_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT,
                    registered_by TEXT,
                    added_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_candidate_texts_candidate "
                "ON candidate_texts(candidate_author_id)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def add_text(self, candidate_author_id: str, text: str,
                 source: str | None = None, registered_by: str | None = None) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO candidate_texts (candidate_author_id, text, source, "
                "registered_by, added_at) VALUES (?, ?, ?, ?, ?)",
                (candidate_author_id, text, source, registered_by, time.time()),
            )
            conn.commit()
            return cursor.lastrowid

    def get_texts(self, candidate_author_id: str) -> list[str]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT text FROM candidate_texts WHERE candidate_author_id = ? ORDER BY id",
                (candidate_author_id,),
            ).fetchall()
        return [r[0] for r in rows]

    def count(self, candidate_author_id: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM candidate_texts WHERE candidate_author_id = ?",
                (candidate_author_id,),
            ).fetchone()
        return row[0]

    def sample_impostor_texts(self, exclude_candidate_author_id: str,
                               max_candidates: int = 8, max_texts_per_candidate: int = 5
                               ) -> dict[str, list[str]]:
        """Тексты ДРУГИХ (не exclude_candidate_author_id) кандидатов, уже
        зарегистрированных в хранилище - используются как пул самозванцев
        (см. author_verification.impostors_similarity) вместо того, чтобы
        принимать `impostor_texts_by_candidate` от клиента отдельным полем
        запроса (тот же класс проблемы, что и с previous_texts - клиент мог
        подложить произвольный текст под произвольным чужим candidate_id
        как "самозванца"). Ограничено max_candidates/max_texts_per_candidate,
        чтобы один вызов /verify не тянул весь накопленный корпус целиком."""
        with self._lock, self._connect() as conn:
            other_ids = [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT candidate_author_id FROM candidate_texts "
                    "WHERE candidate_author_id != ? LIMIT ?",
                    (exclude_candidate_author_id, max_candidates),
                ).fetchall()
            ]
            pool: dict[str, list[str]] = {}
            for other_id in other_ids:
                rows = conn.execute(
                    "SELECT text FROM candidate_texts WHERE candidate_author_id = ? "
                    "ORDER BY id LIMIT ?",
                    (other_id, max_texts_per_candidate),
                ).fetchall()
                if rows:
                    pool[other_id] = [r[0] for r in rows]
        return pool
