"""Near-duplicate search of a text against the training corpus.
Method: word 8-gram shingles, hashed (64-bit) and sampled (hash % 4 == 0) -> compact sorted index (~13 MB).
The index is prebuilt by build_overlap_index.py and shipped as data_processed/overlap_index.npz."""
import json
import re
import zlib
from collections import Counter

import numpy as np

K, SAMPLE = 8, 4
_W = re.compile(r"[a-z0-9']+")
_M1, _M2 = np.uint64(1099511628211), np.uint64(0xBF58476D1CE4E5B9)


def _hashes(text, cache):
    ids = []
    for w in _W.findall(text.lower()):
        v = cache.get(w)
        if v is None:
            v = cache[w] = zlib.crc32(w.encode())
        ids.append(v)
    a, n = np.asarray(ids, dtype=np.uint64), len(ids) - K + 1
    if n <= 0:
        return np.empty(0, np.uint64)
    h = np.zeros(n, np.uint64)
    for j in range(K):
        h = h * _M1 + a[j:j + n]
    h ^= h >> np.uint64(29)
    h *= _M2
    h ^= h >> np.uint64(32)
    return h[(h & np.uint64(SAMPLE - 1)) == 0]


def build(corpus):
    cache, src, hs, cs = {}, [], [], []
    for a in sorted(corpus):
        for b in sorted(corpus[a]):
            for i, t in enumerate(corpus[a][b]):
                h = _hashes(t, cache)
                hs.append(h)
                cs.append(np.full(len(h), len(src), np.uint32))
                src.append([a, b, i])
    H, C = np.concatenate(hs), np.concatenate(cs)
    o = np.argsort(H, kind="stable")
    return H[o], C[o], src


def save(path, corpus):
    H, C, src = build(corpus)
    np.savez_compressed(path, H=H, C=C, src=np.array(json.dumps(src)))
    return len(H), len(src)


def load(path):
    z = np.load(path)
    return z["H"], z["C"], json.loads(str(z["src"]))


def query(text, H, C, src):
    q = _hashes(text, {})
    if not len(q):
        return None
    lo, hi = np.searchsorted(H, q, "left"), np.searchsorted(H, q, "right")
    hit = hi > lo
    cnt = Counter()
    for l, r in zip(lo[hit], hi[hit]):
        cnt.update(set(C[l:r].tolist()))
    top = [{"a": src[c][0], "b": src[c][1], "i": src[c][2], "p": round(100 * m / len(q), 1)} for c, m in cnt.most_common(5)]
    return {"percent": round(100 * int(hit.sum()) / len(q), 1), "top": top}
