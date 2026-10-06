"""Rebuild data_processed/overlap_index.npz after the training corpus changes:  python build_overlap_index.py"""
import pathlib
import overlap
import train_verifier as tv

R = pathlib.Path(__file__).parent / "data_processed"
print(overlap.save(R / "overlap_index.npz", tv.load_corpus_by_author_book(R / "dataset.jsonl")))
