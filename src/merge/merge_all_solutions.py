from __future__ import annotations

import argparse
import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class InputSpec:
    path: str
    steps: List[Dict[str, Any]] | None = None
    drop_zero_rows: bool = True


@dataclass(frozen=True)
class MergeSpec:
    dataset_root: str
    inputs: List[InputSpec]
    output_dir: str
    ensure_all_relevant: bool = True
    imputer: Dict[str, Any] | None = None
    save_float16: bool = True


def _read_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _expand(v: str) -> str:
    v = v.replace("${oc.env:", "${").replace("}", "}")
    return os.path.expanduser(os.path.expandvars(v))


def _import_symbol(qualname: str):
    mod, sym = qualname.rsplit(".", 1)
    return getattr(importlib.import_module(mod), sym)


def _instantiate(spec: Dict[str, Any]):
    target = spec["target"]
    params = dict(spec.get("params", {}))
    cls = _import_symbol(target)
    return cls(**params), spec


def _as_ids(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype.kind in {"U", "S", "O"}:
        try:
            return x.astype(np.int64)
        except Exception:
            return x.astype(object)
    return x.astype(np.int64, copy=False)


def _dedup(ids: np.ndarray, mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if ids.size == 0:
        return ids, mat
    u, idx = np.unique(ids, return_index=True)
    if u.size != ids.size:
        logging.warning("Duplicate client_ids found; keeping first occurrence per id.")
        keep = np.sort(idx)
        return ids[keep], mat[keep]
    return ids, mat


def _drop_all_zero(ids: np.ndarray, mat: np.ndarray, enabled: bool) -> Tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return ids, mat
    nz = (mat != 0).any(axis=1)
    if nz.sum() < mat.shape[0]:
        logging.info("Zero rows detected; dropping them.")
    return ids[nz], mat[nz]


def _run_steps(mat: np.ndarray, steps: List[Dict[str, Any]] | None) -> np.ndarray:
    if not steps:
        return mat
    x = mat.astype(np.float32, copy=False)
    for s in steps:
        obj, raw = _instantiate(s)
        apply_global = bool(raw.get("global", False))
        if apply_global:
            x = obj.fit_transform(x.reshape(-1, 1)).reshape(x.shape)
        else:
            x = obj.fit_transform(x)
    return x


def _align_concat(master_ids: np.ndarray, pieces: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    out_blocks: List[np.ndarray] = []
    pos = {k: i for i, k in enumerate(master_ids.tolist())}
    n = master_ids.size

    for ids, mat in pieces:
        block = np.full((n, mat.shape[1]), np.nan, dtype=np.float32)
        for r, cid in enumerate(ids.tolist()):
            j = pos.get(cid, None)
            if j is not None:
                block[j] = mat[r]
        out_blocks.append(block)

    return np.hstack(out_blocks) if out_blocks else np.empty((n, 0), dtype=np.float32)


def _pad_to_relevant(ids: np.ndarray, mat: np.ndarray, relevant: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    relevant = _as_ids(relevant)
    universe = np.union1d(ids, relevant)
    aligned = np.full((universe.size, mat.shape[1]), np.nan, dtype=np.float32)
    ix = np.searchsorted(universe, ids)
    aligned[ix] = mat
    return universe, aligned


def _impute(mat: np.ndarray, spec: Dict[str, Any] | None) -> np.ndarray:
    if not spec:
        return mat
    obj, _ = _instantiate(spec)
    return obj.fit_transform(mat)


def load_spec(cfg: Dict[str, Any]) -> MergeSpec:
    dataset_root = _expand(cfg["dataset_root"])
    output_dir = _expand(cfg["output_dir"])

    inputs: List[InputSpec] = []
    for item in cfg["inputs"]:
        inputs.append(
            InputSpec(
                path=_expand(item["path"]),
                steps=item.get("steps"),
                drop_zero_rows=bool(item.get("drop_zero_rows", True)),
            )
        )

    return MergeSpec(
        dataset_root=dataset_root,
        inputs=inputs,
        output_dir=output_dir,
        ensure_all_relevant=bool(cfg.get("ensure_all_relevant", True)),
        imputer=cfg.get("imputer"),
        save_float16=bool(cfg.get("save_float16", True)),
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p


def main():
    args = build_argparser().parse_args()
    cfg = load_spec(_read_yaml(Path(args.config)))

    root = Path(cfg.dataset_root)
    rel_path = root / "input" / "relevant_clients.npy"
    relevant = np.load(rel_path) if rel_path.exists() else None

    processed: List[Tuple[np.ndarray, np.ndarray]] = []
    all_ids: List[np.ndarray] = []

    for inp in cfg.inputs:
        base = Path(inp.path)
        ids = _as_ids(np.load(base / "client_ids.npy"))
        mat = np.asarray(np.load(base / "embeddings.npy"), dtype=np.float32)

        ids, mat = _dedup(ids, mat)
        ids, mat = _drop_all_zero(ids, mat, inp.drop_zero_rows)
        mat = _run_steps(mat, inp.steps)

        processed.append((ids, mat))
        all_ids.append(ids)

        logging.info(f"Loaded {base} -> {mat.shape}")

    master = np.unique(np.concatenate(all_ids)) if all_ids else np.array([], dtype=np.int64)
    merged = _align_concat(master, processed)

    if cfg.ensure_all_relevant and relevant is not None:
        master, merged = _pad_to_relevant(master, merged, relevant)

    merged = _impute(merged, cfg.imputer)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "client_ids.npy", master.astype(np.int64, copy=False))
    if cfg.save_float16:
        np.save(out_dir / "embeddings.npy", merged.astype(np.float16, copy=False))
    else:
        np.save(out_dir / "embeddings.npy", merged.astype(np.float32, copy=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
