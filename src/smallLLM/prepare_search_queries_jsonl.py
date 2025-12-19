from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm


@dataclass(frozen=True)
class SearchTextConfig:
    dataset_root: str
    output_file: str
    per_user_limit: int = 90


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _expand_str(v: str) -> str:
    v = v.replace("${oc.env:", "${").replace("}", "}")
    return os.path.expanduser(os.path.expandvars(v))


def _make_cfg(raw: Dict[str, Any], args: argparse.Namespace) -> SearchTextConfig:
    if args.dataset_root is not None:
        raw["dataset_root"] = args.dataset_root
    if args.output_file is not None:
        raw["output_file"] = args.output_file
    if args.per_user_limit is not None:
        raw["per_user_limit"] = int(args.per_user_limit)

    raw["dataset_root"] = _expand_str(raw["dataset_root"])
    raw["output_file"] = _expand_str(raw["output_file"])

    return SearchTextConfig(**raw)


def _to_tokens(s: str) -> np.ndarray:
    xs = str(s)[1:-1].split()
    return np.asarray(list(map(int, xs)), dtype=np.int32)


def _table_to_text(frame: pd.DataFrame, limit: int) -> Tuple[int, str]:
    cid = int(frame["client_id"].iloc[0])
    view = frame.drop(columns="client_id").tail(limit)
    return cid, view.to_string(index=False)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset-root", dest="dataset_root", default=None)
    p.add_argument("--output-file", dest="output_file", default=None)
    p.add_argument("--per-user-limit", dest="per_user_limit", type=int, default=None)
    return p


def main():
    args = build_argparser().parse_args()
    cfg = _make_cfg(_read_yaml(Path(args.config)), args)

    root = Path(cfg.dataset_root)
    keep = np.load(root / "input" / "relevant_clients.npy")

    q = pd.read_parquet(root / "search_query.parquet")
    q = q[q["client_id"].isin(keep)].copy()
    q["query"] = q["query"].apply(_to_tokens)

    mat = pd.DataFrame(np.vstack(q["query"].to_list()))
    mat["timestamp"] = q["timestamp"].values
    mat["client_id"] = q["client_id"].values

    out_dir = root / "text"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / cfg.output_file
    with out_file.open("w", encoding="utf-8") as f:
        for cid in tqdm(mat["client_id"].unique(), desc="writing"):
            sub = mat[mat["client_id"] == cid]
            real_id, txt = _table_to_text(sub, cfg.per_user_limit)
            json.dump({"client_id": str(real_id), "text": txt}, f, cls=NumpyJSONEncoder)
            f.write("\n")


if __name__ == "__main__":
    main()
