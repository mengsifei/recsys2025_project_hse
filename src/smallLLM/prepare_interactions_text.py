from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import polars as pl
import yaml


@dataclass(frozen=True)
class InteractionsTextConfig:
    dataset_root: str
    output_file: str
    per_user_limit: int = 64


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _expand_str(v: str) -> str:
    v = v.replace("${oc.env:", "${").replace("}", "}")
    return os.path.expanduser(os.path.expandvars(v))


def _make_cfg(raw: Dict[str, Any], args: argparse.Namespace) -> InteractionsTextConfig:
    if args.dataset_root is not None:
        raw["dataset_root"] = args.dataset_root
    if args.output_file is not None:
        raw["output_file"] = args.output_file
    if args.per_user_limit is not None:
        raw["per_user_limit"] = int(args.per_user_limit)

    raw["dataset_root"] = _expand_str(raw["dataset_root"])
    raw["output_file"] = _expand_str(raw["output_file"])

    return InteractionsTextConfig(**raw)


def _digits_to_space_separated(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return df.with_columns(pl.col(col).cast(pl.String).str.extract_all(r"\d+").list.join(" "))


def _as_text(col: str, dtype: pl.DataType) -> pl.Expr:
    e = pl.col(col)
    if dtype == pl.String:
        return e
    return e.cast(pl.String)


def _row_to_text(df: pl.DataFrame, kind: str) -> pl.DataFrame:
    parts = [pl.lit(f"event type: {kind.replace('_', ' ')}; ")]
    for col, dtype in df.schema.items():
        if col == "client_id":
            continue
        if col == "timestamp" and hasattr(dtype, "is_temporal") and dtype.is_temporal():
            val = pl.col(col).dt.strftime("%Y-%m-%d %H:%M")
        else:
            val = _as_text(col, dtype)
        parts.extend([pl.lit(f"{col}: "), val, pl.lit("; ")])
    expr = pl.concat_str(parts).str.strip_chars()
    return df.with_columns(expr.alias("text"))


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

    props = pl.read_parquet(root / "product_properties.parquet")
    if "name" in props.columns:
        props = _digits_to_space_separated(props, "name")

    chunks = []
    for kind in ("add_to_cart", "remove_from_cart", "product_buy", "page_visit", "search_query"):
        df = (
            pl.read_parquet(root / f"{kind}.parquet")
            .filter(pl.col("client_id").is_in(keep))
            .sort("timestamp")
            .unique(pl.exclude("timestamp"), keep="last")
        )

        if kind == "search_query" and "query" in df.columns:
            df = _digits_to_space_separated(df, "query")
        elif kind in {"add_to_cart", "remove_from_cart", "product_buy"}:
            if "sku" in df.columns and "sku" in props.columns:
                df = df.join(props, on="sku", how="left")
            if "sku" in df.columns:
                df = df.rename({"sku": "item"})

        chunks.append(_row_to_text(df, kind).select(["client_id", "timestamp", "text"]))

    all_rows = (
        pl.concat(chunks)
        .sort("timestamp", descending=True)
        .with_columns(pl.col("client_id").cum_count().over("client_id").alias("pos"))
        .filter(pl.col("pos") < cfg.per_user_limit)
        .drop("pos")
        .group_by("client_id")
        .agg(pl.col("text"))
        .with_columns(pl.col("text").list.join("\n"))
    )

    out_path = root / "text"
    out_path.mkdir(parents=True, exist_ok=True)
    all_rows.write_parquet(out_path / cfg.output_file)


if __name__ == "__main__":
    main()
