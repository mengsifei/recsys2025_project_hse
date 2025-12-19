from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from lightfm import LightFM
from lightfm.data import Dataset
from scipy import sparse


@dataclass(frozen=True)
class LightFMRunConfig:
    name: str
    data_path: str
    save_path: str
    save_dir_name: str

    loss: str = "warp"
    no_components: int = 64
    epoch: int = 20
    num_threads: int = 8

    buy_weight: float = 1.0
    cart_weight: float = 0.7
    remove_weight: float = -0.5

    user_col: str = "client_id"
    item_col: str = "sku"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _expand_env(v: str) -> str:
    return os.path.expanduser(os.path.expandvars(v))


def _make_config(cfg: Dict[str, Any], args: argparse.Namespace) -> LightFMRunConfig:
    if args.name is not None:
        cfg["name"] = args.name
    if args.data_path is not None:
        cfg["data_path"] = args.data_path
    if args.save_path is not None:
        cfg["save_path"] = args.save_path
    if args.save_dir_name is not None:
        cfg["save_dir_name"] = args.save_dir_name
    if args.loss is not None:
        cfg["loss"] = args.loss
    if args.no_components is not None:
        cfg["no_components"] = int(args.no_components)
    if args.epoch is not None:
        cfg["epoch"] = int(args.epoch)
    if args.num_threads is not None:
        cfg["num_threads"] = int(args.num_threads)

    cfg["data_path"] = _expand_env(cfg["data_path"])
    cfg["save_path"] = _expand_env(cfg["save_path"])

    return LightFMRunConfig(**cfg)


def _read_relevant_users(root_dir: Path) -> np.ndarray:
    return np.load(root_dir / "input" / "relevant_clients.npy")


def _read_events(root_dir: Path) -> pd.DataFrame:
    add_df = pd.read_parquet(root_dir / "add_to_cart.parquet")
    buy_df = pd.read_parquet(root_dir / "product_buy.parquet")
    rem_df = pd.read_parquet(root_dir / "remove_from_cart.parquet")
    return buy_df, rem_df, add_df


def _aggregate_interactions(
    buy_df: pd.DataFrame,
    rem_df: pd.DataFrame,
    add_df: pd.DataFrame,
    keep_users: np.ndarray,
    cfg: LightFMRunConfig,
) -> pd.DataFrame:
    buy_df = buy_df[[cfg.user_col, cfg.item_col]].copy()
    rem_df = rem_df[[cfg.user_col, cfg.item_col]].copy()
    add_df = add_df[[cfg.user_col, cfg.item_col]].copy()

    buy_df["score"] = float(cfg.buy_weight)
    rem_df["score"] = float(cfg.remove_weight)
    add_df["score"] = float(cfg.cart_weight)

    merged = pd.concat([buy_df, rem_df, add_df], ignore_index=True)
    merged = merged[merged[cfg.user_col].isin(keep_users)]

    merged = (
        merged.groupby([cfg.user_col, cfg.item_col], as_index=False)["score"]
        .sum()
        .reset_index(drop=True)
    )
    return merged


def _build_mappings(table: pd.DataFrame, cfg: LightFMRunConfig) -> Tuple[Dataset, Dict[Any, int], Dict[Any, int], Dict[int, Any]]:
    unique_users = table[cfg.user_col].unique().tolist()
    unique_items = table[cfg.item_col].unique().tolist()

    ds = Dataset()
    ds.fit_partial(unique_users, unique_items)

    user_map = ds.mapping()[0]
    item_map = ds.mapping()[2]
    inv_user_map = {v: k for k, v in user_map.items()}
    return ds, user_map, item_map, inv_user_map


def _item_side_features(root_dir: Path, item_map: Dict[Any, int], allowed_items: np.ndarray) -> sparse.csr_matrix:
    props = pd.read_parquet(root_dir / "product_properties.parquet")
    props = props[props["sku"].isin(allowed_items)].copy()
    props = props.reset_index(drop=True)

    tokens = props["name"].astype(str).apply(lambda s: s[1:-1].split())
    tokens = tokens.apply(lambda xs: np.array(list(map(int, xs)), dtype=np.float32))

    props = props.drop(columns=[c for c in ["category", "price"] if c in props.columns])
    props["sku"] = props["sku"].map(item_map)
    props = props.sort_values("sku")

    mat = np.vstack(tokens.values)
    return sparse.csr_matrix(mat)


def _user_side_features(root_dir: Path, user_map: Dict[Any, int], user_universe: np.ndarray, cfg: LightFMRunConfig) -> sparse.csr_matrix:
    q = pd.read_parquet(root_dir / "search_query.parquet")
    q = q[q[cfg.user_col].isin(user_universe)].copy()

    q["query"] = q["query"].astype(str).apply(lambda s: s[1:-1].split())
    q["query"] = q["query"].apply(lambda xs: np.array(list(map(int, xs)), dtype=np.float32))

    q = q.groupby(cfg.user_col, as_index=False)["query"].mean()
    q[cfg.user_col] = q[cfg.user_col].map(user_map)

    all_user_ids = pd.DataFrame({cfg.user_col: pd.Series(list(set(user_map.values())), dtype=np.int64)})
    q = all_user_ids.merge(q, how="left", on=cfg.user_col)

    def _fill(v):
        if isinstance(v, np.ndarray):
            return v
        return np.full((16,), -1.0, dtype=np.float32)

    q["query"] = q["query"].apply(_fill)
    q = q.sort_values(cfg.user_col)

    mat = np.vstack(q["query"].values)
    return sparse.csr_matrix(mat)


def _extract_embeddings(model: LightFM, inv_user_map: Dict[int, Any], keep_users: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    factors = np.asarray(model.get_user_representations()[1], dtype=np.float32)
    ids = np.arange(factors.shape[0], dtype=np.int64)
    original_ids = np.array([inv_user_map.get(int(i)) for i in ids], dtype=object)

    mask = np.isin(original_ids, keep_users)
    return original_ids[mask], factors[mask]


def _save_outputs(out_dir: Path, user_ids: np.ndarray, embeds: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "client_ids.npy", user_ids.astype(np.int64, copy=False))
    np.save(out_dir / "embeddings.npy", embeds.astype(np.float16, copy=False))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--data-path", dest="data_path", default=None)
    p.add_argument("--save-path", dest="save_path", default=None)
    p.add_argument("--save-dir-name", dest="save_dir_name", default=None)
    p.add_argument("--loss", default=None)
    p.add_argument("--no-components", dest="no_components", type=int, default=None)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--num-threads", dest="num_threads", type=int, default=None)
    return p


def main():
    args = build_argparser().parse_args()
    cfg = _make_config(_load_yaml(Path(args.config)), args)

    base_dir = Path(cfg.data_path)
    keep_users = _read_relevant_users(base_dir)

    buy_df, rem_df, add_df = _read_events(base_dir)
    interactions = _aggregate_interactions(buy_df, rem_df, add_df, keep_users, cfg)

    ds, user_map, item_map, inv_user_map = _build_mappings(interactions, cfg)

    items_used = interactions[cfg.item_col].unique()
    users_used = interactions[cfg.user_col].unique()

    item_feats = _item_side_features(base_dir, item_map, items_used)
    user_feats = _user_side_features(base_dir, user_map, users_used, cfg)

    X, W = ds.build_interactions(interactions[[cfg.user_col, cfg.item_col, "score"]].values)

    model = LightFM(loss=cfg.loss, no_components=int(cfg.no_components))
    model.fit(
        X,
        user_features=user_feats,
        item_features=item_feats,
        sample_weight=W,
        epochs=int(cfg.epoch),
        num_threads=int(cfg.num_threads),
        verbose=True,
    )

    user_ids, embeds = _extract_embeddings(model, inv_user_map, keep_users)

    out_dir = Path(cfg.save_path) / "embeddings" / cfg.save_dir_name
    if cfg.name:
        out_dir = out_dir / cfg.name

    _save_outputs(out_dir, user_ids, embeds)


if __name__ == "__main__":
    main()
