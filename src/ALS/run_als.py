from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix

from helpers.data import UBCData


@dataclass(frozen=True)
class ALSRunConfig:
    name: str
    cuda_visible_devices: str
    data_path: str
    save_path: str
    als_factors: int
    col: str

    user_col: str = "client_id"
    weight_buy: float = 3.0
    weight_add: float = 1.0
    weight_visit: float = 1.0
    pad_missing_relevant_clients: bool = True


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _expand_pathlike(v: str) -> str:
    return os.path.expanduser(os.path.expandvars(v))


def _build_config(cfg_dict: Dict[str, Any], args: argparse.Namespace) -> ALSRunConfig:
    if args.name is not None:
        cfg_dict["name"] = args.name
    if args.cuda_visible_devices is not None:
        cfg_dict["cuda_visible_devices"] = str(args.cuda_visible_devices)
    if args.data_path is not None:
        cfg_dict["data_path"] = args.data_path
    if args.save_path is not None:
        cfg_dict["save_path"] = args.save_path
    if args.factors is not None:
        cfg_dict["als_factors"] = int(args.factors)
    if args.col is not None:
        cfg_dict["col"] = args.col

    cfg_dict["data_path"] = _expand_pathlike(cfg_dict["data_path"])
    cfg_dict["save_path"] = _expand_pathlike(cfg_dict["save_path"])

    return ALSRunConfig(**cfg_dict)


class ALSEmbeddingJob:
    def __init__(self, cfg: ALSRunConfig):
        self.cfg = cfg

    def run(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.cuda_visible_devices)
        data = UBCData.from_disk(self.cfg.data_path, self.cfg.data_path)
        df = self._make_interactions(data)
        client_ids, user_factors, item_vocab, item_factors = self._als_fit(df)
        client_ids, user_factors = self._restrict_to_relevant(
            client_ids=client_ids,
            user_factors=user_factors,
            relevant_clients=np.asarray(data.relevant_clients),
            pad_missing=self.cfg.pad_missing_relevant_clients,
        )
        out_dir = Path(self.cfg.save_path) / "embeddings" / self.cfg.name
        self._save(out_dir, client_ids, user_factors, item_vocab, item_factors)

    def _make_interactions(self, data: UBCData) -> pd.DataFrame:
        if self.cfg.col == "url":
            df = data.page_visit.loc[
                data.page_visit[self.cfg.user_col].isin(data.relevant_clients),
                [self.cfg.user_col, "url"],
            ].copy()
            df["weight"] = float(self.cfg.weight_visit)
            return df

        buy = data.product_buy[[self.cfg.user_col, self.cfg.col]].copy()
        add = data.add_to_cart[[self.cfg.user_col, self.cfg.col]].copy()
        buy["weight"] = float(self.cfg.weight_buy)
        add["weight"] = float(self.cfg.weight_add)
        return pd.concat([buy, add], ignore_index=True)

    def _als_fit(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        user_codes, user_vocab = self._encode(df[self.cfg.user_col])
        item_codes, item_vocab = self._encode(df[self.cfg.col])

        mat = csr_matrix(
            (df["weight"].to_numpy(dtype=np.float32, copy=False), (user_codes, item_codes)),
            shape=(int(user_vocab.shape[0]), int(item_vocab.shape[0])),
        )

        try:
            from implicit.gpu.als import AlternatingLeastSquares  # type: ignore
            model = AlternatingLeastSquares(factors=int(self.cfg.als_factors))
        except Exception:
            from implicit.als import AlternatingLeastSquares  # type: ignore
            model = AlternatingLeastSquares(factors=int(self.cfg.als_factors))

        model.fit(mat)
        user_factors = np.asarray(model.user_factors, dtype=np.float32)
        item_factors = np.asarray(model.item_factors, dtype=np.float32)

        return user_vocab, user_factors, item_vocab, item_factors

    @staticmethod
    def _encode(s: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        cat = pd.Categorical(s.astype("object"))
        return cat.codes.astype(np.int32, copy=False), np.asarray(cat.categories)

    @staticmethod
    def _restrict_to_relevant(
        client_ids: np.ndarray,
        user_factors: np.ndarray,
        relevant_clients: np.ndarray,
        pad_missing: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.isin(client_ids, relevant_clients)
        kept_ids = client_ids[mask]
        kept_factors = user_factors[mask]

        if pad_missing:
            missing = relevant_clients[~np.isin(relevant_clients, kept_ids)]
            if missing.size:
                kept_ids = np.concatenate([kept_ids, missing], axis=0)
                zeros = np.zeros((missing.size, kept_factors.shape[1]), dtype=kept_factors.dtype)
                kept_factors = np.vstack([kept_factors, zeros])

        return kept_ids, kept_factors

    @staticmethod
    def _save(out_dir: Path, client_ids: np.ndarray, user_factors: np.ndarray, item_vocab: np.ndarray, item_factors: np.ndarray) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", user_factors.astype(np.float16, copy=False))
        np.save(out_dir / "client_ids.npy", client_ids)
        
        np.save(out_dir / "item_ids.npy", item_vocab)
        np.save(out_dir / "item_factors.npy", item_factors.astype(np.float16, copy=False))



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--cuda-visible-devices", dest="cuda_visible_devices", default=None)
    p.add_argument("--data-path", dest="data_path", default=None)
    p.add_argument("--save-path", dest="save_path", default=None)
    p.add_argument("--factors", type=int, default=None)
    p.add_argument("--col", dest="item col", default=None)
    return p


def main():
    args = build_argparser().parse_args()
    cfg_dict = _load_yaml(Path(args.config))
    cfg = _build_config(cfg_dict, args)
    ALSEmbeddingJob(cfg).run()


if __name__ == "__main__":
    main()
