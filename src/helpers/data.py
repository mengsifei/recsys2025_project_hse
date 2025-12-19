from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd


class UBCData:
    __slots__ = (
        "relevant_clients",
        "product_properties",
        "product_buy",
        "add_to_cart",
        "remove_from_cart",
        "page_visit",
        "search_query",
    )

    def __init__(
        self,
        *,
        relevant_clients: np.ndarray,
        product_properties: pd.DataFrame,
        product_buy: pd.DataFrame,
        add_to_cart: pd.DataFrame,
        remove_from_cart: pd.DataFrame,
        page_visit: pd.DataFrame,
        search_query: pd.DataFrame,
    ) -> None:
        self.relevant_clients = relevant_clients
        self.product_properties = product_properties
        self.product_buy = product_buy
        self.add_to_cart = add_to_cart
        self.remove_from_cart = remove_from_cart
        self.page_visit = page_visit
        self.search_query = search_query

    @staticmethod
    def _p(p: str | Path) -> Path:
        return p if isinstance(p, Path) else Path(p)

    @staticmethod
    def _load_parquet(folder: Path, stem: str) -> pd.DataFrame:
        return pd.read_parquet(folder / f"{stem}.parquet")

    @staticmethod
    def _load_relevant_clients(root: Path) -> np.ndarray:
        return np.load(root / "input" / "relevant_clients.npy")

    @staticmethod
    def _make_meta_maps(products: pd.DataFrame) -> Dict[str, pd.Series]:
        by_sku = products.set_index("sku", drop=False)
        return {
            "category": by_sku["category"],
            "price": by_sku["price"],
        }

    @staticmethod
    def _add_item_meta(events: pd.DataFrame, meta: Dict[str, pd.Series]) -> pd.DataFrame:
        if "sku" not in events.columns:
            return events
        sku = events["sku"]
        return events.assign(
            category=sku.map(meta["category"]),
            price=sku.map(meta["price"]),
        )

    @classmethod
    def from_disk(cls, data_path_raw: str | Path, data_path: str | Path) -> "UBCData":
        root = cls._p(data_path_raw)
        events_root = cls._p(data_path)

        keep = cls._load_relevant_clients(root)

        products = cls._load_parquet(root, "product_properties")
        meta = cls._make_meta_maps(products)

        event_names: Iterable[str] = (
            "product_buy",
            "add_to_cart",
            "remove_from_cart",
            "page_visit",
            "search_query",
        )

        loaded: Dict[str, pd.DataFrame] = {}
        for name in event_names:
            df = cls._load_parquet(events_root, name)
            if name in {"product_buy", "add_to_cart", "remove_from_cart"}:
                df = cls._add_item_meta(df, meta)
            loaded[name] = df

        return cls(
            relevant_clients=keep,
            product_properties=products,
            product_buy=loaded["product_buy"],
            add_to_cart=loaded["add_to_cart"],
            remove_from_cart=loaded["remove_from_cart"],
            page_visit=loaded["page_visit"],
            search_query=loaded["search_query"],
        )
