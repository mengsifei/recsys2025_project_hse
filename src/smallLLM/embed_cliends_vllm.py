from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM
from vllm.inputs import TokensPrompt


@dataclass(frozen=True)
class EmbedRunConfig:
    cuda_visible_devices: str | None
    input_file: str
    model_path: str
    tokenizer_path: str
    output_root: str
    output_name: str
    batch_size: int = 32
    max_length: int = 8192
    vllm: Dict[str, Any] | None = None


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _expand_str(v: str) -> str:
    v = v.replace("${oc.env:", "${").replace("}", "}")
    return os.path.expanduser(os.path.expandvars(v))


def _make_cfg(raw: Dict[str, Any], args: argparse.Namespace) -> EmbedRunConfig:
    if args.input_file is not None:
        raw["input_file"] = args.input_file
    if args.output_name is not None:
        raw["output_name"] = args.output_name
    if args.batch_size is not None:
        raw["batch_size"] = int(args.batch_size)
    if args.max_length is not None:
        raw["max_length"] = int(args.max_length)
    if args.cuda_visible_devices is not None:
        raw["cuda_visible_devices"] = str(args.cuda_visible_devices)

    raw["input_file"] = _expand_str(raw["input_file"])
    raw["model_path"] = _expand_str(raw["model_path"])
    raw["tokenizer_path"] = _expand_str(raw["tokenizer_path"])
    raw["output_root"] = _expand_str(raw["output_root"])

    return EmbedRunConfig(**raw)


def _load_text_dataset(input_file: Path):
    suffix = input_file.suffix.lower()
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=str(input_file), split="train")
    return load_dataset("json", data_files=str(input_file), split="train")


def _batch_iter(ds, bs: int):
    n = len(ds)
    for i in range(0, n, bs):
        yield ds[i : min(i + bs, n)]


def _tokenize_to_prompts(texts: List[str], tok: AutoTokenizer, max_len: int) -> List[TokensPrompt]:
    pack = tok(texts, padding=False, truncation=True, max_length=max_len)
    return [TokensPrompt(prompt_token_ids=ids) for ids in pack["input_ids"]]


def _embed_batch(texts: List[str], llm: LLM, tok: AutoTokenizer, max_len: int) -> np.ndarray:
    prompts = _tokenize_to_prompts(texts, tok, max_len)
    outs = llm.embed(prompts, use_tqdm=False)
    return np.asarray([o.outputs.embedding for o in outs], dtype=np.float32)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--input-file", dest="input_file", default=None)
    p.add_argument("--output-name", dest="output_name", default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--max-length", dest="max_length", type=int, default=None)
    p.add_argument("--cuda-visible-devices", dest="cuda_visible_devices", default=None)
    return p


def main():
    args = build_argparser().parse_args()
    cfg = _make_cfg(_read_yaml(Path(args.config)), args)

    if cfg.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible_devices)

    src_file = Path(cfg.input_file)
    ds = _load_text_dataset(src_file)

    llm = LLM(
        model=cfg.model_path,
        tokenizer=cfg.tokenizer_path,
        task="embed",
        trust_remote_code=True,
        **(cfg.vllm or {}),
    )

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    tok.padding_side = "left"

    ids_out: List[Any] = []
    vecs_out: List[np.ndarray] = []

    for batch in tqdm(list(_batch_iter(ds, cfg.batch_size)), desc="embedding"):
        ids_out.extend(batch["client_id"])
        vecs_out.append(_embed_batch(batch["text"], llm, tok, cfg.max_length))

    mat = np.concatenate(vecs_out, axis=0)

    out_dir = Path(cfg.output_root) / cfg.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embeddings.npy", mat.astype(np.float16, copy=False))
    np.save(out_dir / "client_ids.npy", np.asarray(ids_out).astype(np.int64, copy=False))


if __name__ == "__main__":
    main()
