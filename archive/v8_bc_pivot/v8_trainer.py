"""
V8 元决策训练器（BC pipeline）。

数据：data/v8_jsonl/seed_*.jsonl（v8_data_collector.py 写出的元决策记录）
训练目标：CrossEntropy(logits, teacher_label)
- 跳过 teacher_label 为 None / -1 / >= len(available_actions) 的记录
- train/val 按 seed 切分（前 24 seed 训，后 6 seed val）避免泄漏
- 变长 batching：deck/relics/potions/actions 都 padding + mask
- 优化器：AdamW lr=3e-4 wd=1e-4
- LR schedule：cosine 到 1e-5
- 每 5 epoch 存 sts_models/v8_meta_<epoch>.pt
- 日志追加 docs/v8_training_log.md（每 epoch 一行）
- 设备优先 mps，fallback cpu
- CPU 限制：torch.set_num_threads(2) + OMP_NUM_THREADS=2
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

# CPU 限制（必须在 import torch 之前设环境变量）
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

torch.set_num_threads(2)

from v8_model import (
    MAX_ACTIONS,
    V8MetaModel,
    count_parameters,
    encode_record,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data", "v8_jsonl")
MODEL_DIR = os.path.join(_HERE, "sts_models")
LOG_PATH = os.path.join(_HERE, "docs", "v8_training_log.md")

TRAIN_SEED_CUTOFF = 24      # seed < 24 → train, ≥ 24 → val（按 seed 切，不泄漏）


def _seed_from_filename(path: str) -> Optional[int]:
    """从 'seed_42.jsonl' 提 42。"""
    m = re.search(r"seed_(\d+)\.jsonl$", os.path.basename(path))
    return int(m.group(1)) if m else None


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class V8MetaDataset(Dataset):
    """加载多 jsonl，过滤无效 label。"""

    def __init__(self, jsonl_paths: List[str]):
        self.records: List[Dict[str, Any]] = []
        kept = 0
        skipped = 0
        for p in jsonl_paths:
            with open(p, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    tl = r.get("teacher_label")
                    aa = r.get("available_actions") or []
                    if tl is None or tl < 0 or tl >= len(aa) or len(aa) == 0:
                        skipped += 1
                        continue
                    self.records.append(r)
                    kept += 1
        self.kept = kept
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return encode_record(self.records[idx])


def _pad_1d(seqs: List[List[int]], pad_to: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """seqs → (ids[B,T] long, mask[B,T] float)。"""
    if pad_to is None:
        pad_to = max((len(s) for s in seqs), default=1)
    pad_to = max(pad_to, 1)
    B = len(seqs)
    ids = torch.zeros(B, pad_to, dtype=torch.long)
    mask = torch.zeros(B, pad_to, dtype=torch.float32)
    for i, s in enumerate(seqs):
        L = min(len(s), pad_to)
        if L > 0:
            ids[i, :L] = torch.tensor(s[:L], dtype=torch.long)
            mask[i, :L] = 1.0
    return ids, mask


def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    deck_ids, deck_mask = _pad_1d([b["deck_ids"] for b in batch])
    relic_ids, relic_mask = _pad_1d([b["relic_ids"] for b in batch])
    potion_ids, potion_mask = _pad_1d([b["potion_ids"] for b in batch])
    action_ids, action_mask = _pad_1d([b["action_ids"] for b in batch])

    scalars = torch.tensor([b["scalars"] for b in batch], dtype=torch.float32)
    phase_idx = torch.tensor([b["phase_idx"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    return {
        "deck_ids": deck_ids, "deck_mask": deck_mask,
        "relic_ids": relic_ids, "relic_mask": relic_mask,
        "potion_ids": potion_ids, "potion_mask": potion_mask,
        "scalars": scalars, "phase_idx": phase_idx,
        "action_ids": action_ids, "action_mask": action_mask,
        "labels": labels,
    }


# ----------------------------------------------------------------------------
# train / eval loops
# ----------------------------------------------------------------------------

def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _forward_loss_acc(
    model: V8MetaModel, batch: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, float, int]:
    """返回 (loss, n_correct, n_total)。"""
    logits = model(
        batch["deck_ids"], batch["deck_mask"],
        batch["relic_ids"], batch["relic_mask"],
        batch["potion_ids"], batch["potion_mask"],
        batch["scalars"], batch["phase_idx"],
        batch["action_ids"], batch["action_mask"],
    )
    labels = batch["labels"]
    loss = F.cross_entropy(logits, labels)
    pred = logits.argmax(dim=-1)
    n_correct = (pred == labels).sum().item()
    n_total = labels.numel()
    return loss, n_correct, n_total


def train_one_epoch(model, loader, optim, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    n_batches = 0
    for batch in loader:
        batch = _to_device(batch, device)
        loss, n_correct, n = _forward_loss_acc(model, batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * n
        total_correct += n_correct
        total += n
        n_batches += 1
    if total == 0:
        return float("nan"), float("nan")
    return total_loss / total, total_correct / total


@torch.no_grad()
def eval_loop(model, loader, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        batch = _to_device(batch, device)
        loss, n_correct, n = _forward_loss_acc(model, batch)
        total_loss += loss.item() * n
        total_correct += n_correct
        total += n
    if total == 0:
        return float("nan"), float("nan")
    return total_loss / total, total_correct / total


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def _split_paths() -> Tuple[List[str], List[str]]:
    all_paths = sorted(glob.glob(os.path.join(DATA_DIR, "seed_*.jsonl")))
    train_paths: List[str] = []
    val_paths: List[str] = []
    for p in all_paths:
        s = _seed_from_filename(p)
        if s is None:
            continue
        if s < TRAIN_SEED_CUTOFF:
            train_paths.append(p)
        else:
            val_paths.append(p)
    return train_paths, val_paths


def _append_log(line: str) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a") as fh:
        if new_file:
            fh.write("# V8 元决策训练日志\n\n")
            fh.write("| epoch | train_loss | val_loss | train_acc | val_acc | lr | time(s) |\n")
            fh.write("|------:|-----------:|---------:|----------:|--------:|----:|--------:|\n")
        fh.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_min", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--smoke", action="store_true",
                    help="只跑 1 batch + 1 epoch + 不存 checkpoint / 日志")
    args = ap.parse_args()

    train_paths, val_paths = _split_paths()
    if not train_paths:
        print(f"ERROR: 找不到训练数据 {DATA_DIR}/seed_*.jsonl")
        return 1

    print(f"[V8 trainer] device select...")
    device = _select_device()
    print(f"  device       : {device}")
    print(f"  train files  : {len(train_paths)} (seed < {TRAIN_SEED_CUTOFF})")
    print(f"  val files    : {len(val_paths)}")

    t0 = time.time()
    train_ds = V8MetaDataset(train_paths)
    val_ds = V8MetaDataset(val_paths) if val_paths else None
    print(f"  train recs   : {len(train_ds)} (skipped {train_ds.skipped})")
    if val_ds is not None:
        print(f"  val recs     : {len(val_ds)} (skipped {val_ds.skipped})")
    print(f"  load time    : {time.time() - t0:.2f}s")

    if len(train_ds) == 0:
        print("ERROR: 训练集为空")
        return 1

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate, num_workers=0,
        )

    model = V8MetaModel().to(device)
    print(f"  model params : {count_parameters(model):,}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    num_epochs = max(1, args.num_epochs)
    if args.smoke:
        # smoke: 跑一个 batch 看看通不通
        print(f"\n[SMOKE] 1 batch forward+backward only")
        t1 = time.time()
        batch_iter = iter(train_loader)
        batch = next(batch_iter)
        load_time = time.time() - t1
        batch_dev = _to_device(batch, device)
        t2 = time.time()
        loss, n_correct, n = _forward_loss_acc(model, batch_dev)
        loss_before = loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        # 同 batch 再 forward 一次看 loss 是否变化
        loss2, _, _ = _forward_loss_acc(model, batch_dev)
        loss_after = loss2.item()
        step_time = time.time() - t2
        print(f"  smoke batch_size={n}, parse_time={load_time:.3f}s, "
              f"step_time={step_time:.3f}s")
        print(f"  loss_before={loss_before:.4f}, loss_after_step={loss_after:.4f} "
              f"(diff={loss_before - loss_after:+.4f})")
        # 期望参考：均匀 logits 时 CE ≈ log(avg_action_count) ≈ log(3.25) ≈ 1.18
        print(f"  reference: log(avg_actions=3.25) ≈ 1.18")
        return 0

    # cosine LR schedule（按 epoch 调整）
    def _lr_for_epoch(ep: int) -> float:
        if num_epochs <= 1:
            return args.lr
        # ep ∈ [0, num_epochs-1]
        cos = 0.5 * (1 + math.cos(math.pi * ep / (num_epochs - 1)))
        return args.lr_min + (args.lr - args.lr_min) * cos

    print(f"\n[TRAIN] num_epochs={num_epochs}, batch_size={args.batch_size}")
    print("-" * 70)
    for epoch in range(1, num_epochs + 1):
        lr_now = _lr_for_epoch(epoch - 1)
        for g in optim.param_groups:
            g["lr"] = lr_now

        t_ep = time.time()
        tl, ta = train_one_epoch(model, train_loader, optim, device)
        if val_loader is not None:
            vl, va = eval_loop(model, val_loader, device)
        else:
            vl, va = float("nan"), float("nan")
        dt = time.time() - t_ep
        print(f"  epoch {epoch:3d}/{num_epochs} | "
              f"train_loss={tl:.4f} acc={ta:.3f} | "
              f"val_loss={vl:.4f} acc={va:.3f} | "
              f"lr={lr_now:.2e} | {dt:.1f}s")

        log_line = (
            f"| {epoch} | {tl:.4f} | {vl:.4f} | {ta:.4f} | {va:.4f} | "
            f"{lr_now:.2e} | {dt:.1f} |"
        )
        _append_log(log_line)

        # checkpoint
        if epoch % 5 == 0 or epoch == num_epochs:
            os.makedirs(MODEL_DIR, exist_ok=True)
            ckpt_path = os.path.join(MODEL_DIR, f"v8_meta_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"    checkpoint  → {ckpt_path}")

    print("-" * 70)
    print(f"[TRAIN] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
