"""
V8 极简训练循环（smoke 阶段）。

- 加载 data/v8_jsonl/smoke_seed42.jsonl
- batch_size = min(32, len(data))
- 1 个 epoch
  - policy loss = CrossEntropy(logits, chosen_action_idx)
  - value loss  = MSE(value_pred, floor / 57)
  - total = policy + value
- 保存 state_dict 到 sts_models/v8_smoke.pt
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from v8_model import (
    MAX_ACTIONS,
    STATE_DIM,
    V8SmokeModel,
    count_parameters,
    encode_action,
    encode_state,
)


_HERE = os.path.dirname(os.path.abspath(__file__))


class V8JsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.records: List[Dict[str, Any]] = []
        with open(jsonl_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        state = encode_state(rec)
        avail = rec.get("available_actions", []) or []
        chosen = rec.get("chosen_action", "")
        action_idx = encode_action(chosen, avail)
        # 把超出 MAX_ACTIONS 的动作截断到合法范围
        if action_idx >= MAX_ACTIONS:
            action_idx = 0
        floor_norm = float(rec.get("floor", 0)) / 57.0
        return state, torch.tensor(action_idx, dtype=torch.long), \
               torch.tensor(floor_norm, dtype=torch.float32), \
               len(avail)


def _collate(batch):
    states = torch.stack([b[0] for b in batch])
    actions = torch.stack([b[1] for b in batch])
    values = torch.stack([b[2] for b in batch])
    avail_lens = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return states, actions, values, avail_lens


def main() -> int:
    jsonl_path = os.path.join(_HERE, "data", "v8_jsonl", "smoke_seed42.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"ERROR: jsonl 不存在: {jsonl_path}")
        print("请先跑 scripts/v8_collect_smoke.py")
        return 1

    ds = V8JsonlDataset(jsonl_path)
    if len(ds) == 0:
        print("ERROR: 数据集为空")
        return 1

    batch_size = min(32, len(ds))
    # shuffle 一次
    random.seed(0)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    ds.records = [ds.records[i] for i in indices]

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    model = V8SmokeModel()
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    print(f"V8 trainer smoke")
    print(f"  records       : {len(ds)}")
    print(f"  batch_size    : {batch_size}")
    print(f"  num_batches   : {len(loader)}")
    print(f"  model params  : {count_parameters(model)}")
    print(f"  STATE_DIM     : {STATE_DIM}, MAX_ACTIONS: {MAX_ACTIONS}")
    print("-" * 60)

    model.train()
    for step, (states, actions, values_target, avail_lens) in enumerate(loader):
        logits, values_pred = model(states)

        # mask logits 中超出 avail_lens 的位置（设大负数），保证 CE 不奖励无效 action
        B, A = logits.shape
        arange = torch.arange(A).unsqueeze(0).expand(B, A)
        mask = arange >= avail_lens.unsqueeze(1)
        masked_logits = logits.masked_fill(mask, -1e9)

        # clamp action_idx 到 avail_lens-1（防止 padding 导致 OOB；
        # encode_action 已经只挑 avail 里的 idx，不会 OOB，但保险）
        actions_clamped = torch.minimum(actions, (avail_lens - 1).clamp(min=0))

        policy_loss = F.cross_entropy(masked_logits, actions_clamped)
        value_loss = F.mse_loss(values_pred, values_target)
        total_loss = policy_loss + value_loss

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        print(f"  batch {step:02d}: policy_loss={policy_loss.item():.4f}  "
              f"value_loss={value_loss.item():.6f}  total={total_loss.item():.4f}")

    # 保存
    model_dir = os.path.join(_HERE, "sts_models")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "v8_smoke.pt")
    torch.save(model.state_dict(), save_path)
    size_kb = os.path.getsize(save_path) / 1024.0
    print("-" * 60)
    print(f"V8 trainer smoke 完成")
    print(f"  saved         : {save_path}")
    print(f"  file size     : {size_kb:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
