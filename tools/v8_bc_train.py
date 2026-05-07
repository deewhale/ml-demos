"""V8 阶段 A：用 self-play 100 局 697 条 turn_records 训 V8Model 战斗 head（BC 监督）。

- 5 epoch + AdamW + CE loss
- 训完保存到 sts_models/v8_combat_head_v1.pt
- 在 dataset 上算 final accuracy
- smoke env 验证 wrapper 接通
"""
import json
import logging
import os
import time

import torch

from v8.model import V8Model
from v8.pretrain import CombatPretrainDataset, train_combat_head

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("v8_bc_train")

DATA_PATH = "data/v8_pretrain/turn_records.jsonl"
CKPT_PATH = "sts_models/v8_combat_head_v1.pt"

NUM_EPOCHS = 5
BATCH_SIZE = 32
LR = 3e-4


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("device=%s", device)

    # ---------- 1) 数据集大小 + random baseline ----------
    ds = CombatPretrainDataset(DATA_PATH)
    logger.info("dataset size: %d", len(ds))
    if len(ds) == 0:
        raise RuntimeError("dataset empty")

    # 估计 random baseline = mean(1 / n_options)
    n_opts_list = []
    with open(DATA_PATH) as f:
        for line in f:
            obj = json.loads(line)
            for tr in obj.get("turn_records", []):
                n = tr.get("available_actions_count")
                if n and n > 0:
                    n_opts_list.append(n)
    if n_opts_list:
        avg_n_opts = sum(n_opts_list) / len(n_opts_list)
        random_baseline = sum(1.0 / n for n in n_opts_list) / len(n_opts_list)
        logger.info(
            "avg n_options=%.2f, random baseline acc=%.3f",
            avg_n_opts, random_baseline,
        )
    else:
        random_baseline = 0.0

    # ---------- 2) 初始 model + 训前 acc ----------
    model = V8Model()

    # ---------- 3) BC 训练 ----------
    t0 = time.time()
    result = train_combat_head(
        data_path=DATA_PATH,
        model=model,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=device,
    )
    elapsed = time.time() - t0
    logger.info("train done in %.1fs result=%s", elapsed, result)

    # ---------- 4) dataset 上 final accuracy（eval 模式） ----------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        from v8.state import V8State
        for state_dict, action_idx in ds.records:
            state = V8State.from_dict(state_dict)
            if state.phase != "COMBAT":
                continue
            out = model(state, available_actions=[])
            logits = out["logits"]
            if logits.numel() == 0 or action_idx >= logits.shape[0]:
                continue
            pred = int(torch.argmax(logits).item())
            if pred == action_idx:
                correct += 1
            total += 1
    final_acc = correct / max(1, total)
    logger.info("eval mode final_acc=%.3f total=%d", final_acc, total)

    # ---------- 5) 保存 ckpt ----------
    os.makedirs("sts_models", exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "phase": "A_combat_head_bc",
            "data": DATA_PATH,
            "n_records": len(ds),
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "device": device,
            "elapsed_sec": elapsed,
            "train_avg_loss": result.get("avg_loss"),
            "train_avg_acc": result.get("accuracy"),
            "eval_final_acc": final_acc,
            "random_baseline": random_baseline,
        },
    }
    torch.save(ckpt, CKPT_PATH)
    logger.info("saved %s", CKPT_PATH)

    # ---------- 6) smoke 验证 wrapper 接通 ----------
    try:
        from v8.combat_net_wrapper import V8CombatNetWrapper
        from v8.env import V8Env

        wrapper = V8CombatNetWrapper(model)
        env = V8Env(combat_net_wrapper=wrapper)
        state = env.reset(seed=0)
        steps_taken = 0
        for i in range(20):
            actions = env.get_available_actions()
            if not actions:
                break
            try:
                next_state, reward, done, info = env.step(0)
            except Exception as e:
                logger.warning("env.step error at i=%d: %s", i, e)
                break
            steps_taken += 1
            if done:
                break
        logger.info(
            "smoke env: steps=%d wrapper.call_count=%s",
            steps_taken, getattr(wrapper, "call_count", "n/a"),
        )
        avg_value = getattr(wrapper, "avg_value", None)
        try:
            avg_value_val = avg_value() if callable(avg_value) else avg_value
        except Exception:
            avg_value_val = avg_value
        logger.info("wrapper.avg_value=%s", avg_value_val)
    except Exception as e:
        logger.warning("smoke env failed: %s", e)

    # ---------- 7) 输出 summary ----------
    print("\n========== SUMMARY ==========")
    print(f"dataset_size: {len(ds)}")
    print(f"random_baseline_acc: {random_baseline:.3f}")
    print(f"train_avg_loss: {result.get('avg_loss'):.4f}")
    print(f"train_avg_acc: {result.get('accuracy'):.3f}")
    print(f"eval_final_acc: {final_acc:.3f}")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"ckpt: {CKPT_PATH}")


if __name__ == "__main__":
    main()
