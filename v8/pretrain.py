"""V8 战斗 head 预训练 script。

按 docs/v8_implementation_design.md 组件 6 设计：
- 让 StSRLSolver 自己跑战斗（self-play with random starting deck）
- 收 (combat_state, search_chosen_action) → 训 model 战斗 head（CE on action）
- 起始 deck 用 RNG 加 5-15 张随机 act1 卡，保证 state 分布干净
- 战斗 head 预训练完成后再 RL 训元决策 head

设计原则对照（docs/v8_design_principles.md）：
- 用搜索作老师（用户原话第 2 点：搜索 + model 联合）
- 不复用 v8_combat_actions 脏数据（重新跑 self-play 收新数据）
- smoke 验证有效性再决定要不要（设计文档里写的）

实现状态（Round 2）：
- collect_pretrain_data：**已接 StSRLSolver 真跑战斗 + 收 turn-level 数据**
  * 每 solve_turn 之前 snapshot engine.state → V8State（含 hand / monsters / energy）
  * search 返回的 plan[0] 转成 model._combat_forward 的 action_idx：
      * PlayCard(card_idx, target_idx) → action_idx = card_idx * M + target_idx
      * EndTurn → action_idx = H * M（最后一位）
      * UsePotion / SelectScryDiscard → 跳过此 turn（model 不学这两类，不污染）
  * 写到 turn_records list，jsonl 格式（per-line 一场战斗）
- train_combat_head：**stub + dummy forward smoke**（数据收齐后才真训）
  * 接 jsonl + DataLoader 框架就位
  * forward / loss / backward 流程验证（用 dummy combat state）

不在本文件内：
- smoke 20 局 driver（下一步）
- 真训练 driver
- 元决策 head 训练（trainer.py 的 PPO 负责）
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from v8.model import V8Model
from v8.state import V8State


logger = logging.getLogger(__name__)


# ============================================================
# 配置常量（design doc 默认值，smoke 时可覆盖）
# ============================================================

DEFAULT_NUM_BATTLES: int = 100
DEFAULT_OUTPUT_PATH: str = "data/v8_pretrain/battles.jsonl"

# 起始 deck：Ironclad 标准 10 张（5 Strike + 4 Defend + 1 Bash） + RNG 5-15 张随机 act1 卡
IRONCLAD_BASE_DECK: List[Dict[str, Any]] = (
    [{"name": "Strike", "upgraded": False}] * 5
    + [{"name": "Defend", "upgraded": False}] * 4
    + [{"name": "Bash", "upgraded": False}]
)

# 简化的 act1 红卡池（design doc 强调多样化 deck，这里挑常见的几张作 RNG 池；
# 真训前可扩展到完整 ALL_CARDS 中 character=Ironclad 的子集）
ACT1_RED_CARDS_POOL: List[str] = [
    "Anger", "Cleave", "Clothesline", "Flex", "Havoc", "Headbutt",
    "Heavy Blade", "Iron Wave", "Perfected Strike", "Pommel Strike",
    "Shrug It Off", "Sword Boomerang", "Thunderclap", "True Grit",
    "Twin Strike", "Warcry", "Wild Strike",
]

MIN_RANDOM_CARDS: int = 5
MAX_RANDOM_CARDS: int = 15


# ============================================================
# Self-play 数据收集
# ============================================================


@dataclass
class BattleRecord:
    """单场战斗记录（写到 jsonl）。

    本期保留两类字段：
    - meta：deck / relics / enemy / outcome（已可写入）
    - turn_records：每 turn 的 (combat_state, search_chosen_idx)，用于训战斗 head
                    **当前是 stub**，等用户授权 smoke 时填实
    """

    seed: int
    enemy_id: str
    starting_deck: List[Dict[str, Any]]
    relics: List[str]
    starting_hp: int
    max_hp: int
    # 战斗结束指标（同 deck_evaluator._simulate_combat 输出）
    damage_dealt: float
    damage_taken: float
    turns_played: int
    victory: bool
    # turn-level 数据：list of {"state": V8State.to_dict(), "action_idx": int}
    # 当前 stub 为 []，TODO 填实
    turn_records: List[Dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "enemy_id": self.enemy_id,
                "starting_deck": self.starting_deck,
                "relics": self.relics,
                "starting_hp": self.starting_hp,
                "max_hp": self.max_hp,
                "damage_dealt": self.damage_dealt,
                "damage_taken": self.damage_taken,
                "turns_played": self.turns_played,
                "victory": self.victory,
                "turn_records": self.turn_records,
            },
            ensure_ascii=False,
        )


def _build_random_deck(rng: random.Random) -> List[Dict[str, Any]]:
    """构造一副 Ironclad 起始 deck + 5-15 张随机 act1 红卡。

    设计意图（design doc 组件 6）：起始 deck 多样化保证 state 分布干净，避免
    所有数据都是同一副初始 deck 跑出来的。
    """
    n_random = rng.randint(MIN_RANDOM_CARDS, MAX_RANDOM_CARDS)
    extras = [
        {"name": rng.choice(ACT1_RED_CARDS_POOL), "upgraded": False}
        for _ in range(n_random)
    ]
    deck = list(IRONCLAD_BASE_DECK) + extras
    return deck


def _engine_to_v8_state(engine, deck_dicts, relics) -> V8State:
    """把 StSRLSolver CombatEngine.state snapshot 成 V8State。

    战斗 head 只看 (hand, monsters, energy, hp) 几个字段，其他元决策字段保留默认值。
    `hand` 元素 = card_id 字符串（含 "+1" 升级后缀），需要拆出 name + upgraded + cost。
    """
    cs = engine.state
    player = cs.player

    # ----- hand: card_id list → V8State.hand dict list -----
    # card cost: 优先看 ALL_CARDS[base_id].cost；这里简化，cost 用 1 作为 fallback
    # （战斗 head 实际更看重 cross product 索引位置而不是 cost 数值精度）
    try:
        from packages.engine.content.cards import ALL_CARDS  # type: ignore
    except Exception:  # noqa: BLE001
        ALL_CARDS = {}

    hand_list: List[Dict[str, Any]] = []
    for cid in cs.hand:
        upgraded = cid.endswith("+1")
        base_id = cid[:-2] if upgraded else cid
        cost = 1
        try:
            card_def = ALL_CARDS.get(base_id)
            if card_def is not None:
                cost = int(getattr(card_def, "cost", 1) or 1)
        except Exception:  # noqa: BLE001
            pass
        hand_list.append({
            "name": base_id,
            "upgraded": upgraded,
            "cost": cost,
        })

    # ----- monsters -----
    monsters_list: List[Dict[str, Any]] = []
    for e in cs.enemies:
        if getattr(e, "is_dead", False):
            continue
        intent_dmg = int(getattr(e, "move_damage", 0) or 0)
        intent_hits = int(getattr(e, "move_hits", 1) or 1)
        monsters_list.append({
            "name": getattr(e, "name", "") or e.__class__.__name__,
            "hp": int(getattr(e, "hp", 0) or 0),
            "max_hp": int(getattr(e, "max_hp", 1) or 1),
            "intent_dmg": intent_dmg,
            "intent_hits": intent_hits,
            "block": int(getattr(e, "block", 0) or 0),
        })

    # ----- piles -----
    draw_pile = []
    for cid in getattr(cs, "draw_pile", []) or []:
        upgraded = cid.endswith("+1")
        draw_pile.append({"name": cid[:-2] if upgraded else cid, "upgraded": upgraded})
    discard_pile = []
    for cid in getattr(cs, "discard_pile", []) or []:
        upgraded = cid.endswith("+1")
        discard_pile.append({"name": cid[:-2] if upgraded else cid, "upgraded": upgraded})

    state = V8State(
        hp=int(getattr(player, "hp", 0) or 0),
        max_hp=int(getattr(player, "max_hp", 1) or 1),
        floor=1,
        act=1,
        gold=0,
        deck=list(deck_dicts),
        relics=list(relics),
        in_combat=True,
        phase="COMBAT",
        hand=hand_list,
        draw_pile=draw_pile,
        discard_pile=discard_pile,
        energy=int(getattr(cs, "energy", 0) or 0),
        monsters=monsters_list,
    )
    return state


def _plan_to_action_idx(plan, hand_size: int, monster_count: int) -> Optional[int]:
    """把 search 返回 plan[0] 映射到 model._combat_forward 的 action_idx。

    action_idx 编码（与 v8/model.py:562-643 对齐）：
      * PlayCard(card_idx, target_idx): action_idx = card_idx * M + target_idx
        （target_idx == -1 时映射到 0，因为 model cross product 必须有 target slot）
      * EndTurn: action_idx = H * M（最后一个 slot）
      * UsePotion / SelectScryDiscard / 其他: 返回 None（model 不学，跳过）

    Args:
        plan: list of engine actions（PlayCard / EndTurn / UsePotion / ...）
        hand_size: 当前手牌张数 H（model._combat_forward 的 H）
        monster_count: 当前活敌数 M（model._combat_forward 的 M）

    Returns:
        action_idx int 或 None（跳过此 turn）
    """
    if not plan:
        return None

    # 局部 import 避免顶层 import StSRLSolver
    from packages.engine.state.combat import PlayCard, EndTurn, UsePotion

    a = plan[0]
    M = max(1, monster_count)

    if isinstance(a, EndTurn):
        # H == 0 (空手) 或 M == 0 (无敌) 时 model 输出只有 1 维：end slot 在 idx=0
        if hand_size == 0 or monster_count == 0:
            return 0
        return hand_size * M  # 最后 slot

    if isinstance(a, PlayCard):
        if hand_size == 0 or monster_count == 0:
            return None  # 不应发生，但保险跳过
        card_idx = int(a.card_idx)
        target_idx = int(a.target_idx)
        if card_idx < 0 or card_idx >= hand_size:
            return None
        # target_idx == -1（self / 无 target 卡，比如 Defend）→ 映射到第 0 个 monster slot
        # （model 设计上每张卡都要 cross 所有 monster；自伤卡的 target 槽是冗余的，
        #  CE loss 让 model 学到 self-target 卡无差别选 0 即可）
        if target_idx < 0 or target_idx >= monster_count:
            target_idx = 0
        return card_idx * M + target_idx

    # UsePotion / SelectScryDiscard / 其他 → 跳过
    return None


def _run_one_self_play_battle(
    seed: int,
    enemy_id: str,
    starting_hp: int = 75,
    max_hp: int = 75,
    search_budget_s: float = 5.0,
    max_turns: int = 50,
) -> BattleRecord:
    """跑一场 self-play 战斗，收 BattleRecord（含 turn-level state + action_idx）。

    流程（与 deck_evaluator._simulate_combat 同套引擎，但加了 per-turn snapshot）：
      1. 构造 deck（Ironclad 起始 + 5-15 张随机 act1 卡）
      2. create_combat_from_enemies + start_combat
      3. while not over:
         a. snapshot engine.state → V8State
         b. solver.solve_turn(engine) → plan
         c. plan[0] → action_idx（PlayCard/EndTurn 用上面映射函数）
         d. 把 (state, action_idx) 加到 turn_records（如果 action_idx 不是 None）
         e. 执行 plan 里的全部 actions（与 _simulate_combat 一致）
      4. 写 outcome 字段
    """
    rng = random.Random(seed)
    deck = _build_random_deck(rng)
    relics: List[str] = []

    turn_records: List[Dict[str, Any]] = []
    damage_dealt = 0.0
    damage_taken = 0.0
    turns_played = 0
    victory = False

    try:
        # 复用 deck_evaluator 已经搭好的 import / mapping 基础
        from v8.deck_evaluator import (  # type: ignore
            _ensure_solver_imports,
            _deck_to_card_ids,
            _ENEMY_REGISTRY,
        )

        Random, TurnSolver, EndTurn, create_combat_from_enemies, _ALL = (
            _ensure_solver_imports()
        )
        # _ENEMY_REGISTRY 在 _ensure_solver_imports 内会被赋值
        from v8.deck_evaluator import _ENEMY_REGISTRY as ENEMY_REG  # type: ignore

        enemy_cls = (ENEMY_REG or {}).get(enemy_id)
        if enemy_cls is None:
            raise ValueError(f"unknown enemy_id={enemy_id!r}")

        card_ids = _deck_to_card_ids(deck)
        if not card_ids:
            raise ValueError("deck empty after name->id conversion")

        ai_rng = Random(1000 + seed)
        hp_rng = Random(2000 + seed)
        enemy = enemy_cls(ai_rng=ai_rng, ascension=0, hp_rng=hp_rng)

        engine = create_combat_from_enemies(
            enemies=[enemy],
            player_hp=starting_hp,
            player_max_hp=max_hp,
            deck=card_ids,
            energy=3,
            relics=list(relics),
            potions=[],
        )
        engine.start_combat()

        initial_player_hp = engine.state.player.hp
        initial_enemy_total_hp = sum(e.max_hp for e in engine.state.enemies)

        solver = TurnSolver(
            time_budget_ms=search_budget_s * 1000.0,
            node_budget=5000,
        )

        battle_id = f"seed{seed}_{enemy_id}"

        while not engine.is_combat_over() and turns_played < max_turns:
            # ----- 每个 plan-step 都 snapshot + record -----
            # solve_turn 返回的是整 turn 的 plan（多 action），但 model 学的是
            # 每个时刻的下一个 action。最简单做法：每次 plan 之前 snapshot，
            # 记 plan[0] 的 action_idx；其余 actions 在 engine 推进后下一次 snapshot
            # 时（plan 重算）会覆盖到。这里实现就是：snapshot → solve → 记 plan[0] →
            # 逐个 execute plan 全部 actions（高效，与 _simulate_combat 一致）。
            #
            # NOTE：理论上 model 应该学 plan 全过程（每 action 一个 record），但每
            # 步重算 solve_turn 太贵；折中：每 turn 只记 plan[0]（turn 起始决策）。
            # 这是 design doc 组件 6 的"per-turn record"语义。

            # snapshot 前的 hand / monsters 数（用来算 action_idx 编码）
            hand_size = len(engine.state.hand)
            living_monsters = [e for e in engine.state.enemies if not e.is_dead]
            monster_count = len(living_monsters)

            try:
                plan = solver.solve_turn(engine, room_type="monster")
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "solve_turn raised %s: %s (seed=%d enemy=%s)",
                    type(e).__name__, e, seed, enemy_id,
                )
                plan = None

            if plan and len(plan) > 0:
                # 先 snapshot 当前 state（plan[0] 决策点）
                state_snapshot = _engine_to_v8_state(engine, deck, relics)
                action_idx = _plan_to_action_idx(plan, hand_size, monster_count)
                if action_idx is not None:
                    turn_records.append({
                        "state": state_snapshot.to_dict(),
                        "search_action_idx": int(action_idx),
                        "available_actions_count": int(hand_size * max(1, monster_count) + 1),
                        "battle_id": battle_id,
                    })

                # 执行 plan（与 _simulate_combat 一致）
                ended = False
                for action in plan:
                    if engine.is_combat_over():
                        break
                    try:
                        engine.execute_action(action)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "execute_action(%r) failed: %s", action, e,
                        )
                        break
                    if isinstance(action, EndTurn):
                        ended = True
                if not ended and not engine.is_combat_over():
                    try:
                        engine.execute_action(EndTurn())
                    except Exception as e:  # noqa: BLE001
                        logger.warning("execute_action(EndTurn after plan) failed: %s", e)
                        break
            else:
                # 没招了：snapshot + 记 EndTurn record，然后 EndTurn
                state_snapshot = _engine_to_v8_state(engine, deck, relics)
                if hand_size == 0 or monster_count == 0:
                    end_idx = 0
                else:
                    end_idx = hand_size * max(1, monster_count)
                turn_records.append({
                    "state": state_snapshot.to_dict(),
                    "search_action_idx": int(end_idx),
                    "available_actions_count": int(hand_size * max(1, monster_count) + 1),
                    "battle_id": battle_id,
                })
                try:
                    engine.execute_action(EndTurn())
                except Exception as e:  # noqa: BLE001
                    logger.warning("execute_action(EndTurn) failed: %s", e)
                    break
            turns_played += 1

        final_player_hp = max(0, engine.state.player.hp)
        final_enemy_total_hp = sum(max(0, e.hp) for e in engine.state.enemies)
        damage_dealt = float(max(0, initial_enemy_total_hp - final_enemy_total_hp))
        damage_taken = float(max(0, initial_player_hp - final_player_hp))
        victory = bool(engine.is_victory()) if engine.is_combat_over() else False

    except Exception as e:  # noqa: BLE001
        logger.warning(
            "self-play battle failed seed=%d enemy=%s: %s: %s",
            seed, enemy_id, type(e).__name__, e,
        )

    return BattleRecord(
        seed=seed,
        enemy_id=enemy_id,
        starting_deck=deck,
        relics=relics,
        starting_hp=starting_hp,
        max_hp=max_hp,
        damage_dealt=damage_dealt,
        damage_taken=damage_taken,
        turns_played=turns_played,
        victory=victory,
        turn_records=turn_records,
    )


def collect_pretrain_data(
    num_battles: int = DEFAULT_NUM_BATTLES,
    output_path: str = DEFAULT_OUTPUT_PATH,
    enemies: Optional[List[str]] = None,
    seed_base: int = 0,
) -> str:
    """收集 self-play 战斗数据，写入 jsonl。

    Args:
        num_battles: 战斗场数（design doc 建议 ~5000 → 50000 turn 数据）
        output_path: 输出 jsonl 路径
        enemies: 可选敌人 id list（默认 act1 三敌：Cultist / Lagavulin / Hexaghost）
        seed_base: seed 起点（确定性可复现）

    Returns:
        实际写入的文件路径

    Round 2 实施完成：
        - turn_records 已接 TurnSolver per-turn snapshot + chosen idx
        - 每场战斗每 turn 一条 record（plan[0] 作为 model 学习目标）
    """
    if enemies is None:
        enemies = ["Cultist", "Lagavulin", "Hexaghost"]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(num_battles):
            enemy_id = enemies[i % len(enemies)]
            record = _run_one_self_play_battle(
                seed=seed_base + i,
                enemy_id=enemy_id,
            )
            f.write(record.to_json() + "\n")
            n_written += 1

    logger.info(
        "collect_pretrain_data: wrote %d records to %s", n_written, out_path,
    )
    return str(out_path)


# ============================================================
# 训练战斗 head
# ============================================================


class CombatPretrainDataset(Dataset):
    """读取 jsonl，flatten 成 list of (state_dict, action_idx)。

    TODO 直到 collect_pretrain_data 填实 turn_records，dataset 永远是空的。
    供 train_combat_head 框架先就位。
    """

    def __init__(self, jsonl_path: str):
        self.records: List[Tuple[Dict[str, Any], int]] = []
        path = Path(jsonl_path)
        if not path.exists():
            logger.warning("CombatPretrainDataset: %s does not exist", path)
            return

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turn_records = obj.get("turn_records", []) or []
                for tr in turn_records:
                    state_dict = tr.get("state")
                    # 兼容老字段名 action_idx；新字段 search_action_idx
                    action_idx = tr.get("search_action_idx", tr.get("action_idx"))
                    if state_dict is None or action_idx is None:
                        continue
                    self.records.append((state_dict, int(action_idx)))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], int]:
        return self.records[idx]


def _collate_keep_list(batch: List[Tuple[Dict[str, Any], int]]):
    """Collate：保持 list 形式（model.forward 需逐 sample 处理）。"""
    state_dicts = [b[0] for b in batch]
    action_idxs = [b[1] for b in batch]
    return state_dicts, action_idxs


def train_combat_head(
    data_path: str,
    model: V8Model,
    num_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cpu",
) -> Dict[str, float]:
    """用收集的数据训 model 的战斗 head。

    简化做法（design doc 备注）：训整个 model，但 loss 只在战斗 phase 的 step 上算
    （cross-entropy on search_chosen_idx）。其他 head（元决策 / value）此阶段不动。

    Returns:
        训练 metrics dict（avg loss / accuracy / 总 step 数）

    TODO：
        - 真训前确保 data_path 中 turn_records 已填实（collect_pretrain_data 实现完）
        - 若数据集为空，函数提前返回 metrics={"avg_loss": 0.0, "n_steps": 0}
          （让 smoke import 通过 + dummy forward 验证）
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.train()

    dataset = CombatPretrainDataset(data_path)
    if len(dataset) == 0:
        logger.warning(
            "train_combat_head: dataset empty (data_path=%s). "
            "Skipping real training; this is expected during smoke "
            "(collect_pretrain_data turn_records is stub).",
            data_path,
        )
        return {"avg_loss": 0.0, "accuracy": 0.0, "n_steps": 0}

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_keep_list
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_loss = 0.0
    total_correct = 0
    total_steps = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_steps = 0
        for state_dicts, action_idxs in loader:
            optimizer.zero_grad()
            batch_loss_terms = []
            for state_dict, action_idx in zip(state_dicts, action_idxs):
                state = V8State.from_dict(state_dict)
                # 战斗 head：只关心 phase=COMBAT 的样本
                if state.phase != "COMBAT":
                    continue
                # 战斗 head 不需要外部 actions list（用 hand × monsters）
                out = model(state, available_actions=[])
                logits = out["logits"]
                if logits.numel() == 0 or action_idx >= logits.shape[0]:
                    continue
                target = torch.tensor([action_idx], dtype=torch.long, device=dev)
                loss = F.cross_entropy(logits.unsqueeze(0), target)
                batch_loss_terms.append(loss)

                with torch.no_grad():
                    pred = int(torch.argmax(logits).item())
                    if pred == action_idx:
                        epoch_correct += 1
                epoch_steps += 1

            if not batch_loss_terms:
                continue
            batch_loss = torch.stack(batch_loss_terms).mean()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += float(batch_loss.item())

        if epoch_steps > 0:
            logger.info(
                "train_combat_head epoch=%d/%d loss=%.4f acc=%.3f steps=%d",
                epoch + 1, num_epochs,
                epoch_loss / max(1, epoch_steps),
                epoch_correct / max(1, epoch_steps),
                epoch_steps,
            )
        total_loss += epoch_loss
        total_correct += epoch_correct
        total_steps += epoch_steps

    return {
        "avg_loss": total_loss / max(1, total_steps),
        "accuracy": total_correct / max(1, total_steps),
        "n_steps": float(total_steps),
    }


# ============================================================
# Smoke entry（不真训，验证 import + 框架）
# ============================================================


def _smoke_forward_check(device: str = "cpu") -> Dict[str, float]:
    """构造 dummy combat V8State，跑 model.forward 验证战斗 head 通。

    smoke 用：CE backward 一次，确认梯度能流到战斗 head。
    """
    model = V8Model().to(torch.device(device))
    model.train()

    # 构造 minimal combat state
    state = V8State(
        hp=70, max_hp=80, floor=2, act=1, gold=0,
        deck=[
            {"name": "Strike", "upgraded": False},
            {"name": "Defend", "upgraded": False},
        ],
        relics=[],
        in_combat=True,
        phase="COMBAT",
        hand=[
            {"name": "Strike", "upgraded": False, "cost": 1},
            {"name": "Defend", "upgraded": False, "cost": 1},
        ],
        energy=3,
        monsters=[
            {"name": "Cultist", "hp": 50, "max_hp": 50,
             "intent_dmg": 6, "intent_hits": 1, "block": 0},
        ],
    )

    out = model(state, available_actions=[])
    logits = out["logits"]
    if logits.numel() == 0:
        raise RuntimeError("smoke: combat head returned empty logits")
    target = torch.tensor([0], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits.unsqueeze(0), target)

    optimizer = AdamW(model.parameters(), lr=3e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "smoke_loss": float(loss.item()),
        "logits_shape": float(logits.shape[0]),
    }


__all__ = [
    "BattleRecord",
    "collect_pretrain_data",
    "train_combat_head",
    "CombatPretrainDataset",
]


if __name__ == "__main__":
    # 不真训：默认走 smoke forward check。
    # 真要收数据 + 训练在用户授权 smoke 时再启用：
    #   python -m v8.pretrain --collect 100  # collect data
    #   python -m v8.pretrain --train data/v8_pretrain/battles.jsonl  # train head
    logging.basicConfig(level=logging.INFO)
    metrics = _smoke_forward_check()
    print("pretrain smoke OK:", metrics)
