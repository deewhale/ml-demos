# StSRLSolver 引擎卡牌坏卡台账

由卡牌行为测试 harness（独立 wiki oracle）发现。红灯 = 引擎实现与真实 STS 不符。
病根多为：卡定义有 effect 字符串，但 combat_engine.py 的 play_card 无对应 dispatch 分支 → 效果静默丢弃。

| 卡 | 引擎实际 | wiki 期望 | 病根/effect串 | 出处 |
|---|---|---|---|---|
| Dark Shackles | 敌人无变化 | 敌人本回合 -9 力量 | apply_temp_strength_down 无 handler | wiki.gg/Dark_Shackles |
| Panacea | 玩家无变化 | 玩家 +1 Artifact | gain_artifact 无 handler | wiki.gg/Panacea |
| Panic Button | 仅加格挡 | +30格挡 + 2回合No Block | No Block debuff 无 handler | wiki.gg/Panic_Button |
| J.A.X. | HP/力量无变化 | -3 HP + 2 力量 | lose_3_hp_gain_strength 无 handler | wiki.gg/J.A.X. |
| Sword Boomerang | 敌人仅掉 3（单段） | 单敌 3×3=9（3 段随机目标，单敌全落同一目标） | effect 串 `random_enemy_x_times` 在 `_apply_card_damage` 无 dispatch 分支（仅有 `damage_x_times`/`damage_twice`/`damage_all_x_times`）→ hits 退回 1，只打基础伤害 1 次 | wiki.gg/Sword_Boomerang |
