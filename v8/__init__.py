"""V8 RL 实施。

按 docs/v8_design_principles.md 的 3 点原话设计：
- 路线（MAP）：战损 + 牌组强度
- 战斗内：搜索 + model 学顺序/联动/target
- 选卡（CARD_REWARDS）：post-battle 多维评分

本期组件：
- state.py：V8State 数据结构（model 看的字段清单）
- action_space.py：V8Action schema + 各 phase action 枚举（stub，待 RL env 实现）
- deck_evaluator.py：牌组强度评估接口（stub，待 StSRLSolver 直调实现）
"""
