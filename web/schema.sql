-- V8 RL 训练数据可视化 SQLite schema
-- episode-centric 设计，不再有 batch 维度。所有表设计为 DELETE+INSERT 幂等。
-- 2026-05-25: v3 schema — 去掉 batches 表及所有 batch_id 列，PK 以 ep 为核心。
-- 2026-05-20: floor schema 修复——floor 在 STS 日志里是按 Act 重置 0-17，
-- 不是绝对楼层。floor_events / combats / event_logs / deck_snapshots / deck_cards
-- 五张表 PK 必须含 act 维度，否则 Act 2/3 会覆盖 Act 1 数据（之前丢失 ~2/3 数据）。

-- ============ Mapping（英中翻译，P4 才填充，先建表） ============

CREATE TABLE IF NOT EXISTS i18n_entries (
    kind     TEXT NOT NULL,  -- 'card'|'relic'|'event'|'monster'|'potion'
    en_id    TEXT NOT NULL,  -- 引擎内 ID（如 Strike_R / SlimeBoss / Mushrooms）
    en_name  TEXT,           -- 英文 display name（兜底）
    zh_name  TEXT,           -- 来自 jar 简中本地化
    zh_desc  TEXT,           -- 中文描述
    source   TEXT NOT NULL,  -- 'jar_zho'|'content_py'|'manual'
    raw_json TEXT,           -- 原始 mapping json（含 UPGRADE_DESCRIPTION 等额外字段）
    meta_json TEXT,          -- content_py 提取的额外元数据（cards: rarity / card_type / cost）
    PRIMARY KEY (kind, en_id)
);

CREATE INDEX IF NOT EXISTS idx_i18n_kind ON i18n_entries(kind);

CREATE VIEW IF NOT EXISTS v_cards_zh AS
    SELECT en_id, en_name, zh_name, zh_desc, source FROM i18n_entries WHERE kind = 'card';

CREATE VIEW IF NOT EXISTS v_relics_zh AS
    SELECT en_id, en_name, zh_name, zh_desc, source FROM i18n_entries WHERE kind = 'relic';

CREATE VIEW IF NOT EXISTS v_events_zh AS
    SELECT en_id, en_name, zh_name, zh_desc, source FROM i18n_entries WHERE kind = 'event';

CREATE VIEW IF NOT EXISTS v_monsters_zh AS
    SELECT en_id, en_name, zh_name, zh_desc, source FROM i18n_entries WHERE kind = 'monster';

CREATE VIEW IF NOT EXISTS v_potions_zh AS
    SELECT en_id, en_name, zh_name, zh_desc, source FROM i18n_entries WHERE kind = 'potion';

-- ============ ETL 内部（scan 增量判定） ============

-- ETL scan 用，记录每个源目录的扫描状态，不暴露给前端
CREATE TABLE IF NOT EXISTS etl_metadata (
    source_key    TEXT PRIMARY KEY,
    batch_dir     TEXT NOT NULL,
    log_path      TEXT,
    summary_mtime REAL,
    scanned_at    TEXT NOT NULL
);

-- ============ 训练 / Eval 数据（P1-P2） ============

-- train_log_tail[] 展平，每个 batch_size 步长一行
CREATE TABLE IF NOT EXISTS training_metrics (
    episodes_done       INTEGER PRIMARY KEY,
    batch_size          INTEGER,
    mean_reward         REAL,
    mean_steps          REAL,
    mean_floor          REAL,
    beat_boss_count     INTEGER,
    policy_loss         REAL,
    value_loss          REAL,
    entropy             REAL,
    approx_kl           REAL,
    clip_frac           REAL,
    update_secs         REAL,
    wrapper_calls       INTEGER
);

-- eval_history[] 展平
CREATE TABLE IF NOT EXISTS eval_results (
    episodes_done         INTEGER PRIMARY KEY,
    num_seeds             INTEGER,
    completed             INTEGER,
    reached_boss_rate     REAL,
    act1_boss_beat_rate   REAL,
    act2_boss_beat_rate   REAL,
    won_game_rate         REAL,
    beat_boss_rate        REAL,            -- deprecated（= act1_boss_beat_rate）
    floor_mean            REAL,
    floor_max             INTEGER,
    avg_steps             REAL,
    secs                  REAL
);

-- boss_kill_counts / boss_reach_counts 字典并集展平
CREATE TABLE IF NOT EXISTS eval_boss_counts (
    episodes_done  INTEGER NOT NULL,
    boss_en_id     TEXT NOT NULL,        -- 原始 boss 名（带空格如 'Slime Boss'）
    reach_count    INTEGER DEFAULT 0,
    kill_count     INTEGER DEFAULT 0,
    PRIMARY KEY (episodes_done, boss_en_id),
    FOREIGN KEY (episodes_done) REFERENCES eval_results(episodes_done) ON DELETE CASCADE
);

-- ============ 单局回放（P3） ============

-- 单局 episode 汇总（来自 [heartbeat]）
CREATE TABLE IF NOT EXISTS episodes (
    ep              INTEGER PRIMARY KEY,
    steps           INTEGER,
    reward          REAL,
    floor_reached   INTEGER,
    beat_boss       INTEGER,            -- 0/1
    secs            REAL,
    search_calls    INTEGER,
    eval_deck_calls INTEGER
);

CREATE INDEX IF NOT EXISTS idx_episodes_beat_boss ON episodes(beat_boss);

-- 楼层级事件（来自 [floor]）
-- PK 必须含 act：floor 0-17 在每个 Act 重置，跨 act 同 floor 数字是不同房间。
CREATE TABLE IF NOT EXISTS floor_events (
    ep         INTEGER NOT NULL,
    act        INTEGER NOT NULL,
    floor      INTEGER NOT NULL,
    abs_floor  INTEGER,                 -- 跨 act 绝对楼层 (act-1)*17+floor，act1: 0..17, act2: 18..34, act3: 35..51
    hp_cur     INTEGER,
    hp_max     INTEGER,
    room       TEXT,                    -- monster/elite/boss/rest/shop/event/treasure
    PRIMARY KEY (ep, act, floor),
    FOREIGN KEY (ep) REFERENCES episodes(ep) ON DELETE CASCADE
);

-- 战斗（来自 [combat] enter/exit 合并）
CREATE TABLE IF NOT EXISTS combats (
    ep              INTEGER NOT NULL,
    act             INTEGER NOT NULL,
    floor           INTEGER NOT NULL,
    abs_floor       INTEGER,            -- 跨 act 绝对楼层 (act-1)*17+floor
    room            TEXT,
    enemies_json    TEXT,               -- JSON array of enemy en_id
    hp_before       INTEGER,
    hp_after        INTEGER,
    turn_actions    INTEGER,
    exit_reason     TEXT,
    PRIMARY KEY (ep, act, floor),
    FOREIGN KEY (ep) REFERENCES episodes(ep) ON DELETE CASCADE
);

-- 事件（来自 [event] enter/choice/exit 合并）
CREATE TABLE IF NOT EXISTS event_logs (
    ep            INTEGER NOT NULL,
    act           INTEGER NOT NULL,
    floor         INTEGER NOT NULL,
    abs_floor     INTEGER,              -- 跨 act 绝对楼层 (act-1)*17+floor
    event_id      TEXT,
    event_phase   TEXT,
    choices_json  TEXT,                 -- 进入时可见 choices JSON
    chosen_idx    INTEGER,
    chosen_text   TEXT,
    exit_reason   TEXT,
    PRIMARY KEY (ep, act, floor),
    FOREIGN KEY (ep) REFERENCES episodes(ep) ON DELETE CASCADE
);

-- 元决策（来自 [meta]）
-- act 列 2026-05-26 追加：floor 在 STS 里按 Act 重置 0-17，缺 act 会导致跨 Act 同 floor 碰撞。
CREATE TABLE IF NOT EXISTS meta_decisions (
    ep            INTEGER NOT NULL,
    step          INTEGER NOT NULL,
    act           INTEGER NOT NULL DEFAULT 1,
    phase         TEXT    NOT NULL,   -- CARD_REWARDS, REST, SHOP, MAP, BOSS_REWARDS, NEOW, TREASURE
    floor         INTEGER NOT NULL,
    abs_floor     INTEGER,            -- 跨 act 绝对楼层 (act-1)*17+floor
    chosen        TEXT    NOT NULL,   -- raw chosen label
    chosen_card   TEXT,               -- extracted card_en_id (for CARD_REWARDS/REST upgrade)
    decision_type TEXT,               -- pick/skip (CARD_REWARDS), rest/upgrade (REST), buy/remove/leave (SHOP), etc.
    options_json  TEXT,               -- JSON array of option labels
    PRIMARY KEY (ep, step),
    FOREIGN KEY (ep) REFERENCES episodes(ep) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_meta_decisions_ep ON meta_decisions(ep);
CREATE INDEX IF NOT EXISTS idx_meta_decisions_phase ON meta_decisions(ep, phase);

-- 牌组快照（来自 [deck]）
CREATE TABLE IF NOT EXISTS deck_snapshots (
    ep          INTEGER NOT NULL,
    act         INTEGER NOT NULL,
    floor       INTEGER NOT NULL,
    abs_floor   INTEGER,                -- 跨 act 绝对楼层 (act-1)*17+floor
    room        TEXT NOT NULL,
    cards_raw   TEXT,                   -- 原始 Counter 字符串（带 +1 升级标记）
    relics_raw  TEXT,
    deck_size   INTEGER,
    PRIMARY KEY (ep, act, floor, room),
    FOREIGN KEY (ep) REFERENCES episodes(ep) ON DELETE CASCADE
);

-- 牌组展平后单卡（每个 snapshot 多行，含升级标记）
CREATE TABLE IF NOT EXISTS deck_cards (
    ep          INTEGER NOT NULL,
    act         INTEGER NOT NULL,
    floor       INTEGER NOT NULL,
    abs_floor   INTEGER,                -- 跨 act 绝对楼层 (act-1)*17+floor
    card_en_id  TEXT NOT NULL,
    upgraded    INTEGER DEFAULT 0,      -- 0/1
    count       INTEGER DEFAULT 1,
    PRIMARY KEY (ep, act, floor, card_en_id, upgraded),
    FOREIGN KEY (ep) REFERENCES episodes(ep) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_deck_cards_ep ON deck_cards(ep);
CREATE INDEX IF NOT EXISTS idx_deck_cards_card ON deck_cards(card_en_id);

-- ============ 战斗回合详情（trace 导入） ============

-- 每场战斗每回合 snapshot
CREATE TABLE IF NOT EXISTS combat_turns (
    ep                  INTEGER NOT NULL,
    act                 INTEGER NOT NULL,
    floor               INTEGER NOT NULL,
    abs_floor           INTEGER,            -- 跨 act 绝对楼层（floor + (act-1)*17）
    turn_num            INTEGER NOT NULL,
    energy              INTEGER,
    energy_max          INTEGER,
    hp_cur              INTEGER,
    hp_max              INTEGER,
    block               INTEGER,
    player_buffs_raw    TEXT,               -- 原始 buffs 字串，如 "Metallicize=6 Vulnerable=1"
    hand_raw            TEXT,               -- 原始 hand 文本（slot/卡名/cost 全在）
    hand_json           TEXT,               -- [{slot, card_en_id, cost}] 解析后 JSON
    enemies_state_json  TEXT,               -- [{idx, en_id, hp, hp_max, block, buffs, intent_raw}]
    PRIMARY KEY (ep, act, floor, turn_num),
    FOREIGN KEY (ep) REFERENCES episodes(ep) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_combat_turns_combat ON combat_turns(ep, act, floor);

-- 每回合每次 action（出牌 / end_turn）
CREATE TABLE IF NOT EXISTS combat_card_plays (
    ep                  INTEGER NOT NULL,
    act                 INTEGER NOT NULL,
    floor               INTEGER NOT NULL,
    abs_floor           INTEGER,
    turn_num            INTEGER NOT NULL,
    action_idx          INTEGER NOT NULL,   -- 在 turn 内的递增序号
    action_type         TEXT NOT NULL,      -- 'play_card' / 'end_turn'
    card_en_id          TEXT,               -- play_card 时填
    hand_slot           INTEGER,            -- 出牌时手牌槽位
    target_idx          INTEGER,            -- enemy 索引（如有）
    target_en_id        TEXT,               -- 敌人 en_id
    player_delta_raw    TEXT,               -- 原 player_d 行
    enemy_delta_raw     TEXT,               -- 原 enemy_d 行
    PRIMARY KEY (ep, act, floor, turn_num, action_idx),
    FOREIGN KEY (ep, act, floor, turn_num) REFERENCES combat_turns(ep, act, floor, turn_num) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_card_plays_combat ON combat_card_plays(ep, act, floor);
CREATE INDEX IF NOT EXISTS idx_card_plays_card ON combat_card_plays(card_en_id);
