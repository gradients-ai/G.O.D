-- migrate:up
-- One row per contender in a paid place (2nd, 3rd) of a seeded text or quant tournament that completed with the boss kept:
-- that contender's average rank over rounds 1-3 against the boss's average over the same rounds. Written once, with the
-- placements, and read by the weight setter.
--   hotkey                   the contender.
--   final_position           its place (2 or 3).
--   contender_average_rank   its average rank over compared_round_numbers; NULL when no round ranks both it and the boss.
--   boss_average_rank        the boss's average rank over compared_round_numbers; NULL in the same case.
--   compared_round_numbers   the rounds of 1-3 in which both the contender and the boss have a rank.
--   share_to_boss            true when the boss's average is strictly lower than the contender's: the contender's share of
--                            the pool is paid like the boss's own share.
CREATE TABLE IF NOT EXISTS tournament_boss_rank_comparisons (
    tournament_id TEXT NOT NULL REFERENCES tournaments(tournament_id),
    hotkey TEXT NOT NULL,
    final_position INTEGER NOT NULL,
    contender_average_rank DOUBLE PRECISION,
    boss_average_rank DOUBLE PRECISION,
    compared_round_numbers INTEGER[] NOT NULL DEFAULT '{}',
    share_to_boss BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tournament_id, hotkey)
);

-- migrate:down
DROP TABLE IF EXISTS tournament_boss_rank_comparisons;
