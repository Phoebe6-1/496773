import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# Helpers
# =========================
def ordinal_rank_desc(values: dict) -> dict:
    """
    Return ordinal ranks (1=best) for a dict {name: value}, sorting by value desc.
    Ties are broken deterministically by name.
    """
    items = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))   # kv:key-value pair
    return {name: i + 1 for i, (name, _) in enumerate(items)}


def ordinal_rank_asc(values: dict) -> dict:
    """
    Return ordinal ranks (1=best) for a dict {name: value}, sorting by value asc.
    Ties are broken deterministically by name.
    """
    items = sorted(values.items(), key=lambda kv: (kv[1], kv[0]))
    return {name: i + 1 for i, (name, _) in enumerate(items)}


# =========================
# Core simulator
# =========================
def simulate_season(
    season_df: pd.DataFrame,
    method: str,               # 'percent' or 'rank'
    use_bottom2: bool,         # False -> direct eliminate worst k; True -> bottom2 + judges pick
    decay: float = 0.05,       # exponential decay for imputation beyond last observed week
):
    """
    Simulate one season week-by-week, keeping elimination counts per week identical to the data.

    Inputs columns needed:
      season, week, celebrity_name,
      judge_percent, fan_vote_percent,
      judge_rank, fan_vote_rank

    Outputs:
      results_df: final placement + eliminated_week
      weekly_df:  per-week simulated metrics for all alive contestants
      elimination_log: list of (week, eliminated_name)
    """
    season_df = season_df.copy()
    season = int(season_df["season"].iloc[0])
    season_df["celebrity_name"] = season_df["celebrity_name"].astype(str)

    weeks = sorted(season_df["week"].unique())
    # how many contestants per week in the original data
    week_counts = season_df.groupby("week")["celebrity_name"].nunique().sort_index()

    # elimination count schedule: k_w = count(w) - count(w+1)
    elim_k = {w: int(week_counts.loc[w] - week_counts.loc[w + 1]) for w in weeks[:-1]}

    # initialize alive set from week 1
    alive = set(season_df.loc[season_df["week"] == weeks[0], "celebrity_name"].tolist())

    # store last observed raw values for imputation
    last_state = {name: {"rawJ": None, "rawF": None, "week": None} for name in alive}

    weekly_rows = []
    elimination_log = []

    for w in weeks:
        obs = season_df[season_df["week"] == w].set_index("celebrity_name")
        N_obs = len(obs)

        # compute raw judge/fan for all alive (observed -> from data; missing -> impute)
        rawJ = {}
        rawF = {}
        for name in sorted(alive):
            if name in obs.index:
                rj = float(obs.loc[name, "judge_percent"]) * N_obs
                rf = float(obs.loc[name, "fan_vote_percent"]) * N_obs
                last_state[name] = {"rawJ": rj, "rawF": rf, "week": w}
            else:
                # impute from last seen
                if last_state.get(name, {}).get("week") is None:
                    # extremely rare fallback
                    base_rj, base_rf, base_w = 1.0, 1.0, w - 1
                else:
                    base_rj = last_state[name]["rawJ"]
                    base_rf = last_state[name]["rawF"]
                    base_w = last_state[name]["week"]
                dt = w - base_w
                rj = base_rj * math.exp(-decay * dt)
                rf = base_rf * math.exp(-decay * dt)

            rawJ[name] = max(rj, 1e-12)
            rawF[name] = max(rf, 1e-12)

        # re-normalize into simulated percents
        sumJ = sum(rawJ.values())
        sumF = sum(rawF.values())
        judge_pct = {n: rawJ[n] / sumJ for n in alive}
        fan_pct = {n: rawF[n] / sumF for n in alive}

        # simulated ranks (1=best)
        judge_rank = ordinal_rank_desc(judge_pct)   # higher percent -> better rank
        fan_rank = ordinal_rank_desc(fan_pct)

        # combined metrics
        comb_percent = {n: judge_pct[n] + fan_pct[n] for n in alive}
        comb_rank_sum = {n: judge_rank[n] + fan_rank[n] for n in alive}

        # overall rank under the chosen method (1=best)
        if method == "percent":
            overall_rank = ordinal_rank_desc(comb_percent)
        elif method == "rank":
            overall_rank = ordinal_rank_asc(comb_rank_sum)
        else:
            raise ValueError("method must be 'percent' or 'rank'")

        # record weekly metrics for current alive set
        for n in alive:
            weekly_rows.append(
                {
                    "season": season,
                    "week": int(w),
                    "celebrity_name": n,
                    "observed_in_data": int(n in obs.index),
                    "judge_percent_sim": judge_pct[n],
                    "fan_vote_percent_sim": fan_pct[n],
                    "judge_rank_sim": judge_rank[n],
                    "fan_vote_rank_sim": fan_rank[n],
                    "combined_percent_sim": comb_percent[n],
                    "combined_rank_sum_sim": comb_rank_sum[n],
                    "overall_rank_sim": overall_rank[n],
                }
            )

        # elimination step (skip final week; final week just ranks finalists)
        if w == weeks[-1]:
            continue

        k = elim_k.get(w, 0)
        if k <= 0:
            continue

        # choose eliminations
        if not use_bottom2:
            # direct eliminate worst k by the method's combined metric
            if method == "percent":
                # worst = smallest combined_percent
                ordered = sorted(
                    list(alive),
                    key=lambda n: (comb_percent[n], fan_pct[n], judge_pct[n], n),
                )
                elim_names = ordered[:k]
            else:
                # worst = largest combined_rank_sum
                ordered = sorted(
                    list(alive),
                    key=lambda n: (-comb_rank_sum[n], -fan_rank[n], -judge_rank[n], n),
                )
                elim_names = ordered[:k]

            for en in elim_names:
                elimination_log.append((int(w), en))
                alive.remove(en)

        else:
            # bottom2 + judges pick, repeated k times if double elimination
            for _ in range(k):
                if len(alive) <= 1:
                    break

                if method == "percent":
                    bottom2 = sorted(
                        list(alive),
                        key=lambda n: (comb_percent[n], fan_pct[n], judge_pct[n], n),
                    )[:2]
                    # judges eliminate the one with WORSE judge performance (smaller judge_pct)
                    elim = sorted(
                        bottom2, key=lambda n: (judge_pct[n], fan_pct[n], comb_percent[n], n)
                    )[0]
                else:
                    bottom2 = sorted(
                        list(alive),
                        key=lambda n: (-comb_rank_sum[n], -fan_rank[n], -judge_rank[n], n),
                    )[:2]
                    # judges eliminate the one with WORSE judge rank (larger judge_rank number)
                    elim = sorted(
                        bottom2, key=lambda n: (-judge_rank[n], -fan_rank[n], comb_rank_sum[n], n)
                    )[0]

                elimination_log.append((int(w), elim))
                alive.remove(elim)

    # build weekly dataframe
    weekly_df = pd.DataFrame(weekly_rows)

    # final placements:
    # - finalists are those still alive at final week
    final_week = int(weeks[-1])
    final_rows = weekly_df[(weekly_df["week"] == final_week) & (weekly_df["celebrity_name"].isin(alive))].copy()

    if method == "percent":
        final_rows = final_rows.sort_values(["combined_percent_sim", "celebrity_name"], ascending=[False, True])
    else:
        final_rows = final_rows.sort_values(["combined_rank_sum_sim", "celebrity_name"], ascending=[True, True])

    placement = {}
    for i, name in enumerate(final_rows["celebrity_name"].tolist(), start=1):
        placement[name] = i

    # eliminated contestants: reverse elimination order -> later eliminated => better placement
    next_place = len(final_rows) + 1
    for w, en in reversed(elimination_log):
        placement[en] = next_place
        next_place += 1

    elim_week_map = {en: w for w, en in elimination_log}  # eliminated week for each eliminated contestant

    results_df = pd.DataFrame(
        {
            "season": season,
            "celebrity_name": list(placement.keys()),
            "placement_sim": list(placement.values()),
            "eliminated_week_sim": [elim_week_map.get(n, final_week) for n in placement.keys()],
            "method": method,
            "use_bottom2": use_bottom2,
        }
    ).sort_values("placement_sim")

    return results_df, weekly_df, elimination_log


# =========================
# Main: run all seasons & methods
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="contestants.csv", help="Path to contestants.csv")
    parser.add_argument("--out", type=str, default="output_sim", help="Output directory")
    parser.add_argument("--decay", type=float, default=0.05, help="Imputation decay (0=no decay)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    needed = [
        "season", "week", "celebrity_name",
        "judge_percent", "judge_rank",
        "fan_vote_percent", "fan_vote_rank",
    ]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in CSV: {miss}")

    # The 6 controversy targets you listed
    targets = {
        2: "Jerry Rice",
        4: "Billy Ray Cyrus",
        11: "Bristol Palin",
        17: "Bill Engvall",
        27: "Bobby Bones",
        29: "Nelly",
    }

    all_summaries = []
    all_target_weekly = []

    method_specs = [
        ("percent", False, "percent"),
        ("rank", False, "rank"),
        ("percent", True, "percent_bottom2"),
        ("rank", True, "rank_bottom2"),
    ]

    for season in sorted(df["season"].unique()):
        sdf = df[df["season"] == season].copy()

        for method, use_bottom2, tag in method_specs:
            results_df, weekly_df, elim_log = simulate_season(
                sdf, method=method, use_bottom2=use_bottom2, decay=args.decay
            )

            # save full outputs
            results_df.to_csv(out_dir / f"results_season{season}_{tag}.csv", index=False)
            weekly_df.to_csv(out_dir / f"weekly_metrics_season{season}_{tag}.csv", index=False)

            # record target summary
            tname = targets.get(int(season))
            if tname is not None:
                row = results_df[results_df["celebrity_name"] == tname].iloc[0].to_dict()
                row["tag"] = tag
                all_summaries.append(row)

                # weekly trajectory for this target
                t_week = weekly_df[weekly_df["celebrity_name"] == tname].copy()
                t_week["tag"] = tag
                all_target_weekly.append(t_week)

    summary_df = pd.DataFrame(all_summaries).sort_values(["season", "tag"])
    summary_df.to_csv(out_dir / "controversy_summary.csv", index=False)

    target_weekly_df = pd.concat(all_target_weekly, ignore_index=True) if all_target_weekly else pd.DataFrame()
    target_weekly_df.to_csv(out_dir / "targets_weekly_trajectory.csv", index=False)

    # print a compact view
    print("\n=== Controversy summary (placement & eliminated week under each method) ===")
    show_cols = ["season", "celebrity_name", "tag", "placement_sim", "eliminated_week_sim"]
    print(summary_df[show_cols].to_string(index=False))

    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


