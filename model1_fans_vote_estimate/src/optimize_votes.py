import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

from rules import soft_rank, rule_type_for_season

# =========================
# IO CONFIG (改成你的路径)
# =========================
INPUT_CSV = r"model1_fans_vote_estimate\data\long_input.csv"
OUT_DIR = Path(r"model1_fans_vote_estimate\output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Utility: indexing
# =========================
def build_indices(df_s: pd.DataFrame):
    keys = list(
        df_s.sort_values(["week", "celebrity_name"])[["week", "celebrity_name"]]
        .itertuples(index=False, name=None)
    )
    idx = {k: i for i, k in enumerate(keys)}
    return keys, idx

def week_positions(df_s: pd.DataFrame, idx: dict):
    wk_pos = {}
    for w, g in df_s.groupby("week"):
        wk_pos[int(w)] = [idx[(int(w), name)] for name in g["celebrity_name"].tolist()]
    return wk_pos

# =========================
# Objective: entropy + smoothness
# =========================
def objective_factory(df_s: pd.DataFrame, idx: dict, lambda_smooth: float = 10.0):
    eps = 1e-12
    week_to_names = df_s.groupby("week")["celebrity_name"].apply(lambda s: set(s)).to_dict()
    weeks = sorted(week_to_names.keys())

    def obj(x):
        # max-entropy regularization (minimize sum p log p)
        ent = np.sum(x * np.log(x + eps))

        # temporal smoothness on contestants present in both weeks
        smooth = 0.0
        for w_prev, w_now in zip(weeks[:-1], weeks[1:]):
            common = week_to_names[w_prev].intersection(week_to_names[w_now])
            for name in common:
                i_prev = idx[(w_prev, name)]
                i_now  = idx[(w_now,  name)]
                smooth += (x[i_now] - x[i_prev]) ** 2

        return ent + lambda_smooth * smooth

    return obj

# =========================
# Constraints (Percent seasons)
# Percent rule: combined = judge_percent + fan_percent
# eliminated should have the LOWEST combined percent
# =========================
def constraints_for_week_percent(wk_df: pd.DataFrame, idx: dict):
    elim = wk_df[wk_df["eliminated_this_week"] == 1]
    if elim.empty:
        return []

    cons = []
    w = int(wk_df["week"].iloc[0])

    # separate non-eliminated contestants (for multi-elim weeks)
    non_elim = wk_df[wk_df["eliminated_this_week"] != 1]

    for _, e in elim.iterrows():
        k_name = e["celebrity_name"]
        k_pos = idx[(w, k_name)]
        kJ = float(e["judge_percent"])

        for _, r in non_elim.iterrows():
            j_name = r["celebrity_name"]
            j_pos = idx[(w, j_name)]
            jJ = float(r["judge_percent"])

            # (jJ + Pj) - (kJ + Pk) >= 0
            cons.append({
                "type": "ineq",
                "fun": lambda x, j_pos=j_pos, k_pos=k_pos, jJ=jJ, kJ=kJ:
                    (jJ + x[j_pos]) - (kJ + x[k_pos])
            })

    return cons

# =========================
# Constraints (Rank seasons)
# Rank rule: combine by ranks (judge_rank + fan_rank)
# To keep it differentiable, fan_rank is approximated by soft_rank(P)
# eliminated should have the WORST combined rank metric (largest C)
# =========================
def constraints_for_week_rank(wk_df: pd.DataFrame, idx: dict, tau_rank: float = 0.05):
    elim = wk_df[wk_df["eliminated_this_week"] == 1]
    if elim.empty:
        return []

    cons = []
    w = int(wk_df["week"].iloc[0])

    names = wk_df["celebrity_name"].tolist()
    positions = [idx[(w, n)] for n in names]
    judge_r = wk_df["judge_rank"].to_numpy(dtype=float)

    # compare each eliminated contestant only against non-eliminated ones (handles multi-elim)
    non_elim_names = wk_df.loc[wk_df["eliminated_this_week"] != 1, "celebrity_name"].tolist()

    for _, e in elim.iterrows():
        k_name = e["celebrity_name"]
        k_local = names.index(k_name)

        for j_name in non_elim_names:
            j_local = names.index(j_name)

            # C_k - C_j >= 0  (k is worse)
            def ineq_fun(x, k_local=k_local, j_local=j_local,
                         judge_r=judge_r, positions=positions, tau_rank=tau_rank):
                v = x[positions]
                fan_r = soft_rank(v, tau=tau_rank)
                C = judge_r + fan_r
                return C[k_local] - C[j_local]

            cons.append({"type": "ineq", "fun": ineq_fun})

    return cons

# =========================
# Solve one season (方案一：S28+ 不用 bottom2 约束)
# =========================
def solve_one_season(df_s: pd.DataFrame, lambda_smooth: float = 10.0, tau_rank: float = 0.05):
    keys, idx = build_indices(df_s)
    wk_pos = week_positions(df_s, idx)

    # feasible init: uniform per week
    x0 = np.zeros(len(keys))
    for w, pos in wk_pos.items():
        x0[pos] = 1.0 / len(pos)

    bounds = [(0.0, 1.0)] * len(keys)

    cons = []
    # simplex per week
    for w, pos in wk_pos.items():
        cons.append({"type": "eq", "fun": lambda x, pos=pos: np.sum(x[pos]) - 1.0})

    season = int(df_s["season"].iloc[0])
    rtype = rule_type_for_season(season)

    # ---- key: percent seasons use percent constraints; rank seasons (including S28+) use rank constraints
    for _, wk_df in df_s.groupby("week"):
        if rtype == "S3-27_percent":
            cons += constraints_for_week_percent(wk_df, idx)
        else:
            # S1-2_rank and S28+_bottom2 both estimated under rank-based elimination constraints (no bottom2 constraint here)
            cons += constraints_for_week_rank(wk_df, idx, tau_rank=tau_rank)

    obj = objective_factory(df_s, idx, lambda_smooth=lambda_smooth)

    res = minimize(
        obj, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-10}
    )

    out = df_s.copy()
    out["estimated_fan_vote_share"] = out.apply(
        lambda r: res.x[idx[(int(r["week"]), r["celebrity_name"])]],
        axis=1
    )
    out["rule_type"] = rtype
    return out, res

# =========================
# Main
# =========================
def main():
    df = pd.read_csv(INPUT_CSV)

    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["eliminated_this_week"] = df["eliminated_this_week"].astype(int)

    all_out = []
    logs = []

    for season, df_s in df.groupby("season"):
        df_s = df_s.copy()

        out, res = solve_one_season(
            df_s,
            lambda_smooth=10.0,
            tau_rank=0.05,   # 建议>=0.05更稳
        )

        all_out.append(out)
        logs.append({
            "season": int(season),
            "rule_type": out["rule_type"].iloc[0],
            "success": bool(res.success),
            "message": res.message,
            "objective": float(res.fun),
            "n_vars": int(len(res.x)),
        })

    df_out = pd.concat(all_out, ignore_index=True)
    df_log = pd.DataFrame(logs)

    df_out.to_csv(OUT_DIR / "estimated_votes.csv", index=False)
    df_log.to_csv(OUT_DIR / "solver_log.csv", index=False)

if __name__ == "__main__":
    main()
