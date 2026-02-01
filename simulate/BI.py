import math
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

CSV_IN = "all_data.csv"
CSV_OUT = "all_bias_index.csv"

df = pd.read_csv(CSV_IN)

def tau_b(a, b) -> float:
    """
    Kendall's tau-b with tie correction.
    若出现“全 tie”等导致 tau 不可定义（返回 nan），这里按 0 处理：
    表示该排序无法提供任何可比较的序信息。
    """
    r = kendalltau(a, b, variant="b", nan_policy="omit").correlation
    if r is None or np.isnan(r):
        return 0.0
    return float(r)

rows = []
for (season, week), g in df.groupby(["season", "week"], sort=True):
    n_w = len(g)
    omega = math.comb(n_w, 2) if n_w >= 2 else 0  # ω_w = C(n_w,2)

    # --- 三套“名次序列”（都用 1=最好，越小越好 的方向）---
    RJ = g["judge_rank"].to_numpy()
    RF = g["fan_vote_rank"].to_numpy()

    # Rank 规则的 composite ranking：直接用 total_rank（越小越好）
    RC_rank = g["total_rank"].to_numpy()

    # Percent 规则的 composite ranking：total_percent 越大越好 -> 转成名次(1最好)
    # ties 用 average（与 τ_b 的 tie-correction 一起是自洽的）
    RC_percent = rankdata(-g["total_percent"].to_numpy(), method="average")

    # --- τCJ(w), τCF(w), BI(w) ---
    tauCJ_rank = tau_b(RC_rank, RJ)
    tauCF_rank = tau_b(RC_rank, RF)
    BI_rank = (tauCF_rank - tauCJ_rank) / 2.0

    tauCJ_percent = tau_b(RC_percent, RJ)
    tauCF_percent = tau_b(RC_percent, RF)
    BI_percent = (tauCF_percent - tauCJ_percent) / 2.0

    rows.append({
        "season": season,
        "week": week,
        "n_w": n_w,
        "omega": omega,
        "tauCJ_rank": tauCJ_rank,
        "tauCF_rank": tauCF_rank,
        "BI_rank": BI_rank,
        "tauCJ_percent": tauCJ_percent,
        "tauCF_percent": tauCF_percent,
        "BI_percent": BI_percent,
        # judges 与 fans 的排序是否完全一致（用于构造 W_Δ）
        "judge_fan_identical": bool(np.all(RJ == RF)),
    })

out = pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)
out.to_csv(CSV_OUT, index=False)
print(f"Saved: {CSV_OUT}  (rows={len(out)})")

# ====== 可选：按图中(8)做跨周汇总（只对 W_Δ：judge & fan 不一致的周）======
mask = ~out["judge_fan_identical"]
if mask.any():
    BI_agg_rank = np.average(out.loc[mask, "BI_rank"], weights=out.loc[mask, "omega"])
    BI_agg_percent = np.average(out.loc[mask, "BI_percent"], weights=out.loc[mask, "omega"])
    print("Aggregated BI over W_Δ (weighted by C(n_w,2)):")
    print("  BI (rank rule)   =", BI_agg_rank)
    print("  BI (percent rule)=", BI_agg_percent)
else:
    print("No weeks in W_Δ within your filtered data.")
