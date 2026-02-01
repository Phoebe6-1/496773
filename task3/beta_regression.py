import re
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CSV_PATH = "t3_data.csv"  
OUT_DIR = Path("output_beta_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_SHRINK = 5.0     # ProStrength 平滑强度
EPS_CLIP = 1e-4         # Beta回归要求(0,1)，避免0/1

# 如果你的 fan share 已经是 per-season 一列，优先在这里写（脚本会自动尝试）
FAN_SEASON_COL_CANDIDATES = [
    "fan_share_season", "fan_vote_share_season", "fan_share", "vote_share_season", "P_season"
]

# 如果你的 fan share 是 per-week 多列（wide），尽量在这里写前缀模式（脚本会自动尝试）
# 例如：fan_share_w1, fan_share_w2, ... 或 P_w1, P_w2, ...
FAN_WEEK_REGEXES = [
    r"^fan_share_w(\d+)$",
    r"^fan_vote_share_w(\d+)$",
    r"^vote_share_w(\d+)$",
    r"^P_w(\d+)$",
    r"^P_week(\d+)$",
]

# =========================
# Helpers
# =========================
def parse_duration_event(results_str: str):
    """Return (T, E) where T is elimination week or NaN for finalist, E=1 if eliminated else 0."""
    s = str(results_str)
    m = re.search(r"Eliminated Week\s*(\d+)", s, re.IGNORECASE)
    if m:
        return int(m.group(1)), 1
    return np.nan, 0


def industry_group(x: str) -> str:
    """Map raw industry string to {Performer, Athlete, Media, Model, Other}."""
    s = str(x).lower()

    # Athlete
    if any(k in s for k in ["athlete", "sport", "nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "tennis", "olymp", "wrest", "ufc", "box"]):
        return "Athlete"

    # Model
    if "model" in s:
        return "Model"

    # Media
    if any(k in s for k in ["tv", "television", "host", "presenter", "journalist", "news", "radio", "personality", "reality", "influencer"]):
        return "Media"

    # Performer
    if any(k in s for k in ["actor", "actress", "singer", "musician", "rapper", "comedian", "dancer", "perform", "entertain"]):
        return "Performer"

    return "Other"


def clip01(y, eps=1e-4):
    """Clip into (0,1) for Beta regression."""
    y = np.asarray(y, dtype=float)
    return np.clip(y, eps, 1 - eps)


def find_fan_season_col(df: pd.DataFrame):
    for c in FAN_SEASON_COL_CANDIDATES:
        if c in df.columns:
            return c
    # heuristic: any column containing both 'fan' and 'share' and not week-specific
    for c in df.columns:
        cl = c.lower()
        if ("fan" in cl or "vote" in cl) and ("share" in cl or "prob" in cl) and ("week" not in cl and "_w" not in cl):
            return c
    return None


def find_fan_week_cols(df: pd.DataFrame):
    cols = []
    week_nums = []
    for c in df.columns:
        for rgx in FAN_WEEK_REGEXES:
            m = re.match(rgx, c)
            if m:
                cols.append(c)
                week_nums.append(int(m.group(1)))
                break
    if cols:
        # sort by week number
        cols = [c for _, c in sorted(zip(week_nums, cols), key=lambda x: x[0])]
        week_nums = sorted(week_nums)
        return cols, week_nums

    # heuristic fallback: columns containing fan/vote and week number
    cand = []
    for c in df.columns:
        cl = c.lower()
        m = re.search(r"(week|_w)(\d+)", cl)
        if m and ("fan" in cl or "vote" in cl or cl.startswith("p_") or cl.startswith("pweek") or cl.startswith("p_w")):
            cand.append((int(m.group(2)), c))
    if cand:
        cand.sort()
        cols = [c for _, c in cand]
        week_nums = [w for w, _ in cand]
        return cols, week_nums

    return None, None


# =========================
# 1) Load
# =========================
df = pd.read_csv(CSV_PATH)

required_cols = ["season", "celebrity_name", "results",
                 "celebrity_homecountry/region", "ballroom_partner",
                 "celebrity_age_during_season", "celebrity_industry"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# =========================
# 2) Build (T, E) and season length W_s from judge columns
# =========================
judge_cols = [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]
if not judge_cols:
    raise ValueError("No judge score columns found (pattern: weekX_judgeY_score).")

week_nums = sorted({int(re.search(r"week(\d+)_", c).group(1)) for c in judge_cols})

df[["T_raw", "E"]] = df["results"].apply(lambda x: pd.Series(parse_duration_event(x)))

# infer W_s: last week that exists in that season (any non-NA across any contestant and any judge column)
def season_total_weeks(sub: pd.DataFrame):
    last = 0
    for w in week_nums:
        cols_w = [c for c in judge_cols if c.startswith(f"week{w}_")]
        if sub[cols_w].notna().any().any():
            last = w
    return last

W_s = df.groupby("season", as_index=False).apply(season_total_weeks)
W_s = W_s.rename(columns={None: "W_s"})
df = df.merge(W_s, on="season", how="left")

# set T: eliminated week if eliminated; else final week W_s
df["T"] = df["T_raw"]
df.loc[df["E"] == 0, "T"] = df.loc[df["E"] == 0, "W_s"]
df["T"] = df["T"].astype(int)

# normalized longevity
df["R"] = df["T"] / df["W_s"].replace(0, np.nan)

# =========================
# 3) X: Home region, ProStrength/Count (leave-one-season-out), Age z+z^2, Industry groups, Season dummies
# =========================
# US indicator
df["US"] = (df["celebrity_homecountry/region"].astype(str).str.lower() == "united states").astype(int)

# Age z, z^2
age = df["celebrity_age_during_season"].astype(float)
df["age_z"] = (age - age.mean()) / (age.std(ddof=0) + 1e-12)
df["age_z2"] = df["age_z"] ** 2

# Industry grouping one-hot
df["industry_grp"] = df["celebrity_industry"].apply(industry_group)
df = pd.get_dummies(df, columns=["industry_grp"], drop_first=True)

# Season one-hot
df = pd.get_dummies(df, columns=["season"], drop_first=True)

# ProStrength / ProCount: use other seasons only (exclude current season)
global_mean_R = df["R"].mean()

# build counts/means by (dancer, season) exclusion
# For each dancer p, compute overall sum/count of R, and sum/count within each season.
dancer = df["ballroom_partner"].astype(str)
df["_dancer"] = dancer

overall = df.groupby("_dancer")["R"].agg(["sum", "count"]).rename(columns={"sum": "sum_all", "count": "cnt_all"})
df = df.merge(overall, on="_dancer", how="left")

# season columns got one-hot, but we still need original season id for leave-one-season-out;
# reconstruct original season id from results? easiest: use pre-dummy season in a separate series:
# We saved season into one-hot, so before dummy we should keep it:
# (Workaround: re-read season from CSV)
season_raw = pd.read_csv(CSV_PATH)["season"]
df["_season_raw"] = season_raw.values

within = df.groupby(["_dancer", "_season_raw"])["R"].agg(["sum", "count"]).rename(columns={"sum":"sum_in_s", "count":"cnt_in_s"})
df = df.merge(within, on=["_dancer", "_season_raw"], how="left")

# exclude current season stats
df["ProCount"] = (df["cnt_all"] - df["cnt_in_s"]).fillna(0).astype(int)
sum_other = (df["sum_all"] - df["sum_in_s"]).fillna(0.0)
mean_other = np.where(df["ProCount"] > 0, sum_other / df["ProCount"], np.nan)

# shrinkage toward global mean
lam = LAMBDA_SHRINK
df["ProStrength"] = np.where(
    df["ProCount"] > 0,
    (df["ProCount"] / (df["ProCount"] + lam)) * mean_other + (lam / (df["ProCount"] + lam)) * global_mean_R,
    global_mean_R
)

# =========================
# 4) Build per-season y_judges in (0,1): average weekly judge-share over active weeks
# =========================
# compute weekly total judge points per contestant
for w in week_nums:
    cols_w = [c for c in judge_cols if c.startswith(f"week{w}_")]
    df[f"judge_total_w{w}"] = df[cols_w].sum(axis=1, skipna=True)

id_cols = ["celebrity_name", "_season_raw", "T"]
judge_long = df[id_cols + [f"judge_total_w{w}" for w in week_nums]].copy()
judge_long = judge_long.melt(
    id_vars=id_cols,
    var_name="week_col",
    value_name="judge_total"
)
judge_long["week"] = judge_long["week_col"].str.extract(r"w(\d+)$").astype(int)

# active weeks only: week <= T
judge_long = judge_long[judge_long["week"] <= judge_long["T"]].copy()

# weekly denominator: sum across active contestants in same season-week
judge_long["denom"] = judge_long.groupby(["_season_raw", "week"])["judge_total"].transform("sum")

# judge share in that week
judge_long["judge_share_w"] = np.where(judge_long["denom"] > 0, judge_long["judge_total"] / judge_long["denom"], np.nan)

# per-season mean judge share
y_j = judge_long.groupby(["_season_raw", "celebrity_name"])["judge_share_w"].mean().reset_index()
y_j = y_j.rename(columns={"judge_share_w": "y_judges"})

df = df.merge(y_j, on=["_season_raw", "celebrity_name"], how="left")

# =========================
# 5) Build per-season y_fans in (0,1): mean of per-week fan shares (or use per-season column if exists)
# =========================
fan_season_col = find_fan_season_col(df)
if fan_season_col is not None:
    df["y_fans"] = df[fan_season_col].astype(float)
else:
    fan_week_cols, fan_week_nums = find_fan_week_cols(df)
    if fan_week_cols is None:
        raise ValueError(
            "No fan vote estimate columns found.\n"
            "Please add your fan vote estimates to the CSV as either:\n"
            "  (A) one per-season column named like: fan_share_season / fan_share / vote_share_season, OR\n"
            "  (B) per-week columns named like: fan_share_w1, fan_share_w2, ... (or P_w1, P_w2, ...)\n"
            "Then re-run. You can also add your exact column names into FAN_SEASON_COL_CANDIDATES or FAN_WEEK_REGEXES."
        )

    fan_long = df[["celebrity_name", "_season_raw", "T"] + fan_week_cols].copy()
    fan_long = fan_long.melt(
        id_vars=["celebrity_name", "_season_raw", "T"],
        var_name="week_col",
        value_name="fan_share_w"
    )
    # extract week number
    wk = fan_long["week_col"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    fan_long["week"] = wk

    # active weeks only
    fan_long = fan_long[fan_long["week"] <= fan_long["T"]].copy()

    y_f = fan_long.groupby(["_season_raw", "celebrity_name"])["fan_share_w"].mean().reset_index()
    y_f = y_f.rename(columns={"fan_share_w": "y_fans"})
    df = df.merge(y_f, on=["_season_raw", "celebrity_name"], how="left")

# clip y into (0,1)
df["y_judges"] = clip01(df["y_judges"], EPS_CLIP)
df["y_fans"] = clip01(df["y_fans"], EPS_CLIP)

# =========================
# 6) Assemble X matrix
# =========================
# base covariates
X_cols = ["US", "ProStrength", "ProCount", "age_z", "age_z2"]

# industry dummies
X_cols += [c for c in df.columns if c.startswith("industry_grp_")]

# season dummies (created by get_dummies on "season")
X_cols += [c for c in df.columns if c.startswith("season_")]

model_df = df[["celebrity_name", "_season_raw", "y_judges", "y_fans"] + X_cols].dropna().copy()

# add intercept
model_df["Intercept"] = 1.0
X_cols = ["Intercept"] + X_cols

# =========================
# 7) Beta regression (preferred). If BetaModel not available, fallback to fractional logit (GLM Binomial).
# =========================
def fit_beta_or_fractional_logit(y, X, label):
    try:
        # statsmodels Beta regression
        from statsmodels.othermod.betareg import BetaModel
        mod = BetaModel(y, X)
        res = mod.fit(disp=False)
        kind = "beta"
        return res, kind
    except Exception as e_beta:
        # fallback: fractional logit via GLM Binomial with logit link
        import statsmodels.api as sm
        mod = sm.GLM(y, X, family=sm.families.Binomial())
        res = mod.fit()
        kind = "fractional_logit"
        print(f"[WARN] BetaModel unavailable for {label}. Falling back to fractional logit (GLM Binomial). Reason: {e_beta}")
        return res, kind


X_mat = model_df[X_cols].astype(float)
yJ = model_df["y_judges"].astype(float)
yF = model_df["y_fans"].astype(float)

resJ, kindJ = fit_beta_or_fractional_logit(yJ, X_mat, "judges")
resF, kindF = fit_beta_or_fractional_logit(yF, X_mat, "fans")

# =========================
# 8) Collect coefficients + compare
# =========================
def coef_table(res, name):
    out = pd.DataFrame({
        "term": res.params.index if hasattr(res.params, "index") else X_cols,
        f"coef_{name}": np.asarray(res.params),
        f"se_{name}": np.asarray(res.bse),
        f"p_{name}": np.asarray(res.pvalues),
    })
    return out

tabJ = coef_table(resJ, "judges")
tabF = coef_table(resF, "fans")
cmp = tabJ.merge(tabF, on="term", how="inner")
cmp["coef_diff_fans_minus_judges"] = cmp["coef_fans"] - cmp["coef_judges"]
cmp["abs_diff"] = cmp["coef_diff_fans_minus_judges"].abs()

# (optional) rough z-test for difference assuming independence (conservative / heuristic)
cmp["se_diff_approx"] = np.sqrt(cmp["se_fans"]**2 + cmp["se_judges"]**2)
cmp["z_diff_approx"] = cmp["coef_diff_fans_minus_judges"] / cmp["se_diff_approx"].replace(0, np.nan)

cmp = cmp.sort_values("abs_diff", ascending=False)

# save tables
model_df.to_csv(OUT_DIR / "model_data_per_season.csv", index=False)
cmp.to_csv(OUT_DIR / "coef_compare.csv", index=False)

# =========================
# 9) Plots
# =========================
# Plot A: coefficient scatter (fans vs judges)
plt.figure(figsize=(7, 6), dpi=180)
x = cmp.loc[cmp["term"] != "Intercept", "coef_judges"].values
y = cmp.loc[cmp["term"] != "Intercept", "coef_fans"].values
plt.scatter(x, y)
mn = min(x.min(), y.min())
mx = max(x.max(), y.max())
plt.plot([mn, mx], [mn, mx], linestyle="--")  # y=x reference
plt.xlabel("Coefficient in Judges Model")
plt.ylabel("Coefficient in Fans Model")
plt.title(f"Coefficient Comparison (per-season, {kindJ} vs {kindF})")
plt.tight_layout()
plt.savefig(OUT_DIR / "coef_scatter_fans_vs_judges.png")
plt.close()

# Plot B: top-k absolute differences bar chart
topk = 12
top = cmp[(cmp["term"] != "Intercept")].head(topk).iloc[::-1]  # reverse for nicer bar order
plt.figure(figsize=(9, 6), dpi=180)
plt.barh(top["term"], top["coef_diff_fans_minus_judges"])
plt.axvline(0, linewidth=1)
plt.xlabel("Coefficient Difference (Fans - Judges)")
plt.title(f"Top {topk} Features with Largest Effect Differences")
plt.tight_layout()
plt.savefig(OUT_DIR / "top_diff_bar.png")
plt.close()

# =========================
# 10) Minimal textual summary for paper writing
# =========================
summary = {
    "n_used": int(len(model_df)),
    "model_judges": kindJ,
    "model_fans": kindF,
    "top_terms_by_abs_diff": top["term"].tolist(),
}
pd.Series(summary).to_json(OUT_DIR / "run_summary.json", indent=2)

print("Done.")
print(f"Saved outputs to: {OUT_DIR.resolve()}")
print(f"Judges model type: {kindJ} | Fans model type: {kindF}")
