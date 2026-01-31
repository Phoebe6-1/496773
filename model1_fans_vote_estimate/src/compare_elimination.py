import pandas as pd
import numpy as np
from pathlib import Path

# ====== 配置区 ======
INPUT_CSV = "model1_fans_vote_estimate\output\estimated_votes.csv"   # 改成你的文件路径
OUT_DIR = Path("model1_fans_vote_estimate\output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== 工具函数 ======
def rank_desc(s: pd.Series) -> pd.Series:
    """值越大越好，rank=1最好"""
    return s.rank(ascending=False, method="min").astype(int)

def rank_asc(s: pd.Series) -> pd.Series:
    """值越小越好，rank=1最好"""
    return s.rank(ascending=True, method="min").astype(int)

def add_week_final_ranking(df_w: pd.DataFrame) -> pd.DataFrame:
    """
    对单个 (season, week) 增加：
    - final_score_or_metric：按规则得到的综合指标（percent为分数和；rank为名次和）
    - final_rank：当周最终排名（1最好）
    - pred_eliminated_flag：是否被预测淘汰
    - （S28+额外）in_pred_bottom2：是否在预测bottom2
    """
    df_w = df_w.copy()
    rule = df_w["rule_type"].iloc[0]

    # 默认列
    df_w["in_pred_bottom2"] = 0

    if rule == "S3-27_percent":
        # 规则：两个 percentage 直接求和
        # combined 越大越好 → 淘汰者为 combined 最小
        df_w["final_score_or_metric"] = df_w["judge_percent"] + df_w["estimated_fan_vote_share"]
        df_w["final_rank"] = rank_desc(df_w["final_score_or_metric"])
        pred_elim = df_w.loc[df_w["final_score_or_metric"].idxmin(), "celebrity_name"]

    elif rule == "S1-2_rank":
        # 规则：名次和 = judge_rank + fan_rank，其中 fan_rank 由投票占比排序得到（大→好→rank小）
        df_w["fan_rank_from_P"] = rank_desc(df_w["estimated_fan_vote_share"])
        df_w["final_score_or_metric"] = df_w["judge_rank"].astype(float) + df_w["fan_rank_from_P"].astype(float)
        df_w["final_rank"] = rank_asc(df_w["final_score_or_metric"])
        pred_elim = df_w.loc[df_w["final_score_or_metric"].idxmax(), "celebrity_name"]

    elif rule == "S28+_bottom2":
        # Step1: 先算名次和（综合差）
        df_w["fan_rank_from_P"] = rank_desc(df_w["estimated_fan_vote_share"])
        df_w["final_score_or_metric"] = df_w["judge_rank"].astype(float) + df_w["fan_rank_from_P"].astype(float)
        df_w["final_rank"] = rank_asc(df_w["final_score_or_metric"])

        # Step2: bottom2 = 名次和最大的两人
        bottom2 = df_w.nlargest(2, "final_score_or_metric")
        df_w["in_pred_bottom2"] = df_w["celebrity_name"].isin(bottom2["celebrity_name"]).astype(int)

        # Step3: judges save：淘汰 bottom2 中 judge_rank 更差者（值更大）
        pred_elim = bottom2.loc[bottom2["judge_rank"].idxmax(), "celebrity_name"]

    else:
        raise ValueError(f"Unknown rule_type: {rule}")

    df_w["pred_eliminated_flag"] = (df_w["celebrity_name"] == pred_elim).astype(int)
    return df_w

def main():
    df = pd.read_csv(INPUT_CSV)

    # 类型修正
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["eliminated_this_week"] = df["eliminated_this_week"].astype(int)

    # 必要列检查
    need = {
        "season", "week", "celebrity_name",
        "judge_percent", "judge_rank",
        "eliminated_this_week",
        "estimated_fan_vote_share",
        "rule_type"
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input csv: {missing}")

    # 逐周加最终排名 + 生成 prediction_by_week
    aug_list = []
    week_rows = []

    for (season, week), df_w in df.groupby(["season", "week"], sort=True):
        df_w_aug = add_week_final_ranking(df_w)
        aug_list.append(df_w_aug)

        true_elims = df_w_aug.loc[df_w_aug["eliminated_this_week"] == 1, "celebrity_name"].tolist()
        pred_elim = df_w_aug.loc[df_w_aug["pred_eliminated_flag"] == 1, "celebrity_name"].iloc[0] \
                   if (df_w_aug["pred_eliminated_flag"].sum() > 0) else ""

        # correct：预测淘汰者是否属于真实淘汰者集合；无淘汰周记为 NaN
        if len(true_elims) == 0:
            correct = np.nan
        else:
            correct = int(pred_elim in true_elims)

        week_rows.append({
            "season": int(season),
            "week": int(week),
            "rule_type": df_w_aug["rule_type"].iloc[0],
            "true_eliminated": ";".join(true_elims) if true_elims else "",
            "pred_eliminated": pred_elim,
            "correct": correct,
            "n_contestants": int(len(df_w_aug)),
        })

    df_aug = pd.concat(aug_list, ignore_index=True)
    df_pred_week = pd.DataFrame(week_rows)

    # 输出
    df_pred_week.to_csv(OUT_DIR / "prediction_by_week.csv", index=False)
    df_aug.to_csv(OUT_DIR / "estimated_votes_with_final_rank.csv", index=False)

    print("Saved:")
    print(" - output/prediction_by_week.csv")
    print(" - output/estimated_votes_with_final_rank.csv")

if __name__ == "__main__":
    main()
