import pandas as pd
import numpy as np

# 改成你的文件路径
INPUT_CSV = r"model1_fans_vote_estimate\output\prediction_by_week.csv"

def main():
    df = pd.read_csv(INPUT_CSV)

    # 基础清洗
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    # 确保字符串列存在（有些空单元会读成 NaN）
    if "true_eliminated" not in df.columns or "pred_eliminated" not in df.columns:
        raise ValueError("CSV must contain columns: true_eliminated, pred_eliminated")
    df["true_eliminated"] = df["true_eliminated"].fillna("").astype(str)
    df["pred_eliminated"] = df["pred_eliminated"].fillna("").astype(str)

    # 若 correct 列不存在，则现场计算（pred 是否在 true 集合里；空淘汰周记 NaN）
    if "correct" not in df.columns:
        def compute_correct(row):
            true_list = [x for x in row["true_eliminated"].split(";") if x.strip()]
            if len(true_list) == 0:
                return np.nan
            pred = row["pred_eliminated"].strip()
            return int(pred in true_list)
        df["correct"] = df.apply(compute_correct, axis=1)
    else:
        # correct 有时读成 float（NaN），统一处理
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce")

    # 识别每个 season 的最后一周
    last_week = df.groupby("season")["week"].max().to_dict()
    df["is_last_week"] = df.apply(lambda r: int(r["week"] == last_week[r["season"]]), axis=1)

    # 标记“无淘汰周”（true_eliminated 为空）
    df["no_elimination"] = df["true_eliminated"].str.strip().eq("")

    # 按你的规则：每季最后一周无淘汰 => 视为正确
    # 其它无淘汰周：不计入accuracy（NaN，不算对错）
    df["correct_adj"] = df["correct"]

    mask_last_noelim = (df["is_last_week"] == 1) & (df["no_elimination"])
    df.loc[mask_last_noelim, "correct_adj"] = 1.0

    # 其它 no_elimination 周不计入（保持 NaN）
    # 这里顺便把“非空淘汰周但correct为空”的也排掉（数据异常）
    valid = df["correct_adj"].notna()
    overall_acc = df.loc[valid, "correct_adj"].mean()

    print(f"Overall accuracy (with last-week no-elim treated as correct): {overall_acc:.4f}")
    print(f"Counted weeks: {valid.sum()} / total rows: {len(df)}")

    # ====== 每季 accuracy ======
    season_acc = (
        df.loc[valid]
        .groupby("season")["correct_adj"]
        .mean()
        .reset_index()
        .rename(columns={"correct_adj": "season_accuracy"})
    )
    print("\nSeason-level accuracy (head):")
    print(season_acc.head(10).to_string(index=False))

    # ====== 可选：按 rule_type 汇总 ======
    if "rule_type" in df.columns:
        rule_acc = (
            df.loc[valid]
            .groupby("rule_type")["correct_adj"]
            .mean()
            .reset_index()
            .rename(columns={"correct_adj": "rule_accuracy"})
        )
        print("\nRule-type accuracy:")
        print(rule_acc.to_string(index=False))

    # 如需保存汇总结果：
    season_acc.to_csv(r"model1_fans_vote_estimate\output\accuracy_by_season.csv", index=False)
    if "rule_type" in df.columns:
        rule_acc.to_csv(r"model1_fans_vote_estimate\output\accuracy_by_rule_type.csv", index=False)

if __name__ == "__main__":
    main()
