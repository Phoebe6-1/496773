import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()


def add_official_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark 'official method' by season using (method, use_bottom2):
      - season <= 2: official = rank + no bottom2
      - 3 <= season <= 27: official = percent + no bottom2
      - season >= 28: official = percent + bottom2
    """
    s = df["season"].astype(int)
    is_official = (
        ((s <= 2) & (df["method"] == "rank") & (df["use_bottom2"] == False)) |
        (((s >= 3) & (s <= 27)) & (df["method"] == "percent") & (df["use_bottom2"] == False)) |
        ((s >= 28) & (df["method"] == "percent") & (df["use_bottom2"] == True))
    )
    df = df.copy()
    df["is_official"] = is_official

    # official tag per contestant-season (for summary display)
    official_tag = (
        df[df["is_official"]]
        .groupby(["season", "celebrity_name"])["tag"]
        .agg(lambda x: x.iloc[0] if len(x) else np.nan)
        .reset_index()
        .rename(columns={"tag": "official_tag"})
    )
    df = df.merge(official_tag, on=["season", "celebrity_name"], how="left")
    return df


def load_actual_ranks(actual_path: str | None) -> pd.DataFrame:
    """
    If actual_path is None, use the built-in 6-contestant mapping from your Table 2.
    Otherwise, read a CSV with columns: season, celebrity_name, R_origin
    """
    if actual_path is None:
        actual = pd.DataFrame([
            {"season": 2,  "celebrity_name": "Jerry Rice",       "R_origin": 2},
            {"season": 4,  "celebrity_name": "Billy Ray Cyrus",  "R_origin": 5},
            {"season": 11, "celebrity_name": "Bristol Palin",    "R_origin": 3},
            {"season": 17, "celebrity_name": "Bill Engvall",     "R_origin": 4},
            {"season": 27, "celebrity_name": "Bobby Bones",      "R_origin": 1},
            {"season": 29, "celebrity_name": "Nelly",            "R_origin": 3},
        ])
    else:
        actual = pd.read_csv(actual_path)
        required = {"season", "celebrity_name", "R_origin"}
        missing = required - set(actual.columns)
        if missing:
            raise ValueError(f"--actual 文件缺少列：{missing}，需要至少包含 {required}")

    actual = actual.copy()
    actual["season"] = actual["season"].astype(int)
    actual["R_origin"] = actual["R_origin"].astype(int)
    actual["name_key"] = actual["celebrity_name"].map(norm_name)
    return actual[["season", "name_key", "R_origin"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="output_sim/controversy_summary.csv", help="Path to controversy_summary.csv")
    parser.add_argument("--out", default="metrics_out", help="Output directory")
    parser.add_argument("--actual", default=None, help="Optional: CSV with season, celebrity_name, R_origin")
    args = parser.parse_args()

    in_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # basic checks
    needed_cols = {"season", "celebrity_name", "placement_sim", "method", "use_bottom2", "tag"}
    miss = needed_cols - set(df.columns)
    if miss:
        raise ValueError(f"输入CSV缺少列：{miss}；你的CSV应包含至少 {needed_cols}")

    df = df.copy()
    df["season"] = df["season"].astype(int)
    df["name_key"] = df["celebrity_name"].map(norm_name)

    # merge actual ranks
    actual = load_actual_ranks(args.actual)
    df = df.merge(actual, on=["season", "name_key"], how="left")

    if df["R_origin"].isna().any():
        missing = df.loc[df["R_origin"].isna(), ["season", "celebrity_name"]].drop_duplicates()
        raise ValueError("以下选手没有匹配到真实名次 R_origin（检查名字/season是否一致）：\n"
                         + missing.to_string(index=False))

    df["R_origin"] = df["R_origin"].astype(int)

    # mark official method rows
    df = add_official_flag(df)

    # ΔR: exclude official rows by setting NaN
    df["deltaR"] = (df["R_origin"] - df["placement_sim"]).abs().astype(float)
    df.loc[df["is_official"], "deltaR"] = np.nan  # 关键：官方方法不计入 ΔR

    # MS across ALL 4 methods (as in your formula)
    ms_all = (
        df.groupby(["season", "celebrity_name"], as_index=False)["placement_sim"]
          .std(ddof=1)
          .rename(columns={"placement_sim": "MS_all"})
    )
    df = df.merge(ms_all, on=["season", "celebrity_name"], how="left")

    # (Optional) MS across alternatives only (exclude official)
    ms_alt = (
        df[~df["is_official"]]
        .groupby(["season", "celebrity_name"], as_index=False)["placement_sim"]
        .std(ddof=1)
        .rename(columns={"placement_sim": "MS_alt"})
    )
    df = df.merge(ms_alt, on=["season", "celebrity_name"], how="left")

    # detail output
    detail_cols = [
        "season", "celebrity_name", "tag", "method", "use_bottom2",
        "official_tag", "is_official",
        "R_origin", "placement_sim", "eliminated_week_sim",
        "deltaR", "MS_all", "MS_alt"
    ]
    detail = df[detail_cols].sort_values(["season", "celebrity_name", "tag"])
    detail_path = out_dir / "detail_metrics.csv"
    detail.to_csv(detail_path, index=False)

    # contestant summary
    contestant_summary = (
        df.groupby(["season", "celebrity_name"], as_index=False)
          .agg(
              R_origin=("R_origin", "first"),
              official_tag=("official_tag", "first"),
              MS_all=("MS_all", "first"),
              MS_alt=("MS_alt", "first"),
              mean_deltaR_alt=("deltaR", "mean"),     # 这里 deltaR 已经排除了官方行（NaN不计）
              max_deltaR_alt=("deltaR", "max"),
          )
          .sort_values(["season"])
    )
    contestant_path = out_dir / "contestant_summary.csv"
    contestant_summary.to_csv(contestant_path, index=False)

    # method summary (exclude official rows automatically because deltaR is NaN there)
    method_summary = (
        df.groupby("tag", as_index=False)
          .agg(
              n_total=("tag", "size"),
              n_alt=("deltaR", lambda x: x.notna().sum()),
              mean_deltaR_alt=("deltaR", "mean"),
              median_deltaR_alt=("deltaR", "median"),
          )
          .sort_values("mean_deltaR_alt", ascending=False)
    )
    method_path = out_dir / "method_summary.csv"
    method_summary.to_csv(method_path, index=False)

    print(f"[OK] Saved:\n- {detail_path}\n- {contestant_path}\n- {method_path}")
    print("\nPreview (contestant_summary):")
    print(contestant_summary.to_string(index=False))


if __name__ == "__main__":
    main()
