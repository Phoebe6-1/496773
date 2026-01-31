import pandas as pd
import numpy as np
import re

# 1. 加载数据
df = pd.read_csv("2026_MCM_Problem_C_Data.csv")

# 2. 准备基础变量
score_cols = [c for c in df.columns if "judge" in c and "score" in c]
max_week = 11

# ==== 从 results 解析淘汰周 ====
exit_info = df[['season', 'celebrity_name', 'results']].copy()

def parse_elim_week(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r'Eliminated\s*Week\s*(\d+)', str(x), flags=re.I)
    return int(m.group(1)) if m else np.nan

exit_info['elim_week'] = exit_info['results'].apply(parse_elim_week)

# 3. 宽表 -> 长表（每周一行）
melted_rows = []
for w in range(1, max_week + 1):
    judge_col_names = [c for c in score_cols if f"week{w}_" in c]

    temp = df[['season', 'celebrity_name']].copy()
    temp['week'] = w
    temp['avg_score'] = df[judge_col_names].mean(axis=1)  # 跳过 NaN
    melted_rows.append(temp)

long_df = pd.concat(melted_rows, ignore_index=True)

# 4. 丢弃赛季未举行到某周的行
exists_mask = long_df.groupby(['season', 'week'])['avg_score'].transform(lambda x: x.notna().any())
long_df = long_df[exists_mask].dropna(subset=['avg_score']).copy()

# 5. 用 results 标记淘汰当周
long_df = long_df.merge(
    exit_info[['season', 'celebrity_name', 'elim_week']],
    on=['season', 'celebrity_name'],
    how='left'
)

long_df['eliminated_this_week'] = (long_df['week'] == long_df['elim_week']).fillna(False).astype(int)

# 6. 删除淘汰周之后的周（保留到淘汰当周为止）
long_df = long_df.sort_values(['season', 'celebrity_name', 'week'])

long_df['prev_eliminated'] = (
    long_df.groupby(['season', 'celebrity_name'])['eliminated_this_week']
    .shift(1)
    .fillna(0)
    .astype(int)
)
long_df['cum_elim'] = long_df.groupby(['season', 'celebrity_name'])['prev_eliminated'].cumsum()
long_df = long_df[long_df['cum_elim'] == 0].copy()

# 7. 计算 judge_percent 和 judge_rank（不用 apply，避免 season/week 进 index）
g = long_df.groupby(['season', 'week'], sort=False)

week_sum = g['avg_score'].transform('sum')
long_df['judge_percent'] = np.where(week_sum > 0, long_df['avg_score'] / week_sum, 0.0)
long_df['judge_rank'] = g['avg_score'].rank(ascending=False, method='min').astype(int)

# ====== NEW: 把 judge_percent==0 的行视为“淘汰发生在上一周”，并删除该行 ======

# 1) 找到 judge_percent 为 0 的行（浮点更稳用一个很小阈值）
eps = 1e-12
zero_mask = long_df['judge_percent'].abs() <= eps

# 2) 把“下一周为0”的标记回传给上一周：上一周 eliminated_this_week = 1
#    对于每个 (season, celebrity) 序列，shift(-1) 会把下一周的信息移到上一周那一行
prev_should_elim = zero_mask.groupby([long_df['season'], long_df['celebrity_name']]).shift(-1).fillna(False)

long_df.loc[prev_should_elim, 'eliminated_this_week'] = 1

# 3) 删除 judge_percent==0 的那一行
long_df = long_df[~zero_mask].copy()

# 4) 重新按 eliminated_this_week 截断：保留到淘汰当周为止，淘汰后续全部删除
long_df = long_df.sort_values(['season', 'celebrity_name', 'week'])

long_df['prev_elim2'] = (
    long_df.groupby(['season', 'celebrity_name'])['eliminated_this_week']
    .shift(1).fillna(0).astype(int)
)
long_df['cum_elim2'] = long_df.groupby(['season', 'celebrity_name'])['prev_elim2'].cumsum()
long_df = long_df[long_df['cum_elim2'] == 0].copy()

# 清理临时列
long_df.drop(columns=['prev_elim2', 'cum_elim2'], inplace=True, errors='ignore')


# 8. 导出
final_columns = ['season', 'week', 'celebrity_name', 'judge_percent', 'judge_rank', 'eliminated_this_week']
processed_data = long_df[final_columns].sort_values(['season', 'week', 'judge_rank'])
processed_data['judge_percent'] = round(processed_data['judge_percent'],3)

processed_data.to_csv('processed_data.csv', index=False)
print(processed_data.head(10))
