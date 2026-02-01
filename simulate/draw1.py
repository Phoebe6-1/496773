import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 加载数据
df = pd.read_csv("weekly_bias_index.csv")

# 2. 数据清洗：剔除异常行（你原来的逻辑保持不动）
bi_cols = ['BI_rank', 'BI_percent']
mask = df[bi_cols].apply(
    lambda x: x.between(0.2, 1.0, inclusive='both') | x.between(-1.0, -0.6, inclusive='both')
).any(axis=1)
df_clean = df[~mask].copy()

# 排序 + 标签
df_clean = df_clean.sort_values(['season', 'week'])
df_clean['label'] = df_clean.apply(lambda row: f"S{int(row['season'])}W{int(row['week'])}", axis=1)

# ★新增：给 fill_between 用的数值 x 轴
df_clean['xpos'] = np.arange(len(df_clean))

# 3. 马卡龙风格主题（柔和背景+柔和网格）
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": "white",   
    "axes.facecolor":   "white",   
    "axes.edgecolor":   "#E6D7E3",
    "grid.color":       "#EADFEB",
    "grid.linewidth":   1.0,
    "axes.titleweight": "bold",
    "font.size":        11,
})

fig, ax = plt.subplots(figsize=(15, 8), dpi=200)

# 4. 画“图二那种”阴影带：以 BI_rank 的 mean ± std 为例
mu = df_clean['BI_rank'].mean()
sd = df_clean['BI_rank'].std(ddof=1)

# ★淡粉色阴影（你想要的效果）
ax.fill_between(
    df_clean['xpos'].values,
    (mu - sd),
    (mu + sd),
    color="#FADADD",     # 淡粉色
    alpha=0.35,
    zorder=0,
    label="±1 Std (Rank BI)"   # 不想显示图例就删掉这一行 + 后面的 legend 里不放它
)

# （可选）像图二一样加一条 mean 虚线（更像“蓝色阴影”那套视觉）
ax.axhline(mu, color="#B9A7B5", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)

# 5. 两条折线（马卡龙配色）
rank_color = "#86BBD8"     # 柔和浅蓝
percent_color = "#F4A7B9"  # 柔和粉（也可以换成淡橙：#F9C784）

# 用 seaborn 也行，但这里直接用 matplotlib 更好控层级与效果
ax.plot(df_clean['xpos'], df_clean['BI_rank'],
        marker='o', markersize=6, linewidth=2.2,
        color=rank_color, label='BI (Rank-based)', zorder=3)

ax.plot(df_clean['xpos'], df_clean['BI_percent'],
        marker='s', markersize=6, linewidth=2.2,
        color=percent_color, label='BI (Percent-based)', zorder=3)

# 6. Y=0 基准线（更柔和一点）
ax.axhline(0, color="#9E9E9E", linewidth=1.0, linestyle='--', alpha=0.6, zorder=2)

# 7. 标题与坐标轴
ax.set_title('Comparison of Weekly Judge Bias Indices', fontsize=20, pad=18)
ax.set_xlabel('Season & Week', fontsize=16, labelpad=10)
ax.set_ylabel('Bias Index Value', fontsize=16, labelpad=10)

# 8. 把数值 x 显示回 label
ax.set_xticks(df_clean['xpos'])
ax.set_xticklabels(df_clean['label'], rotation=45, ha='right', fontsize=9)

fig.subplots_adjust(left=0.10)   # 0.10~0.14 之间自己微调

# 9. 让整体更“软”：弱化边框
for spine in ax.spines.values():
    spine.set_alpha(0.35)

# 10. 图例（白底半透明）
leg = ax.legend(loc='upper left', frameon=True, fontsize=12)
leg.get_frame().set_facecolor("#FFFFFF")
leg.get_frame().set_alpha(0.85)
leg.get_frame().set_edgecolor("#E6D7E3")

plt.tight_layout()

# 保存图片和清洗后的数据
plt.savefig('beautiful_bi_plot_pastel_shadow.png', bbox_inches='tight')
df_clean.drop(columns=['label']).to_csv('filtered_bi_data.csv', index=False)

plt.show()
