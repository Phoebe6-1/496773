# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd

# # 1. 加载真实数据
# try:
#     df = pd.read_csv('accuracy_by_season.csv')
#     x_data = df['season']
#     y_data = df['season_accuracy']
# except FileNotFoundError:
#     print("错误：未找到 accuracy_by_season.csv 文件，请检查路径。")
#     exit()

# # 计算统计量
# mean_acc = y_data.mean()
# std_acc = y_data.std()
# max_val, min_val = y_data.max(), y_data.min()
# max_season = x_data[y_data.idxmax()]
# min_season = x_data[y_data.idxmin()]

# # 2. 风格配置（学术风）
# sns.set_style("whitegrid")
# sns.set_context("paper", font_scale=1.5)
# plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# fig, ax = plt.subplots(figsize=(16, 8), dpi=300)  # 更大高清画布

# # 3. 背景填充 + 阴影带（波动范围）
# ax.fill_between(x_data, y_data, 0.65, color='#3b82f6', alpha=0.08, zorder=1)
# ax.fill_between(x_data, mean_acc - std_acc, mean_acc + std_acc, 
#                 color='#1e40af', alpha=0.15, label='±1 Std Dev Range')

# # 4. 主折线（1-27季深蓝，28-34季橙色高亮）
# ax.plot(x_data[:27], y_data[:27], color='#1d4ed8', linewidth=3, 
#         marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, zorder=4)
# ax.plot(x_data[27:], y_data[27:], color='#f97316', linewidth=3.5, 
#         marker='o', markersize=10, markerfacecolor='white', markeredgewidth=2.5,
#         label='Seasons with bottom-two (28-34)', zorder=5)

# # 5. 平均线（灰色虚线）
# ax.axhline(mean_acc, color='#64748b', linestyle='--', linewidth=2, alpha=0.8)

# # 6. 极值与争议赛季智能标注
# # 峰值（绿色）
# ax.annotate(f'Peak: {max_val:.1%}', xy=(max_season, max_val), xytext=(0, 18),
#             textcoords='offset points', ha='center', fontsize=12, fontweight='bold', color='#15803d',
#             arrowprops=dict(arrowstyle='->', color='#22c55e', lw=2))

# # 谷值（红色）
# ax.annotate(f'Lowest: {min_val:.1%}', xy=(min_season, min_val), xytext=(0, -28),
#             textcoords='offset points', ha='center', fontsize=12, fontweight='bold', color='#b91c1c',
#             arrowprops=dict(arrowstyle='->', color='#ef4444', lw=2))

# # 争议赛季标注
# ax.annotate('Season 2\n(Jerry Rice Controversy)', xy=(2, y_data[1]), xytext=(5, -50),
#             textcoords='offset points', fontsize=11, color='#7c2d12',
#             arrowprops=dict(arrowstyle='->', color='#7c2d12'))
# ax.annotate('Season 27\n(Bobby Bones Controversy)', xy=(27, y_data[26]), xytext=(5, 75),
#             textcoords='offset points', fontsize=11, color='#7c2d12',
#             arrowprops=dict(arrowstyle='->', color='#7c2d12'))


# # 平均值文字
# ax.text(x_data.max() + 0.8, mean_acc, f'Mean: {mean_acc:.1%}', va='center',
#         fontsize=12, fontweight='bold', color='#64748b', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# # 7. 坐标轴精修
# ax.set_ylim(0.65, 1.05)
# ax.set_yticks(np.arange(0.65, 1.06, 0.05))
# ax.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0.65, 1.06, 0.05)])

# ax.set_xticks(x_data[::2])  # 每隔一季显示
# ax.set_xlabel('Season Index', fontsize=14, fontweight='bold', labelpad=15)
# ax.set_ylabel('Model Consistency Accuracy', fontsize=14, fontweight='bold', labelpad=15)

# # 8. 网格与边框
# ax.grid(axis='y', linestyle=':', alpha=0.5, color='#94a3b8')
# for spine in ['top', 'right']:
#     ax.spines[spine].set_visible(False)

# # 9. 标题（主标题+副标题）
# plt.suptitle('Model Consistency Analysis Across DWTS Seasons (1-34)', 
#              fontsize=20, fontweight='bold', y=0.98)
# plt.title('Highlight: Rank-Based Method Return from Season 28', 
#           fontsize=14, color='#64748b', pad=20)

# # 10. 图例
# ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=12)

# plt.tight_layout()

# # 保存矢量PDF（报告首选） + PNG（预览）
# plt.savefig('advanced_consistency_analysis.pdf', bbox_inches='tight', dpi=300)
# plt.savefig('advanced_consistency_analysis.png', bbox_inches='tight', dpi=300)

# # plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 1. 加载真实数据
try:
    df = pd.read_csv('accuracy_by_season.csv')
    x_data = df['season']
    y_data = df['season_accuracy']
except FileNotFoundError:
    print("错误：未找到 accuracy_by_season.csv 文件，请检查路径。")
    exit()

# 计算统计量
mean_acc = y_data.mean()
std_acc = y_data.std()
max_val, min_val = y_data.max(), y_data.min()
max_season = x_data[y_data.idxmax()]
min_season = x_data[y_data.idxmin()]

# 2. 风格配置（学术风 - 调大字号基准）
sns.set_style("whitegrid")
# 将 font_scale 从 1.5 调至 1.8，整体放大
sns.set_context("paper", font_scale=1.8) 
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

# 3. 背景填充 + 阴影带
ax.fill_between(x_data, y_data, 0.65, color='#3b82f6', alpha=0.08, zorder=1)
ax.fill_between(x_data, mean_acc - std_acc, mean_acc + std_acc, 
                color='#1e40af', alpha=0.15, label='±1 Std Dev Range')

# 4. 主折线
ax.plot(x_data[:27], y_data[:27], color='#1d4ed8', linewidth=3, 
        marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, zorder=4)
ax.plot(x_data[27:], y_data[27:], color='#f97316', linewidth=3.5, 
        marker='o', markersize=10, markerfacecolor='white', markeredgewidth=2.5,
        label='Seasons with bottom-two (28-34)', zorder=5)

# 5. 平均线
ax.axhline(mean_acc, color='#64748b', linestyle='--', linewidth=2, alpha=0.8)

# 6. 极值与争议赛季智能标注（调大字号）
# 峰值
ax.annotate(f'Peak: {max_val:.1%}', xy=(max_season, max_val), xytext=(0, 20),
            textcoords='offset points', ha='center', fontsize=16, fontweight='bold', color='#15803d',
            arrowprops=dict(arrowstyle='->', color='#15803d', lw=2))

# 谷值
ax.annotate(f'Lowest: {min_val:.1%}', xy=(min_season, min_val), xytext=(0, -30),
            textcoords='offset points', ha='center', fontsize=16, fontweight='bold', color='#b91c1c',
            arrowprops=dict(arrowstyle='->', color='#b91c1c', lw=2))

# 争议赛季
ax.annotate('Season 2\n(Jerry Rice Controversy)', xy=(2, y_data[1]), xytext=(5, -55),
            textcoords='offset points', fontsize=13, color='#7c2d12',
            arrowprops=dict(arrowstyle='->', color='#7c2d12'))
ax.annotate('Season 27\n(Bobby Bones Controversy)', xy=(27, y_data[26]), xytext=(5, 80),
            textcoords='offset points', fontsize=13, color='#7c2d12',
            arrowprops=dict(arrowstyle='->', color='#7c2d12'))

# 平均值文字
ax.text(x_data.max() + 0.8, mean_acc, f'Mean: {mean_acc:.1%}', va='center',
        fontsize=15, fontweight='bold', color='#64748b', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#64748b', alpha=0.8))

# 7. 坐标轴精修（调大标签字号）
ax.set_ylim(0.65, 1.05)
ax.set_yticks(np.arange(0.65, 1.06, 0.05))
ax.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0.65, 1.06, 0.05)])

ax.set_xticks(x_data[::2])
ax.set_xlabel('Season Index', fontsize=18, fontweight='bold', labelpad=15)
ax.set_ylabel('Model Consistency Accuracy', fontsize=18, fontweight='bold', labelpad=15)

# 8. 网格与边框
ax.grid(axis='y', linestyle=':', alpha=0.5, color='#94a3b8')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# 9. 标题（调大字号）
plt.suptitle('Model Consistency Analysis Across DWTS Seasons (1-34)', 
             fontsize=28, fontweight='bold', y=0.98)
plt.title('Highlight: Rank-Based Method Return from Season 28', 
          fontsize=20, color='#64748b', pad=20)

# 10. 图例（调大字号）
ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=16)

plt.tight_layout()

# 保存
plt.savefig('advanced_consistency_analysis_v2.png', bbox_inches='tight', dpi=300)
