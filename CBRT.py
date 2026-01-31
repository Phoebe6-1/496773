import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# 数据
categories = ['S1-2 rank', 'S28+ bottom2', 'S3-27 percent']
values = [0.923, 0.810, 0.945]
n = len(categories)

# 调整坐标
xpos = np.arange(n)
ypos = np.zeros(n)
zpos = np.zeros(n)
dx = 0.43
dy = 0.1
dz = values

# 颜色设置：适配新版本的色板调用方式（弃用get_cmap）
norm = mcolors.Normalize(0.75, 1)
cmap = plt.colormaps['Pastel1']  # 新版本推荐写法
colors = cmap(norm(dz))

# 绘图
fig = plt.figure(figsize=(24, 6))
ax = fig.add_subplot(111, projection='3d')

# 核心修复：移除无效的shade_factor，用shade+alpha控制阴影（适配新版本）
bars = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average', 
                shade=True,  # 开启阴影（新版本仅支持True/False，无强度参数）
                alpha=0.9,lightsource=None)  # 用轻微透明弱化阴影，替代shade_factor

# 核心新增：在每个柱子顶部标注具体数值（颜色和柱子一致但稍深）
def darken_color(color, factor=0.7):
    """将颜色加深（factor<1加深，越小越深）"""
    # 分离RGBA通道，RGB通道乘以factor（alpha保持1）
    r, g, b, a = color
    return (r*factor, g*factor, b*factor, 1.0)

for i in range(n):
    # 获取当前柱子的颜色并加深
    bar_color = colors[i]
    text_color = darken_color(bar_color, factor=0.27)  # 0.7是加深程度，可调整
    
    # 标注位置：x在柱子中心，y固定0，z在柱子高度+0.01（避免重叠）
    ax.text(xpos[i] + dx/2,  # x坐标（柱子中心）
            ypos[i],         # y坐标（和柱子一致）
            dz[i] + 0.01,    # z坐标（柱子顶部+0.01，留出空隙）
            f'{dz[i]:.3f}',  # 数值格式化为3位小数
            fontsize=12,     # 字体大小
            color=text_color,# 加深后的柱子颜色
            ha='center',     # 水平居中
            va='bottom',     # 垂直靠下（贴在柱子顶部）
            fontweight='medium')  # 字体稍粗，增强可读性

# 额外优化：调整3D渲染的光照角度，让阴影更浅（新版本控制阴影的核心方式）
ax._dist = 330  # 调整相机距离，弱化阴影对比度
ax.azim = -70  # 保持原有方位角
ax.elev = 15   # 保持原有仰角

# 弱化网格/刻度（保留浅色系风格）
ax.xaxis._axinfo["grid"]['color'] = (0.95, 0.95, 0.95, 0.3)
ax.yaxis._axinfo["grid"]['color'] = (0.95, 0.95, 0.95, 0.3)
ax.zaxis._axinfo["grid"]['color'] = (0.95, 0.95, 0.95, 0.5)
ax.xaxis._axinfo["tick"]['color'] = (0.8, 0.8, 0.8, 0.9)
ax.zaxis._axinfo["tick"]['color'] = (0.8, 0.8, 0.8, 0.9)

# 原有美化设置
ax.set_xticks(xpos + dx/2)
ax.set_xticklabels(categories, rotation=15, ha='right')
ax.set_yticks([])
ax.set_zlabel('Rule Accuracy', fontsize=12)
ax.set_title('3D Bar Chart: Model Accuracy by Voting Rules', 
             fontsize=16,
             x=0.55,  # x轴位置（0左，1右，默认0.5，0.95右移）
             y=0.97)   # y轴位置（0下，1上，默认1.0，0.9下移）
ax.set_zlim(0, 1.0)
ax.view_init(elev=15, azim=-72)

# 颜色条：适配新版本，完全实色
mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array(dz)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='Accuracy Intensity')
cbar.solids.set_alpha(0.9)  # 颜色条100%实色

# 保存图片
plt.savefig('3d_bar_chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig('3d_bar_chart.png', dpi=300, bbox_inches='tight')
print('3D Bar Chart (Matplotlib 3.7+ compatible) generated and saved.')