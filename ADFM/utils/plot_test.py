import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('seaborn-whitegrid')

# # Define your font settings
# font = {
#     'family': 'Times New Roman',  # Font family (e.g., 'serif', 'sans-serif', 'monospace')
#     'weight': 'normal',  # Font weight ('normal', 'bold', 'light', 'heavy')
#     'size': 32,  # Font size
#     'style': 'normal'  # Font style ('normal', 'italic', 'oblique')
# }
#
# # Set the font properties
# plt.rcParams['font.family'] = font['family']
# plt.rcParams['font.weight'] = font['weight']
# plt.rcParams['font.size'] = font['size']
# plt.rcParams['font.style'] = font['style']

values_series1 = [0.957,0.694, 0.646, 0.345]
values_series2 = [0.95825386, 0.94789432, 0.84886426, 0.568264]
values_series3 = [0.961, 0.95756006, 0.94735723, 0.77110775]

# methods = ['Original', '3D Target', 'Varied Cloth']
methods = ['Original', 'Platform', 'Sphere', 'Rod']

# 设定每组柱子之间的空间
bar_width = 0.15

# 计算每组柱子的位置
index = np.arange(len(methods))
index_series1 = index - bar_width
index_series2 = index
index_series3 = index +  bar_width

# 设置图形的大小
plt.figure(figsize=(10, 6))

# 绘制柱状图
bars1 = plt.bar(index_series1, values_series1, bar_width, label='Fling bot', color='orange')
bars2 = plt.bar(index_series2, values_series2, bar_width, label='Naive model', color='g')
bars3 = plt.bar(index_series3, values_series3, bar_width, label='Foundation model', color='b')

# 添加方法名称到x轴
plt.xticks(index, methods, fontname='Arial', fontsize=12)

# 设置x轴和y轴的标签
plt.xlabel('Tasks', fontname='Arial', fontsize=14)
plt.ylabel('Normalized Performance', fontname='Arial', fontsize=14)

# 添加图例
plt.legend()

# 确保标签和标题的布局
plt.tight_layout()

# y轴范围
plt.ylim(0.3, 1.0)

# 保存图形
plt.savefig('comparison_bar_chart.pdf', format='pdf', dpi=300)

# 展示图形
plt.show()
