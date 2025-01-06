import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
file_path = '2years.csv'  # 文件路径
df = pd.read_csv(file_path)

# 确保日期列正确解析
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d %H:%M')

# 设置日期为索引
df.set_index('Date', inplace=True)

# 对数据进行平滑处理（使用移动平均）
df_smooth = df.rolling(window=24, min_periods=1).mean()  # 平滑窗口为24个时间点（可根据数据密度调整）

# 绘图
fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(10, 20), sharex=True)

# 每个子图对应变量
variables = [
    ('wind', 'Wind speed (m/s)', 'pink'),
    ('temperaturaair', 'Temperature Air (°C)', 'blue'),
    ('tide', 'Tide Level', 'yellow'),
    ('humidity', 'Relative Humidity (%)', 'skyblue'),
    ('atmosphericpressure', 'Atmospheric Pressure (hpa)', 'red'),
    ('rainfall', 'Cumulative Rainfall (mm)', 'gold'),
    ('radiation', 'Relative Humidity', 'deepskyblue'),
    ('temperaturaacqua', 'Temperature Acqua (°C)', 'green'),
    ('waterlevel1hourago', 'Water Level 1 Hour Ago', 'orange')
]

# 遍历变量并绘制平滑后的曲线
for ax, (column, ylabel, color) in zip(axes, variables):
    ax.plot(df_smooth.index, df_smooth[column], color=color, linewidth=1.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True)

# 设置 x 轴和标题
axes[-1].set_xlabel('Date', fontsize=12)
fig.suptitle('Smoothed Time Series of Environmental Variables', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])  # 调整子图间距

# 保存图像
plt.savefig('smoothed_time_series_plot.png', dpi=300)
plt.show()
