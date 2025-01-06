import numpy as np
import pandas as pd
import scipy.stats as stats

# 读取数据
df = pd.read_csv('filtered_dataset.csv')
water_level = df['water level'].dropna().values

# 定义GEV分布参数
shape, loc, scale = -0.3, 0, 1

# 生成10000个拟合数据
num_samples = 10000

# 时间序列从2024-01-01开始，每小时的时间戳
start_date = '2024-01-01'
time_range = pd.date_range(start=start_date, periods=num_samples, freq='h')
time_frac = np.linspace(0, 1, num_samples)  # 从0到1的时间百分比

# 随机峰值，每12个数据一个周期
num_peaks = int(num_samples / 12)  # 峰值的数量
peak_heights = np.random.uniform(0.1, 1.0, num_peaks)  # 随机峰值高度
frequencies = np.random.uniform(0.01, 0.1, num_peaks)  # 随机周期频率

# 组合多个周期的水位变化
water_level_trend = np.zeros(num_samples)
for i in range(num_peaks):
    start_idx = i * 12
    end_idx = start_idx + 12
    peak_idx = start_idx + 6 # 峰值发生的位置
    rise = np.linspace(0, peak_heights[i], peak_idx - start_idx)  # 上升阶段
    decline = np.linspace(peak_heights[i], 0, end_idx - peak_idx)  # 降落阶段
    water_level_trend[start_idx:peak_idx] = rise
    water_level_trend[peak_idx:end_idx] = decline

# 使用拟合的GEV限制水位数据范围
water_level_trend = np.clip(water_level_trend, loc, loc + scale * (1 + shape))

# 生成时间戳和水位数据
simulated_data = pd.DataFrame({'water level': water_level_trend})
simulated_data['Date'] = time_range.strftime('%Y/%m/%d %H:%M')

# 保存生成的数据
simulated_data.to_csv('simulated_data_with_gev_limit_0_1_3.csv', index=False)

# 输出前   条数据并保存为新数据集
subset_data = simulated_data.head(2160)
subset_data.to_csv('simulated_data_k=0.3.csv', index=False)
print(subset_data.head())


subset_data = simulated_data.head(8640)
subset_data.to_csv('simulated_data1_k=0.3.csv', index=False)
print(subset_data.head())