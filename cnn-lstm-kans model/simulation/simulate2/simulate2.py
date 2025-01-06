import numpy as np
import pandas as pd
from scipy.stats import genextreme
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. 数据读取与预处理
try:
    data = pd.read_csv('filtered_dataset.csv')
except FileNotFoundError:
    print("指定的数据文件不存在，请检查文件名和路径是否正确。")
    raise

# 检查是否存在响应变量 'waterlevel' 和解释变量
if 'waterlevel' in data.columns:
    y = data['waterlevel'].values.reshape(-1, 1)  # 确保y为二维数组形式
    print("响应变量 'waterlevel' 读取成功，形状为:", y.shape)
else:
    raise KeyError("列 'waterlevel' 不存在于 DataFrame 中。")

X = data[[ 'waterlevel1hourago','tide', 'wind', 'atmosphericpressure',
          'temperaturaair', 'temperaturaacqua', 'rainfall', 'radiation', 'humidity']]
print("解释变量读取成功，形状为:", X.shape)

# 数据标准化
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)
print("响应变量标准化后形状为:", y_scaled.shape)

# 为每个解释变量单独创建StandardScaler对象并进行标准化，保存到字典中
scalers_X = {}
for feature in X.columns:
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(X[[feature]])
    X.loc[:, feature] = feature_scaled.flatten()
    scalers_X[feature] = scaler
    print(f"{feature} 变量标准化后形状为:", feature_scaled.shape)

X_scaled = np.column_stack([X[col].values for col in X.columns])
print("所有解释变量标准化后拼接的形状为:", X_scaled.shape)

# 多元回归分析
model_formula = "waterlevel ~   waterlevel1hourago+tide +  rainfall + wind + atmosphericpressure  +temperaturaair + temperaturaacqua + radiation + humidity"
model = ols(model_formula, data=pd.DataFrame(np.concatenate([y, X_scaled], axis=1), columns=['waterlevel'] + list(X.columns))).fit()
print(model.summary())

# GEV参数估计

gev_data = data['waterlevel']

shape, loc, scale = genextreme.fit(gev_data)

params = genextreme.fit(gev_data)
# 打印拟合的参数
print(f"Shape (ξ): {shape}, Location (μ): {loc}, Scale (σ): {scale}")


# 使用调整后的参数生成模拟数据
num_simulations =14975
simulated_values = genextreme.rvs(*params, size=num_simulations)

if 'scaler_y' in locals():
    simulated_values = scaler_y.inverse_transform(simulated_values.reshape(-1, 1)).flatten()

# 输出所有变量的模拟数据
simulated_data = pd.DataFrame(simulated_values, columns=['waterlevel'])

for feature in X.columns:
    feature_scaled = X_scaled[:, X.columns.get_loc(feature)].reshape(-1, 1)
    feature_data = scalers_X[feature].inverse_transform(feature_scaled).flatten()
    simulated_data[feature] = feature_data

simulated_data['Date'] = pd.date_range(start='2024-01-01', periods=len(simulated_data), freq='h').strftime('%Y/%m/%d %H:%M')

# 保存到CSV文件
simulated_data.to_csv('input6.csv', index=False)

# 输出前   条数据并保存为新数据集
subset_data = simulated_data.head(2160)
subset_data.to_csv('simulated_data_k=real_test.csv', index=False)
print(subset_data.head())


subset_data = simulated_data.head(8640)
subset_data.to_csv('simulated_data_k=real_test2.csv', index=False)
print(subset_data.head())