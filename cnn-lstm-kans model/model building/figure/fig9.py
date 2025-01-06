import matplotlib.pyplot as plt
import pandas as pd

# 数据
data = {
    "training set percentage": [70, 60, 50],
    "LSTM model": [(0.93, 0.00446), (0.95, 0.00515), (0.94, 0.00446)],
    "LSTM-KANS model": [(0.97, 0.00115), (0.99, 0.00178), (0.99, 0.00117)],
    "CNN-LSTM model": [(0.95, 0.00281), (0.96, 0.00337), (0.97, 0.00281)],
    "CNN-LSTM-KANS model": [(0.98, 0.00069), (0.98, 0.000144), (0.99, 0.00013)]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 提取R2和MSE数据
r2 = df.drop(columns=["training set percentage"]).apply(lambda x: x[0])
mse = df.drop(columns=["training set percentage"]).apply(lambda x: x[1])

# 可视化R2和MSE
fig, ax1 = plt.subplots(figsize=(10, 6))

# R2图
ax1.bar(df['training set percentage'], r2, color='skyblue', label='R2')
ax1.set_xlabel('Training Set Percentage')
ax1.set_ylabel('R2', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

# MSE图
ax2 = ax1.twinx()
ax2.plot(df['training set percentage'], mse, color='red', marker='o', label='MSE')
ax2.set_ylabel('MSE', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 标题和图例
plt.title('Model Performance Comparison')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.grid(True)
plt.tight_layout()
plt.show()
