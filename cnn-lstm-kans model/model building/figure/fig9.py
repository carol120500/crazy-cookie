import matplotlib.pyplot as plt
import numpy as np

# 数据
percentage = [70, 60, 50]
lstm_r2 = [0.95, 0.95, 0.94]
lstm_mse = [0.0038, 0.00422, 0.00446]
lstm_kans_r2 = [0.99, 0.98, 0.98]
lstm_kans_mse = [0.00113, 0.00115, 0.00173]
cnn_lstm_r2 = [0.98, 0.98, 0.95]
cnn_lstm_mse = [0.00119, 0.002, 0.00337]
cnn_lstm_kans_r2 = [0.99, 0.99, 0.98]
cnn_lstm_kans_mse = [0.00072, 0.00069, 0.0013]

# 创建子图，1行2列
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 颜色定义
colors = ['blue', 'orange', 'green', 'red']
models = ['LSTM', 'LSTM-KANs', 'CNN-LSTM', 'CNN-LSTM-KANs']

# 绘制R²折线图
axs[0].plot(percentage, lstm_r2, marker='o', label='LSTM', color='blue')
axs[0].plot(percentage, lstm_kans_r2, marker='o', label='LSTM-KANS', color='orange')
axs[0].plot(percentage, cnn_lstm_r2, marker='o', label='CNN-LSTM', color='green')
axs[0].plot(percentage, cnn_lstm_kans_r2, marker='o', label='CNN-LSTM-KANS', color='red')

axs[0].set_title('MSE Performance by Training Set Percentage')
axs[0].set_xlabel('Training Set Percentage')
axs[0].set_ylabel('R²')
axs[0].legend(loc='best')
axs[0].grid(True)

# 绘制MSE折线图
axs[1].plot(percentage, lstm_mse, marker='o', label='LSTM', color='blue')
axs[1].plot(percentage, lstm_kans_mse, marker='o', label='LSTM-KANS', color='orange')
axs[1].plot(percentage, cnn_lstm_mse, marker='o', label='CNN-LSTM', color='green')
axs[1].plot(percentage, cnn_lstm_kans_mse, marker='o', label='CNN-LSTM-KANS', color='red')

axs[1].set_title('MSE Performance by Training Set Percentage')
axs[1].set_xlabel('Training Set Percentage')
axs[1].set_ylabel('MSE')
axs[1].legend(loc='best')
axs[1].grid(True)

# 调整布局并显示
plt.tight_layout()
plt.show()
