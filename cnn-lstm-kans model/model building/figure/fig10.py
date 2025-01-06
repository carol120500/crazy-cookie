import matplotlib.pyplot as plt
import numpy as np

# 数据
percentage = [70, 60, 50]
lstm_r2 = [0.95, 0.95, 0.94]
lstm_mse = [0.0038, 0.00422, 0.00446]
lstm_kans_r2 = [0.99, 0.99, 0.98]
lstm_kans_mse = [0.00113, 0.00115, 0.00173]
cnn_lstm_r2 = [0.98, 0.98, 0.95]
cnn_lstm_mse = [0.00119, 0.002, 0.00337]
cnn_lstm_kans_r2 = [0.99, 0.99, 0.98]
cnn_lstm_kans_mse = [0.00072, 0.00069, 0.0013]

# 创建6个子图，每个对应一个百分比
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# 颜色定义
colors = ['blue', 'orange', 'green', 'red']

# 70% Training Set
axs[0, 0].bar(['LSTM', 'LSTM-KANS', 'CNN-LSTM', 'CNN-LSTM-KANS'],
              [lstm_r2[0], lstm_kans_r2[0], cnn_lstm_r2[0], cnn_lstm_kans_r2[0]],
              color=colors)
axs[0, 0].set_title('70% Training Set')
axs[0, 0].set_ylabel('R²')

axs[1, 0].bar(['LSTM', 'LSTM-KANS', 'CNN-LSTM', 'CNN-LSTM-KANS'],
              [lstm_mse[0], lstm_kans_mse[0], cnn_lstm_mse[0], cnn_lstm_kans_mse[0]],
              color=colors)
axs[1, 0].set_title('70% Training Set')
axs[1, 0].set_ylabel('MSE')

# 60% Training Set
axs[0, 1].bar(['LSTM', 'LSTM-KANS', 'CNN-LSTM', 'CNN-LSTM-KANS'],
              [lstm_r2[1], lstm_kans_r2[1], cnn_lstm_r2[1], cnn_lstm_kans_r2[1]],
              color=colors)
axs[0, 1].set_title('60% Training Set')

axs[1, 1].bar(['LSTM', 'LSTM-KANS', 'CNN-LSTM', 'CNN-LSTM-KANS'],
              [lstm_mse[1], lstm_kans_mse[1], cnn_lstm_mse[1], cnn_lstm_kans_mse[1]],
              color=colors)
axs[1, 1].set_title('60% Training Set')

# 50% Training Set
axs[0, 2].bar(['LSTM', 'LSTM-KANS', 'CNN-LSTM', 'CNN-LSTM-KANS'],
              [lstm_r2[2], lstm_kans_r2[2], cnn_lstm_r2[2], cnn_lstm_kans_r2[2]],
              color=colors)
axs[0, 2].set_title('50% Training Set')

axs[1, 2].bar(['LSTM', 'LSTM-KANS', 'CNN-LSTM', 'CNN-LSTM-KANS'],
              [lstm_mse[2], lstm_kans_mse[2], cnn_lstm_mse[2], cnn_lstm_kans_mse[2]],
              color=colors)
axs[1, 2].set_title('50% Training Set')

plt.tight_layout()
plt.show()
