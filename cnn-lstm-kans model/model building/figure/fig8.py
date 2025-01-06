import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取对应的数据集 CSV 文件
filepath_lstm = 'lstm_dataset.csv'
lstm_results = pd.read_csv(filepath_lstm)

filepath_lstm_kans = 'lstm_kans_dataset2.csv'
lstm_kans_results = pd.read_csv(filepath_lstm_kans)

filepath_cnn_lstm = 'cnn_lstm_dataset2.csv'
cnn_lstm_results = pd.read_csv(filepath_cnn_lstm)

filepath_cnn_lstm_kans = 'cnn_lstm_kans_dataset2.csv'
cnn_lstm_kans_results = pd.read_csv(filepath_cnn_lstm_kans)

# 将 'Date' 列转换为 datetime 格式
lstm_results['Date'] = pd.to_datetime(lstm_results['Date'])
lstm_kans_results['Date'] = pd.to_datetime(lstm_kans_results['Date'])
cnn_lstm_results['Date'] = pd.to_datetime(cnn_lstm_results['Date'])
cnn_lstm_kans_results['Date'] = pd.to_datetime(cnn_lstm_kans_results['Date'])

# 只显示2023年1月的日期
lstm_results = lstm_results[(lstm_results['Date'] >= '2023-01-01') & (lstm_results['Date'] <= '2023-01-07')]
lstm_kans_results = lstm_kans_results[(lstm_kans_results['Date'] >= '2023-01-01') & (lstm_kans_results['Date'] <= '2023-01-07')]
cnn_lstm_results = cnn_lstm_results[(cnn_lstm_results['Date'] >= '2023-01-01') & (cnn_lstm_results['Date'] <= '2023-01-07')]
cnn_lstm_kans_results = cnn_lstm_kans_results[(cnn_lstm_kans_results['Date'] >= '2023-01-01') & (cnn_lstm_kans_results['Date'] <= '2023-01-07')]

# 创建一个 4x1 子图面板
fig, axs = plt.subplots(4, 1, figsize=(10, 20))

# 第一幅图：LSTM模型
axs[0].plot(lstm_results['Date'], lstm_results['Predicted'], color='green', label='LSTM')
axs[0].plot(lstm_results['Date'], lstm_results['Actual'], color='black', label='Observed', linestyle='--')
axs[0].set_title('LSTM Predictions')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Water Level (m)')
axs[0].xaxis.set_major_locator(mdates.DayLocator())
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
axs[0].legend(loc='lower left')  # 标签位置：左下角
axs[0].grid(True)
axs[0].set_facecolor('#f0f0f0')  # 设置灰色背景

# 第二幅图：LSTM-Kans模型
axs[1].plot(lstm_kans_results['Date'], lstm_kans_results['Predicted'], color='blue', label='LSTM_Kans')
axs[1].plot(lstm_kans_results['Date'], lstm_kans_results['Actual'], color='black', label='Observed', linestyle='--')
axs[1].set_title('LSTM-Kans Predictions')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Water Level (m)')
axs[1].xaxis.set_major_locator(mdates.DayLocator())
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
axs[1].legend(loc='lower left')  # 标签位置：左下角
axs[1].grid(True)
axs[1].set_facecolor('#f0f0f0')  # 设置灰色背景

# 第三幅图：CNN-LSTM模型
axs[2].plot(cnn_lstm_results['Date'], cnn_lstm_results['Predicted'], color='purple', label='CNN_LSTM')
axs[2].plot(cnn_lstm_results['Date'], cnn_lstm_results['Actual'], color='black', label='Observed', linestyle='--')
axs[2].set_title('CNN-LSTM Predictions')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Water Level (m)')
axs[2].xaxis.set_major_locator(mdates.DayLocator())
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
axs[2].legend(loc='lower left')  # 标签位置：左下角
axs[2].grid(True)
axs[2].set_facecolor('#f0f0f0')  # 设置灰色背景

# 第四幅图：CNN-LSTM-Kans模型
axs[3].plot(cnn_lstm_kans_results['Date'], cnn_lstm_kans_results['Predicted'], color='red', label='CNN_LSTM_Kans')
axs[3].plot(cnn_lstm_kans_results['Date'], cnn_lstm_kans_results['Actual'], color='black', label='Observed', linestyle='--')
axs[3].set_title('CNN-LSTM-Kans Predictions')
axs[3].set_xlabel('Time')
axs[3].set_ylabel('Water Level (m)')
axs[3].xaxis.set_major_locator(mdates.DayLocator())
axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
axs[3].legend(loc='lower left')  # 标签位置：左下角
axs[3].grid(True)
axs[3].set_facecolor('#f0f0f0')  # 设置灰色背景

plt.tight_layout()
plt.show()

