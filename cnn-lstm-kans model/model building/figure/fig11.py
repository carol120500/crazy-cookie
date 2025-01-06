import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取对应的数据集 CSV 文件
filepath_cnn_lstm = 'cnn_lstm_dataset.csv'
cnn_lstm_results = pd.read_csv(filepath_cnn_lstm)

filepath_cnn_lstm_kans = 'cnn_lstm_kans_dataset.csv'
cnn_lstm_kans_results = pd.read_csv(filepath_cnn_lstm_kans)

# 将 'Date' 列转换为 datetime 格式
cnn_lstm_results['Date'] = pd.to_datetime(cnn_lstm_results['Date'])
cnn_lstm_kans_results['Date'] = pd.to_datetime(cnn_lstm_kans_results['Date'])

# 只显示2023年1月的日期
cnn_lstm_results = cnn_lstm_results[(cnn_lstm_results['Date'] >= '2023-03-25') & (cnn_lstm_results['Date'] <= '2023-03-28')]
cnn_lstm_kans_results = cnn_lstm_kans_results[(cnn_lstm_kans_results['Date'] >= '2023-03-25') & (cnn_lstm_kans_results['Date'] <= '2023-03-28')]

# 创建一个单一图表
fig, ax = plt.subplots(figsize=(10, 5))

# CNN-LSTM模型和CNN-LSTM-Kans模型
ax.plot(cnn_lstm_results['Date'], cnn_lstm_results['Predicted'], color='blue', label='CNN_LSTM')


ax.plot(cnn_lstm_kans_results['Date'], cnn_lstm_kans_results['Predicted'], color='red', label='CNN_LSTM_Kans')
ax.plot(cnn_lstm_kans_results['Date'], cnn_lstm_kans_results['Actual'], color='black', linestyle='--',label='Observed')

# 设置图表属性
ax.set_title('CNN-LSTM & CNN-LSTM-Kans Predictions')
ax.set_xlabel('Time')
ax.set_ylabel('Water Level (m)')
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
ax.legend()
ax.grid(True)
ax.set_facecolor('#f0f0f0')  # 设置灰色背景

plt.tight_layout()
plt.show()
