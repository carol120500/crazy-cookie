import pandas as pd

# 读取数据
data = pd.read_csv('inputdata2.csv')

# 转换 'Date' 为日期时间格式
data['Date'] = pd.to_datetime(data['Date'])

# 提取每天的最高水位及相关变量
highest_water_level = data.groupby(data['Date'].dt.date).apply(lambda x: x[x['waterlevel'] == x['waterlevel'].max()])

# 重命名列名
highest_water_level = highest_water_level.reset_index(drop=True)

# 格式化 'Date' 列
highest_water_level['Date'] = highest_water_level['Date'].dt.strftime('%Y/%m/%d %H:%M')

# 存储为新的数据集 'inputdata5'
highest_water_level.to_csv('inputdata5.csv', index=False)

# 输出结果
print(highest_water_level)
