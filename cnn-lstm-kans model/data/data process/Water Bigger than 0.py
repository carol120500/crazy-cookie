import pandas as pd

# 输入和输出文件路径
input_file = '2years.csv'  # 替换为实际输入文件路径
output_file = 'filtered_dataset.csv'  # 替换为实际输出文件路径

# 读取数据集
data = pd.read_csv(input_file)

# 筛选 waterlevel 大于 0.5 的数据
filtered_data = data[data['waterlevel'] > 0]

# 保留所有需要的变量
columns_to_keep = [
    'Date', 'tide', 'wind', 'atmosphericpressure',
    'temperaturaair', 'temperaturaacqua', 'rainfall',
    'radiation', 'humidity', 'waterlevel1hourago',
     'waterlevel','number'
]
filtered_data = filtered_data[columns_to_keep]

# 保存新数据集到 CSV 文件
filtered_data.to_csv(output_file, index=False)

print(f"筛选完成！新数据集已保存到 {output_file}")
