import pandas as pd
from sklearn.impute import KNNImputer

# 创建示例数据集
filepath = 'try1.csv'
data = pd.read_csv(filepath) ##从CSV文件读取数据并存储在data变量中。
print(data.head()) ##打印数据的前几行以查看内容。
print(data.shape) ##打印数据的形状（行数和列数）。


# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 显示原始数据
print("原始数据:")
print(df)

# 选择要插补的列
features_to_impute = df.drop(columns=['Date'])

# 初始化KNN插补器
imputer = KNNImputer(n_neighbors=3)

# 进行插补
imputed_values = imputer.fit_transform(features_to_impute)

# 将插补后的值转换回DataFrame
imputed_df = pd.DataFrame(imputed_values, columns=features_to_impute.columns)

# 添加回'数据'列
imputed_df['Date'] = df['Date']

# 调整列顺序
imputed_df = imputed_df[['Date'] + list(features_to_impute.columns)]

# 显示插补后的数据
print("\n插补后的数据:")
print(imputed_df)
imputed_df.to_csv('outputdata8.csv', index=False)
print("数据已导出为 'outputdata1.csv'")