import pandas as pd

# 读取数据
# 假设文件是 CSV 格式
# 例如：2year.csv 文件中包含上述列
df = pd.read_csv('2years.csv')

# 确保日期列是 datetime 类型
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d %H:%M')

# 定义季节分类函数
def assign_season(month):
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Autumn'
    else:
        return 'Winter'

# 添加季节列
df['Season'] = df['Date'].dt.month.apply(assign_season)

# 按季节分割数据
spring = df[df['Season'] == 'Spring']
summer = df[df['Season'] == 'Summer']
autumn = df[df['Season'] == 'Autumn']
winter = df[df['Season'] == 'Winter']

# 将不同季节的数据保存到文件中（可选）
spring.to_csv('spring_data.csv', index=False)
summer.to_csv('summer_data.csv', index=False)
autumn.to_csv('autumn_data.csv', index=False)
winter.to_csv('winter_data.csv', index=False)

# 打印每个季节的基本信息
print("春天的数据：", spring.shape)
print("夏天的数据：", summer.shape)
print("秋天的数据：", autumn.shape)
print("冬天的数据：", winter.shape)
