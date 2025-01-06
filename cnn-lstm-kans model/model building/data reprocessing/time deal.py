import pandas as pd

df_2years = pd.read_csv("2years.csv")

# 确保 datetime 列为 datetime 格式
df_2years['datetime'] = pd.to_datetime(df_2years['Date'], format='%Y/%m/%d %H:%M')

# 筛选 3-month 数据集：2023 年 1 月 1 日到 2023 年 3 月 31 日
start_3month = "2023-01-01"
end_3month = "2023-03-31"
df_3month = df_2years[(df_2years['datetime'] >= start_3month) & (df_2years['datetime'] <= end_3month)]

# 筛选 1-year 数据集：2023 年 1 月 1 日到 2023 年 12 月 31 日
start_1year = "2023-01-01"
end_1year = "2023-12-31"
df_1year = df_2years[(df_2years['datetime'] >= start_1year) & (df_2years['datetime'] <= end_1year)]

# 保存为独立的 CSV 文件
df_3month.to_csv("3month_dataset.csv", index=False)
df_1year.to_csv("1year_dataset.csv", index=False)

print("Datasets saved as '3month_dataset.csv' and '1year_dataset.csv'")
