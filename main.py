import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# 1. 读取数据
df = pd.read_excel("加油站周销量真实数据.xlsx", dtype={'ds': str})
df['ds'] = pd.to_datetime(df['ds'] + '-1', format='%Y-W%W-%w')   # 周一为代表日

# ================== 关键改动1：使用2023+2024两年完整数据训练 ==================
train = df[df['ds'].dt.year.isin([2023, 2024])].copy()   # 以前只用2024，现在用两年！

# 2. 建模（参数我又微调了一下，两年数据后更稳）
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.25,      # 两年数据后趋势更稳，稍微收紧一点
    seasonality_prior_scale=20         # 季节性更强，让它多记住一点历史规律
)

m.add_regressor('is_promo', standardize=False)
m.add_regressor('is_holiday', standardize=False)
m.fit(train)

# 3. 生成2025年完整52周（确保W01不丢）
future = m.make_future_dataframe(periods=55, freq='W-MON')
future = future[future['ds'] >= '2025-01-01']
future = future.iloc[:52].copy()       # 严格取52周

# 2025年活动先全设0（后面知道再改）
future['is_promo'] = 0
future['is_holiday'] = 0

# 如果你已经计划好2025年哪几周促销，直接在这里标1
# 例如：第8、18、28、38周搞大促销
# future.loc[future['ds'].dt.week.isin([8,18,28,38]), 'is_promo'] = 1

forecast = m.predict(future)

# 4. 输出结果
result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
result['周'] = '2025-W' + result['ds'].dt.isocalendar().week.astype(str).str.zfill(2)
result['预测销量'] = result['yhat'].round(0).astype(int)
result = result[['周', '预测销量', 'yhat_lower', 'yhat_upper']]
result.columns = ['周', '预测销量', '下限', '上限']

result.to_excel("2025年每周销量预测_两年数据版.xlsx", index=False)

print("=== 两年数据版预测完成 ===")
print("训练数据：2023 + 2024 共104周")
print("2025年预计总销量：", result['预测销量'].sum())
print("第一周（W01）预测：", result.iloc[0]['预测销量'])
print("最后一周（W52）预测：", result.iloc[-1]['预测销量'])