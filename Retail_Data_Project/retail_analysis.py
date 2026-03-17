import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('retail_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Summary
print(df.describe())

# Sales trend
df.groupby('Date')['Sales'].sum().plot()
plt.title("Sales Trend")
plt.savefig("sales_trend.png")
plt.close()

# Correlation
corr = df[['Sales','Profit']].corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# Prediction (simple)
df['Day'] = df['Date'].dt.day
X = df[['Day']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

plt.scatter(X, y)
plt.plot(X, pred)
plt.title("Sales Prediction")
plt.savefig("prediction_plot.png")
plt.close()

print("Project Completed!")
