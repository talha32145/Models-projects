import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = pd.read_csv('BTC-2017min.csv')

# date ko datetime mein convert
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('BTC-2017min.csv')

df["date"] = pd.to_datetime(df['date'])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["hour"] = df["date"].dt.hour


x = df[["year","month","day","hour","open","high","low","Volume BTC","Volume USD"]]
y = df["close"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=5,random_state=42)
model.fit(x_train,y_train)

pred = model.predict(x_test)

accuracy = r2_score(y_test,pred)*100
print(f"accuracy= {accuracy:.4f}")
new_data = pd.DataFrame([{
    "year": 2025,
    "month": 9,
    "day": 3,
    "hour": 12,
    "open": 111000,   # Example opening price
    "high": 112000,   # Example high
    "low": 110500,    # Example low
    "Volume BTC": 1200,  
    "Volume USD": 130000000
}])


prediction=model.predict(new_data)
print(f"Prediction of close = {prediction[0]}")

plt.figure(figsize=(10,6))
plt.plot(y_test.values[:200],label="Test value",color="blue")
plt.plot(pred[:200],label="Predicted value",color="red")
plt.legend()
plt.title("Test vs Predicted")
plt.show()
