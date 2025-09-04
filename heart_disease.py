import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df=pd.read_csv("heart.csv")


x=df[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","thal"]]

y=df["target"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scale=StandardScaler()
x_train=scale.fit_transform(x_train)
x_test=scale.transform(x_test)

model=LogisticRegression(max_iter=2000,random_state=42,penalty="l2",C=1,n_jobs=3)

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

accuracy=accuracy_score(y_test,y_predict)*100


print(f"accuracy = {accuracy:.2f}%")

checking=pd.DataFrame({
    "test":y_test,
    "Predict":y_predict
})

print(checking.head(20))

plt.plot(y_test.values[:200],color="black",label="tested data")
plt.plot(y_predict[:200],color="green",label="predicted data")
plt.legend()
plt.show()