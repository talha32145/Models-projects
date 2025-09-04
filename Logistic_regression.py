import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

df=pd.read_csv("Titanic.csv")

cleaning_data=df.fillna({"Age":df["Age"].mean(),"Fare":7},inplace=True)

droping_null=df.dropna(axis=1,thresh=len(df)*0.5,inplace=True)
df["Sex"]=pd.factorize(df["Sex"])[0]+1


x=df[["Age","Sex","Fare"]]
y=df["Survived"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

accuracy=r2_score(y_pred,y_test)*100
print(f"accuracy = {accuracy:.2f}%")

checking=pd.DataFrame({
    "test": y_test,
    "pred":y_pred
})

print(checking.head(10))

new_values=pd.DataFrame({
    "Age":[34.5],
    "Sex":[2],
    "Fare":[7.8292]
})

pred=model.predict(new_values)

print(f"Predicted value = {pred[0]}")

# plt.plot(y_test.values[:200],label="Tested data",color="green")
# plt.plot(y_pred[:200],label="Predicted data",color="black")
# plt.legend()
# plt.show()