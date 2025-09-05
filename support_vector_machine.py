from sklearn.svm import LinearSVC
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix,precision_score

df=pd.read_csv("Breast_cancer_dataset.csv")

df["diagnosis"]=pd.factorize(df["diagnosis"])[0]+1

x = df[['radius_mean','texture_mean','perimeter_mean','area_mean',
        'smoothness_mean','compactness_mean','concavity_mean',
        'concave points_mean','symmetry_mean','fractal_dimension_mean',
        'radius_se','texture_se','perimeter_se','area_se',
        'smoothness_se','compactness_se','concavity_se',
        'concave points_se','symmetry_se','fractal_dimension_se',
        'radius_worst','texture_worst','perimeter_worst','area_worst',
        'smoothness_worst','compactness_worst','concavity_worst',
        'concave points_worst','symmetry_worst','fractal_dimension_worst']]



y=df["diagnosis"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearSVC(random_state=42,max_iter=5000,C=0.01,penalty="l2",class_weight="balanced")

model.fit(x_train,y_train)

y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)*100

print(f"accuracy = {accuracy:.2f}%")

recall=recall_score(y_test,y_pred,pos_label=1)*100

print(f"Recall score= {recall:.2f}%")

confusion_matrixs=confusion_matrix(y_test,y_pred)
print(confusion_matrixs)

precision=precision_score(y_test,y_pred)*100

print(f"precision= {precision:.2f}%")



plt.plot(y_test.values[:200],color="blue",label="Test data")
plt.plot(y_pred[:200],color="yellow",label="predicted data")

plt.legend()


plt.show()

