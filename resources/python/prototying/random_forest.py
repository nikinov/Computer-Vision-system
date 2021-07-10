import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


df = pd.read_csv("csv/images.csv")

df.drop(['Images_Analyzed'], axis=1, inplace=True)
df.drop(['User'], axis=1, inplace=True)

# handle missing values
# df = df.dropna

# convert non-numeric data to numeric

df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 0

# define dependent variables
Y = df["Productivity"].values
Y = Y.astype('int')

# define independent variables
X = df.drop(labels=['Productivity'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=12)


model = RandomForestClassifier(n_estimators=10, random_state=20)

model.fit(X_train, Y_train)


prediction = model.predict(X_test)

metrics = metrics.accuracy_score(Y_test, prediction)

print("Accuracy = " + str(metrics))

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)

