import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# reading dataset from csv file
dataset = pd.read_csv("score.csv")

# separating train and test sets from each other
train, test = train_test_split(dataset, test_size=0.3)

X_train = train.drop("Hours", axis=1)
y_train = train.loc[:, "Hours"]

X_test = test.drop("Hours", axis=1)
y_test = test.loc[:, "Hours"]

model_1 = LogisticRegression()

# LogisticRegression
model_1.fit(X_train, y_train)

predictions = model_1.predict(X_test)

print(predictions)
