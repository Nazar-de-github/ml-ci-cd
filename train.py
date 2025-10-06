import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from model import train_model, evaluate_model
from metrics_and_plots import save_metrics, plot_confusion_matrix

data = pd.read_csv("data.csv")
features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(data[features])
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

model = train_model(X_train, y_train)

metrics = evaluate_model(model, X_test, y_test)
save_metrics(metrics)

print(metrics)


