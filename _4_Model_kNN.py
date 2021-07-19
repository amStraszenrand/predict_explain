# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as plx


from sklearn import neighbors

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


# %%
import _2_Preprocessing

X_train, X_test, y_train, y_test = _2_Preprocessing.X_train, _2_Preprocessing.X_test, _2_Preprocessing.y_train, _2_Preprocessing.y_test

# %%
import Objects.my_KNeighborsClassifier as my_kNN

model = my_kNN.my_KNeighborsClassifier(n_neighbors=5, weights="uniform", my_predict_proba_threshold=0.75)

# %%
model.fit(X_train, y_train)

# %%
X_predict_explain=X_test
#print(X_predict_explain)

y_predict_explain=model.predict_explain(X_predict_explain)
y_predict_explain.to_csv("test.csv")
# %%

# %%
