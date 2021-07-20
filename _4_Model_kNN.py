# %%
import seaborn as sns

# %%
import _2_Preprocessing

X_train, X_test, y_train, y_test = _2_Preprocessing.X_train, _2_Preprocessing.X_test, _2_Preprocessing.y_train, _2_Preprocessing.y_test

# %%
import Objects.my_KNeighborsClassifier as my_kNN

model = my_kNN.my_KNeighborsClassifier(n_neighbors=5, weights="uniform")

# %%
model.fit(X_train, y_train)
X_predict_explain=X_test
#print(X_predict_explain)

y_predict_explain=model.predict_explain(X_predict_explain)
y_predict_explain.to_json("kNN_predict_explain.json")
# %%

import Objects.my_Confusion_Matrix as my_cm

y_pred= y_predict_explain["Prediction"]

sns.heatmap(my_cm.my_confusion_matrix(y_test, y_pred, y_predict_explain["Confidence"]), annot=True)

# %%

# %%

# %%
