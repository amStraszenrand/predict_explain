# %%
import _2_Preprocessing

df_name = "nursery"

df, X_train, X_test, y_train, y_test =_2_Preprocessing.get_train_test_split(df_name)
# %%

#X_test = X_test[:5]
#y_test = y_test[:5]

# %%
import Objects.my_KNeighborsClassifier as my_kNN
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

my_model = my_kNN.my_KNeighborsClassifier(n_neighbors=5,  weights="distance", metric="manhattan")
logreg = LogisticRegression(C=3, max_iter=10000, tol=0.000001)
dtc = DecisionTreeClassifier(max_depth=14)
rfc = RandomForestClassifier(n_estimators=200, max_samples=0.8, max_depth=14)
# %%
from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

my_model.fit(X_train_smote, y_train_smote)
logreg.fit(X_train, y_train)
dtc.fit(X_train_smote, y_train_smote)
rfc.fit(X_train_smote, y_train_smote)
# %%
X_predict=X_test
#print(X_predict)

# %%
from sklearn.metrics import classification_report

y_predict_explain =my_model.predict_explain(X_predict)
y_predict_explain.to_json("kNN_predict_explain.json")

print(classification_report(y_test, y_predict_explain["Prediction"]))

# %%
index = [8024, 8793, 12229]
y_predict_explain
# %%
import plotly as plx

plotly_colors = plx.colors.qualitative.Plotly[1:]
y_plotly_colors = {y:plotly_colors[i] for i, y in enumerate(y_train.unique())}
i = 2
_, fig = my_model.predict_explain(X_predict.loc[[index[i]]], True, y_plotly_colors)
fig.show()

# %%
fig.write_html(f"Html/Explanation_{index[i]}.html")
fig.write_image(f"Figures/Explanation_{index[i]}.png")
#%%

# %%
y_predict_logreg=logreg.predict(X_predict)

print("logreg:")
print(classification_report(y_test, y_predict_logreg))

# %%
y_predict_dtc=dtc.predict(X_predict)

print("dtc:")
print(classification_report(y_test, y_predict_dtc))
# %%
y_predict_rfc=rfc.predict(X_predict)

print("rfc:")
print(classification_report(y_test, y_predict_rfc))

# # %%
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# param_grid = {
#     "n_neighbors" : range(1,7),
#     "weights": ["uniform", "distance"],
#     "metric": ["euclidean", "manhattan", "chebyshev"]
#     }

# rs = GridSearchCV(my_model, param_grid, scoring='balanced_accuracy',
#                   cv=5, verbose=0, n_jobs=-1)

# rs.fit(X_train, y_train)

# print('Best score:', round(rs.best_score_, 3))
# print('Best parameters:', rs.best_params_)
# sgd_best_rs = rs.best_estimator_
# y_pred_test_rs = sgd_best_rs.predict(X_test)
# print(classification_report(y_test, y_pred_test_rs))

# # %%
# param_grid = {
#     "tol": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
#     "C": [0.1, 0.2, 0.5, 1,1.5, 2,3,4,5,6,7,8,9,10],
#     "max_iter": [10000, 20000, 50000],
    
#     }

# rs = RandomizedSearchCV(logreg, param_grid, scoring='balanced_accuracy',
#                   cv=5, verbose=0, n_jobs=-1, n_iter=10)

# rs.fit(X_train, y_train)

# print('Best score:', round(rs.best_score_, 3))
# print('Best parameters:', rs.best_params_)
# sgd_best_rs = rs.best_estimator_
# y_pred_test_rs = sgd_best_rs.predict(X_test)
# print(classification_report(y_test, y_pred_test_rs))

 # %%
# param_grid = {
#     "max_depth": range(1,25)
#     }

# rs = GridSearchCV(dtc, param_grid, scoring='balanced_accuracy',
#                   cv=5, verbose=0, n_jobs=-1)

# rs.fit(X_train, y_train)

# print('Best score:', round(rs.best_score_, 3))
# print('Best parameters:', rs.best_params_)
# sgd_best_rs = rs.best_estimator_
# y_pred_test_rs = sgd_best_rs.predict(X_test)
# print(classification_report(y_test, y_pred_test_rs))

# # %%
# param_grid = {
#     "n_estimators" : [10,50,10,200,400,500,750,1000],
#     "criterion" : ["gini", "entropy"],
#     "max_depth": range(1,15),
#     "min_samples_split": range(2,25),
#     "min_samples_leaf": range(1,20),
#     "max_samples" : [0.05, 0.1, 0.2,0.25, 0.4, 0.5, 0.8, 1]
#     }

# rs = RandomizedSearchCV(rfc, param_grid, scoring='balanced_accuracy',
#                   cv=5, verbose=0, n_jobs=-1, n_iter=200)

# rs.fit(X_train, y_train)

# print('Best score:', round(rs.best_score_, 3))
# print('Best parameters:', rs.best_params_)
# sgd_best_rs = rs.best_estimator_
# y_pred_test_rs = sgd_best_rs.predict(X_test)
# print(classification_report(y_test, y_pred_test_rs))
# # %%

# %%
