# %%
import _2_Preprocessing

df_name = "nursery"

df, X_train, X_test, y_train, y_test =_2_Preprocessing.get_train_test_split(df_name)

# %%

#X_test = X_test[:5]
#y_test = y_test[:5]

# %%
import Objects.my_KNeighborsClassifier as my_kNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

my_model = my_kNN.my_KNeighborsClassifier(n_neighbors=5,  weights="distance", metric="manhattan")
dtc = DecisionTreeClassifier(max_depth=14)
rfc = RandomForestClassifier(n_estimators=200, max_samples=0.8, max_depth=14)

# %%
from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

my_model.fit(X_train_smote, y_train_smote)
dtc.fit(X_train_smote, y_train_smote)
rfc.fit(X_train_smote, y_train_smote)

# %%
X_predict=X_test
#print(X_predict)

# %%
from sklearn.metrics import classification_report

y_predict_explain =my_model.predict_explain(X_predict)
y_predict_explain.to_json("kNN_predict_explain.json")

print("my_model:")
print(classification_report(y_test, y_predict_explain["Prediction"]))

# %%
y_predict_dtc=dtc.predict(X_predict)

print("dtc:")
print(classification_report(y_test, y_predict_dtc))
# %%
y_predict_rfc=rfc.predict(X_predict)

print("rfc:")
print(classification_report(y_test, y_predict_rfc))

# %%
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# param_grid = {
#     "n_neighbors" : range(1,7),
#     "weights": ["uniform", "distance"],
#     "metric": ["euclidean", "manhattan", "chebyshev"]
#     }

# rs = GridSearchCV(my_model, param_grid, scoring='accuracy',
#                   cv=5, verbose=0, n_jobs=-1)

# rs.fit(X_train, y_train)

# print('Best score:', round(rs.best_score_, 3))
# print('Best parameters:', rs.best_params_)
# sgd_best_rs = rs.best_estimator_
# y_pred_test_rs = sgd_best_rs.predict(X_test)
# print(classification_report(y_test, y_pred_test_rs))

# # %%
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
