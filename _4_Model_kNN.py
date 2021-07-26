# %%
import seaborn as sns

# %%
import _2_Preprocessing

X_train, X_test, y_train, y_test = _2_Preprocessing.X_train, _2_Preprocessing.X_test, _2_Preprocessing.y_train, _2_Preprocessing.y_test

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

knn = KNeighborsClassifier(n_neighbors=5, weights="uniform")
knn.fit(X_train, y_train)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# %%
import Objects.my_KNeighborsClassifier as my_kNN

my_model = my_kNN.my_KNeighborsClassifier(n_neighbors=5, weights="uniform")

# %%
my_model.fit(X_train, y_train)
X_predict_explain=X_test
#print(X_predict_explain)

y_predict_explain=my_model.predict_explain(X_predict_explain)
y_predict_explain.to_json("kNN_predict_explain.json")
# %%

