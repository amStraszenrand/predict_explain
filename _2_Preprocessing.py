# %%
import _1_LoadData
# %%
df = _1_LoadData.df_iris
# %%
import _1_LoadData
print(_1_LoadData.a)
# %%
import testo
print(testo.a)

# %%
import tschak
print(tschak.b)



# %%

X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
X = X[['mean area', 'mean compactness']]

y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
y = pd.get_dummies(y, drop_first=True)
# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# %%
