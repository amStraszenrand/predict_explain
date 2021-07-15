# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# %%
import _1_LoadData

# %%
df_name = "titanic"

# %%
def get_train_test_split(df_name):
    if df_name == "iris":
        df = _1_LoadData.df_iris
        X = df.drop("species", axis=1)
        y = df["species"]
        return train_test_split(X, y, random_state=1, stratify=y, test_size=0.25)
    if df_name == "titanic":
        df = _1_LoadData.df_titanic
        if df["Embarked"].isna().sum() > 0: 
            df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
        means_titanic_age = df.groupby("Pclass").mean()["Age"]
        df['Age'] = df.apply(
            lambda row: means_titanic_age.loc[int(row["Pclass"])] if np.isnan(row['Age']) else row['Age'],
            axis=1
        )
        X = df.drop(["Survived", "Cabin", "Name", "Ticket"], axis=1)
        y = df["Survived"]
        return train_test_split(pd.get_dummies(X, drop_first=True), y, random_state=1, stratify=y, test_size=0.1)


# %%
X_train, X_test, y_train, y_test = get_train_test_split(df_name)
# %%
