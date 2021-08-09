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
def get_train_test_split(df_name):
    if df_name == "iris":
        df = _1_LoadData.df_iris
        X = df.drop("species", axis=1)
        y = df["species"]
        return df, *train_test_split(X, y, random_state=1, stratify=y, test_size=0.25)
    elif df_name == "titanic":
        df = _1_LoadData.df_titanic
        if df["Embarked"].isna().sum() > 0: 
            df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
        means_titanic_age = np.round(df.groupby("Pclass").mean()["Age"], 2)
        df['Age'] = df.apply(
            lambda row: means_titanic_age.loc[int(row["Pclass"])] if np.isnan(row['Age']) else row['Age'],
            axis=1
        )
        X = df.drop(["Survived", "Cabin", "Name", "Ticket", "Embarked"], axis=1)
        y = df["Survived"]
        return df, *train_test_split(pd.get_dummies(X, drop_first=True), y, random_state=1, stratify=y, test_size=0.1)
    elif df_name == "loans":
        df = _1_LoadData.df_loans
    elif df_name == "cells":
        df = _1_LoadData.df_cells
        df.loc[df["BareNuc"] == "?", "BareNuc"] = df["BareNuc"].mode().values[0]
        df["BareNuc"] = df["BareNuc"].astype(int)
        X = df.drop(["ID", "Class"], axis=1)
        y = df["Class"].map({2 : 0, 4 : 1})
        return df, *train_test_split(pd.get_dummies(X, drop_first=True), y, random_state=1, stratify=y, test_size=0.2)
    elif df_name == "nursery":
        df = _1_LoadData.df_nursery
        mask = df["class"] == "recommend"
        df.drop(df[mask].index, inplace=True)
        X = df.drop(["class"], axis = 1)
        y = df["class"]
        #y = df["class"].map({"not_recom": 0, "priority": 1, "recommend": 1, "spec_prior": 1, "very_recom": 1})[mask]
        return df, *train_test_split(pd.get_dummies(X, drop_first=False), y, random_state=1, stratify=y, test_size=0.25)
        #return df, *train_test_split(X, y, random_state=1, stratify=y, test_size=0.25)