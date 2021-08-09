# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# %%
df_iris = pd.read_csv("Data/iris.csv")
df_titanic = pd.read_csv("Data/titanic.csv", index_col=0)
df_loans = pd.read_csv("Data/loans_2007.csv", index_col = 0)#, dtype={"member_id" : int})
df_cells = pd.read_csv("Data/cell_samples.csv")

# %%

# https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/
# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.142.2022&rep=rep1&type=pdf

df_nursery = pd.read_csv("Data/nursery_ordered.data", index_col = False, names = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "class"])

# %%
