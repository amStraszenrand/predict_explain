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
df_titanics = pd.read_csv("Data/titanic.csv", index_col=0)

# %%
a = 4 +4
# %%
df_iris
# %%

# %%
