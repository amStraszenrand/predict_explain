# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as plx
import altair as alt

# %%
import _4_Model_kNN

X_train, X_test, y_train, y_test = _4_Model_kNN.X_train, _4_Model_kNN.X_test, _4_Model_kNN.y_train, _4_Model_kNN.y_test
X_predict_explain, y_predict_explain = _4_Model_kNN.X_predict_explain, _4_Model_kNN.y_predict_explain

# %%
import Objects.Plot_kNN

# %%
plot_kNN = Plot_kNN(X_train,y_train)
plot_kNN.TSNE(X_predict_explain, y_predict_explain)


# %%
plot_kNN.PCA(X_predict_explain, y_predict_explain)
# %%

# %%
