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
import Objects.Plot_kNN as plot_kNN

plot_kNN = plot_kNN.Plot_kNN(X_train,y_train)

# %%
fig = plot_kNN.TSNE(X_predict_explain, y_predict_explain)
fig.write_html("tsne.html")
fig.show()

# %%
fig = plot_kNN.PCA(X_predict_explain, y_predict_explain)
fig.write_html("pca.html")
fig.show()
# %%
import Objects.my_Confusion_Matrix as my_cm

y_pred= y_predict_explain["Prediction"]

plt.figure(figsize=(10,8))
sns.heatmap(my_cm.my_confusion_matrix(y_test, y_pred, y_predict_explain["Confidence"]), annot=True);
plt.xticks(rotation=45)
plt.ylim((3, -0.5))
plt.yticks(rotation=45)
plt.show();
# %%
sns.heatmap(np.array(my_cm.my_confusion_matrix(y_test, y_pred, y_predict_explain["Confidence"])))

# %%
