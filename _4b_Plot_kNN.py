# %%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as plx
import altair as alt

# %%
import _4_Model_kNN

X_train, X_test, y_train, y_test = _4_Model_kNN.X_train, _4_Model_kNN.X_test, _4_Model_kNN.y_train, _4_Model_kNN.y_test
X_predict = _4_Model_kNN.X_predict

y_predict_explain, y_predict_rfc, y_predict_rfc = _4_Model_kNN.y_predict_explain, _4_Model_kNN.y_predict_dtc, _4_Model_kNN.y_predict_rfc

y_predict_explain_rfc = pd.DataFrame({
    "Prediction": y_predict_rfc, 
    "Confidence" : [True] * len(y_predict_rfc)
}, index=y_predict_explain.index)

# %%
X_predict
# %%
import Objects.my_TSNE_PCA_Plot as plot_predictions

plot_predictions = plot_predictions.Plot_TSNE_PCA(X_train,y_train)

# %%
fig = plot_predictions.TSNE()
fig.write_html("Html/tsne_blank.html")
fig.show()


# %%
fig = plot_predictions.TSNE(X_predict, y_predict_explain)
fig.write_html("Html/tsne_predictions.html")
fig.show()

# %%
fig = plot_predictions.TSNE(X_predict, y_predict_explain, with_train_data=False)
fig.write_html("Html/tsne_predictions_solo.html")
fig.show()

# %%
fig = plot_predictions.TSNE(X_predict, y_predict_explain_rfc)
fig.write_html("Html/tsne_predictions_rfc.html")
fig.show()

# %%
fig = plot_predictions.TSNE(X_predict, y_predict_explain_rfc, with_train_data=False)
fig.write_html("Html/tsne_predictions_rfc_solo.html")
fig.show()
# %%
fig = plot_predictions.PCA()
fig.write_html("Html/pca_blank.html")
fig.show()

# %%
fig = plot_predictions.PCA(X_predict, y_predict_explain)
fig.write_html("Html/pca_predictions.html")
fig.show()

# %%
fig = plot_predictions.PCA(X_predict, y_predict_explain, with_train_data=False)
fig.write_html("Html/pca_predictions_solo.html")
fig.show()

# %%
fig = plot_predictions.PCA(X_predict, y_predict_explain_rfc)
fig.write_html("Html/pca_predictions_rfc.html")
fig.show()

# %%
fig = plot_predictions.PCA(X_predict, y_predict_explain_rfc, with_train_data=False)
fig.write_html("Html/pca_predictions_rfc_solo.html")
fig.show()
# %%
import Objects.my_Confusion_Matrix as my_cm

y_pred= y_predict_explain["Prediction"]

plt.figure(figsize=(10,8))
sns.heatmap(my_cm.my_confusion_matrix(y_test, y_pred, y_predict_explain["Confidence"]), annot=True, fmt=".1f");
plt.xticks(rotation=45)
plt.ylim((y_test.nunique(), -0.5))
plt.yticks(rotation=45)
plt.show();
# %%

# %%
