# %%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as plx
import altair as alt

# %%
import _4_Model_kNN
 
 # %%
X_train, X_test, y_train, y_test = _4_Model_kNN.X_train, _4_Model_kNN.X_test, _4_Model_kNN.y_train, _4_Model_kNN.y_test
X_predict = _4_Model_kNN.X_predict

y_predict_explain, y_predict_dtc, y_predict_rfc = _4_Model_kNN.y_predict_explain, _4_Model_kNN.y_predict_dtc, _4_Model_kNN.y_predict_rfc

y_predict_explain_rfc = pd.DataFrame({
    "Prediction": y_predict_rfc, 
    "Confidence" : [True] * len(y_predict_rfc)
}, index=y_predict_explain.index)

# %%
i=1
_, fig = _4_Model_kNN.my_model.predict_explain(X_predict.iloc[[i]], True, y_plotly_colors)
fig.show()

# %%
fig.write_html(f"Html/Explanation_{index[i]}.html")
fig.write_image(f"Img/Explanation_{index[i]}.svg")


# %%
mask = y_predict_explain["Confidence"] == True
y_predict_explain[mask].head(50)
#%%
y_predict_explain.iloc[1, 2]

 # %%
index = [2553, 9342]

X_predict = X_predict.loc[index]
y_test = y_test.loc[index]
y_predict_explain = y_predict_explain.loc[index]
y_predict_explain_rfc = y_predict_explain_rfc.loc[index]
# %%

#%%
plotly_colors = plx.colors.qualitative.Plotly[1:]
y_plotly_colors = {y:plotly_colors[i] for i, y in enumerate(y_train.unique())}
# %%
import Objects.my_TSNE_PCA_Plot as plot_predictions

plot_predictions = plot_predictions.Plot_TSNE_PCA(X_train,y_train, y_plotly_colors)

# # %%
# fig = plot_predictions.TSNE()
# fig.write_html("Html/tsne_blank.html")
# fig.show()

# %%
fig = plot_predictions.TSNE(X_predict, y_predict_explain)
fig.write_html("Html/tsne_predictions.html")
fig.write_image("Img/tsne_predictions.svg")

fig.show()

#  %%
fig = plot_predictions.TSNE(X_predict, y_predict_explain, with_train_data=False)
fig.write_html("Html/tsne_predictions_solo.html")
fig.write_image("Img/tsne_predictions_solo.svg")
fig.show()

# %%



# %%

# # %%
# fig = plot_predictions.TSNE(X_predict, y_predict_explain_rfc)
# fig.write_html("Html/tsne_predictions_rfc.html")
# fig.show()

# # %%
# fig = plot_predictions.TSNE(X_predict, y_predict_explain_rfc, with_train_data=False)
# fig.write_html("Html/tsne_predictions_rfc_solo.html")
# fig.show()

# # %%
# fig = plot_predictions.PCA()
# fig.write_html("Html/pca_blank.html")
# fig.show()

# %%
fig = plot_predictions.PCA(X_predict, y_predict_explain)
fig.write_html("Html/pca_predictions.html")
fig.show()

# %%
fig = plot_predictions.PCA(X_predict, y_predict_explain, with_train_data=False)
fig.write_html("Html/pca_predictions_solo.html")
fig.show()

# # %%
# fig = plot_predictions.PCA(X_predict, y_predict_explain_rfc)
# fig.write_html("Html/pca_predictions_rfc.html")
# fig.show()

# # %%
# fig = plot_predictions.PCA(X_predict, y_predict_explain_rfc, with_train_data=False)
# fig.write_html("Html/pca_predictions_rfc_solo.html")
# fig.show()
# %%
import Objects.my_Confusion_Matrix as my_cm

y_pred= y_predict_explain["Prediction"]

# %%
my_conf = my_cm.my_confusion_matrix(y_test, y_pred, y_predict_explain["Confidence"])
my_conf_heat = my_conf[["not_recom_unsure", "not_recom_confident", "very_recom_unsure", "very_recom_confident", "priority_unsure",	"priority_confident", "spec_prior_unsure",	"spec_prior_confident"]].reindex(['not_recom', 'very_recom', 'priority', "spec_prior"])
# %%

plt.figure(figsize=(10,8))
sns.heatmap(my_conf_heat, annot=True, fmt=".1f",cmap="BuPu");
plt.xticks(rotation=45)
plt.ylim((y_test.nunique(), -0.5))
plt.yticks(rotation=45)
plt.show();
plt.savefig("Img/my_confusion_matrix.svg")
# %%

# %%
y_predict_explain
# %%


# %%

# %%
my_conf_heat
# %%
