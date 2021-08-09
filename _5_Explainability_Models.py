# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as plx

# %%
import _2_Preprocessing

df_name = "nursery"

df, X_train, X_test, y_train, y_test =_2_Preprocessing.get_train_test_split(df_name)
X_train
# %%

import _4_Model_kNN

my_model = _4_Model_kNN.my_model
logreg=_4_Model_kNN.logreg
dtc = _4_Model_kNN.dtc
rfc =_4_Model_kNN.rfc

# %%
#
# ----- Get importance of features -----
#

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(X_train, y_train)

eli5.show_weights(perm, feature_names = X_train.columns.tolist())
 # %%
most_important_features_values = perm.feature_importances_[np.argsort(perm.feature_importances_)][-5:]
most_important_features = X_train.columns[np.isin(perm.feature_importances_, most_important_features_values)]


# %%
perm = PermutationImportance(rfc, random_state=1).fit(X_train, y_train)

eli5.show_weights(perm, feature_names = X_train.columns.tolist())

# %%

#
# ----- Get partial effects of features -----
#

X_train
# %%

from pdpbox import pdp, get_dataset, info_plots

most_important_features = ["health_priority", "health_recommended", "parents_pretentious", "parents_usual", "has_nurs_improper",	"has_nurs_less_proper",	"has_nurs_proper",	"has_nurs_very_crit"]

for feature_name in most_important_features[:1]:
    pdp_dist = pdp.pdp_isolate(model=my_model, dataset=X_test, model_features=X_test.columns.tolist(), feature=feature_name)

    pdp.pdp_plot(pdp_dist, feature_name)
    plt.xlabel([1,2,3,4])


plt.show()


# %%


from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

# %%


pip install sklearn.ensemble.partial_dependence

# %%
sample_data_for_prediction = X_test.iloc[0].astype(float)  # to test function

def shap_single_explainer(model, X_predict, X_backgroundt):
    # Create object that can calculate shap values
    explainer = shap.KernelExplainer(model.predict_proba, X_backgroundt)
    shap_values = explainer.shap_values(X_predict)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], X_predict)


# %%

#
# ----- Get SHAPE explanation for some y -----
#
# %%
y_test[y_test == "spec_prior"]
# %%
[y_test[y_test == y].index[0] for y in y_test.value_counts().index]
# %%
i = 11300
print(y_test.loc[i])
shap_single_explainer(my_model, X_test.loc[i], X100)
# %%
i = 6780
print(y_test.loc[i])
shap_single_explainer(my_model, X_test.loc[i], X100)
# %%
i = 8772
print(y_test.loc[i])
shap_single_explainer(my_model, X_test.loc[i], X100)

# %%
i = 9042
print(y_test.loc[i])
shap_single_explainer(my_model, X_test.loc[i], X100)
# %%
i = 8097
print(y_test.loc[i])
shap_single_explainer(my_model, X_test.loc[i], X100)
# %%
i = 5466
print(y_test.loc[i])
shap_single_explainer(my_model, X_test.loc[i], X100)

# %%

#
# ----- Get feedback amplitude between two features -----
#

shap_values[1]
# %%
for feature_name in most_important_features:
    shap.dependence_plot(feature_name, shap_values[1], X_plot)

# %%

#
# ----- Get LIME explanation for some y -----
#

import lime, lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train), feature_names=X_train.columns.to_list(), class_names=y_test.value_counts().index.to_list()
, discretize_continuous=True)


# %%
[y_test[y_test == y].index[0] for y in y_test.value_counts().index]

# %%
i = 2449
print(y_test.loc[i])

exp = explainer.explain_instance(X_test.loc[i], my_model.predict_proba, top_labels=1)
exp.show_in_notebook( show_table=True, show_all=True)
# %%
exp.save_to_file(f"Html/lime_explanation_{i}.html")
# %%
X_predict=X_test

y_predict_explain =my_model.predict_explain(X_predict)


# %%

mask =(y_predict_explain["Prediction"] == "very_recom") & (y_predict_explain["Confidence"] == False)
y_predict_explain[mask].to_csv("here.csv")


# %%
y_predict_explain.loc[7668, "Explanation"]





# %%
i = 6780
print(y_test.loc[i])

exp = explainer.explain_instance(X_test.loc[i], my_model.predict_proba, top_labels=1)
exp.show_in_notebook(show_table=False, show_all=True)
# %%
i = 7
print(y_test.loc[i])

exp = explainer.explain_instance(X_test.loc[i], my_model.predict_proba, top_labels=1)
exp.show_in_notebook(show_table=False, show_all=True)