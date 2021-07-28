# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as plx

# %%
import _2_Preprocessing

X_train, X_test, y_train, y_test = _2_Preprocessing.X_train, _2_Preprocessing.X_test, _2_Preprocessing.y_train, _2_Preprocessing.y_test

# %%

import _4_Model_kNN

knn = _4_Model_kNN.knn
dtc = _4_Model_kNN.dtc
my_model = _4_Model_kNN.my_model

# %%
#
# ----- Get importance of features -----
#

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(X_test, y_test)
most_important_features_values = perm.feature_importances_[np.argsort(perm.feature_importances_)][-2:]
most_important_features = X_test.columns[np.isin(perm.feature_importances_, most_important_features_values)]

eli5.show_weights(perm, feature_names = X_test.columns.tolist())

# %%

#
# ----- Get direction + amplitude of features -----
#

import shap  # package used to calculate Shap values

explainer = shap.KernelExplainer(my_model.predict_proba, X_train)
X_plot = X_train.head(10)
shap_values = explainer.shap_values(X_plot)

shap.summary_plot(shap_values[1], X_plot)

# %%

#
# ----- Get partial effects of features -----
#

from pdpbox import pdp, get_dataset, info_plots

for feature_name in most_important_features:
    pdp_dist = pdp.pdp_isolate(model=my_model, dataset=X_test, model_features=X_test.columns.tolist(), feature=feature_name)

    pdp.pdp_plot(pdp_dist, feature_name)
plt.show()

# %%
sample_data_for_prediction = X_test.iloc[0].astype(float)  # to test function

def patient_risk_factors(model, patient_data):
    # Create object that can calculate shap values
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = explainer.shap_values(patient_data)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient_data)


# %%

#
# ----- Get SHAPE explanation for some y -----
#

[y_test[y_test == y].index[0] for y in y_test.value_counts().index]
# %%
i = 93
print(y_test.loc[i])
patient_risk_factors(my_model, X_test.loc[i])
# %%
i = 113
print(y_test.loc[i])
patient_risk_factors(my_model, X_test.loc[i])
# %%
i = 7
print(y_test.loc[i])
patient_risk_factors(my_model, X_test.loc[i])

# %%

#
# ----- Get feedback amplitude between two features -----
#

for feature_name in most_important_features:
    shap.dependence_plot(feature_name, shap_values[1], X_plot)

# %%

#
# ----- Get LIME explanation for some y -----
#

import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train), feature_names=X_train.columns.to_list(), class_names=y_test.value_counts().index.to_list()
, discretize_continuous=True)


# %%
[y_test[y_test == y].index[0] for y in y_test.value_counts().index]

# %%
i = 93
print(y_test.loc[i])

exp = explainer.explain_instance(X_test.loc[i], my_model.predict_proba, top_labels=1)
exp.show_in_notebook(show_table=False, show_all=True)
# %%
i = 113
print(y_test.loc[i])

exp = explainer.explain_instance(X_test.loc[i], my_model.predict_proba, top_labels=1)
exp.show_in_notebook(show_table=False, show_all=True)
# %%
i = 7
print(y_test.loc[i])

exp = explainer.explain_instance(X_test.loc[i], my_model.predict_proba, top_labels=1)
exp.show_in_notebook(show_table=False, show_all=True)
# %%

#
# ----- Get alepython explanation for some y -----
#
from alepython import ale_plot

my_model.fit(X_train, y_train.map({"Iris-versicolor": 0, "Iris-setosa":1, "Iris-virginica":2}))

plt.rc("figure", figsize=(9, 6))
for feature_name in most_important_features:
    ale_plot(
        my_model,
        X_train,
        feature_name,
        bins=20,
        monte_carlo=True,
        monte_carlo_rep=100,
        monte_carlo_ratio=0.6,
    )
# %%


# %%

#
# ----- Get ML.INTERPRET explanation for some y -----
#

from interpret.glassbox import ExplainableBoostingClassifier
ebm = ExplainableBoostingClassifier(random_state=17)

X_interpret = X_train.iloc[:10]
y_interpret = y_train.iloc[:10]

ebm.fit(X_interpret, y_interpret)
# %%
# %%
