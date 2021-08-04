# %%
import numpy as np
from numpy.core import numeric
import pandas as pd
from matplotlib import pyplot as plt
from pandas_profiling.config import Interactions
import seaborn as sns
import plotly.express as plx
import altair as alt


# %%
import _2_Preprocessing

df_name = "nursery"

df, X_train, X_test, y_train, y_test =_2_Preprocessing.get_train_test_split(df_name)

# %%
fig = plx.histogram(df, x = "parents", color="class")
fig.show()

# %%
fig = plx.histogram(df, x = "has_nurs", color="class")
fig.show()

# %%
fig = plx.histogram(df, x = "form", color="class")
fig.show()

# %%
fig = plx.histogram(df, x = "children", color="class")
fig.show()

# %%
fig = plx.histogram(df, x = "housing", color="class")
fig.show()

# %%
fig = plx.histogram(df, x = "finance", color="class")
fig.show()

# %%
fig = plx.histogram(df, x = "social", color="class")
fig.show()

# %%
fig = plx.histogram(df, x = "health", color="class")
fig.show()

# %%




# %%
from pandas_profiling import ProfileReport


profile = ProfileReport(df, 
                        title="Pandas Profiling Report",
                        explorative=True,
                        interactions={
                            "continuous" : True
                        }
)
profile.to_file("Html/pandas_report.html")


# %%
import sweetviz

df["class"] = df["class"].map({"not_recom": 0., "priority": 2., "spec_prior": 2., "very_recom": 2., "recommend" : 2.}).astype(bool)

report = sweetviz.analyze(df, "class")
report.show_html()

# %%
