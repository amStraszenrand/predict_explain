# %%
import numpy as np
from numpy.core import numeric
import pandas as pd
from matplotlib import pyplot as plt
from pandas_profiling.config import Interactions
import seaborn as sns
import plotly
import plotly.express as plx
import altair as alt


# %%
import _2_Preprocessing

df_name = "nursery"

df, X_train, X_test, y_train, y_test =_2_Preprocessing.get_train_test_split(df_name)

# %%
plotly_colors = plx.colors.qualitative.Plotly[1:]
y_plotly_colors = {y:plotly_colors[i] for i, y in enumerate(y_train.unique())}

# %%
feature ="parents"
fig = plx.histogram(df, x = feature, color="class", color_discrete_map=y_plotly_colors)
fig.write_html(f"Html/histogram_{feature}.html")
plotly.io.write_image(fig, f"Img/histogram_{feature}.svg")
fig.show()

# %%
feature ="has_nurs"
fig = plx.histogram(df, x = feature, color="class", color_discrete_map=y_plotly_colors)
fig.write_html(f"Html/histogram_{feature}.html")
plotly.io.write_image(fig, f"Img/histogram_{feature}.svg")
fig.show()

# %%
feature ="form"
fig = plx.histogram(df, x = feature, color="class", color_discrete_map=y_plotly_colors)
fig.write_html(f"Html/histogram_{feature}.html")
plotly.io.write_image(fig, f"Img/histogram_{feature}.svg")
fig.show()

# %%
feature ="children"
fig = plx.histogram(df, x = feature, color="class", color_discrete_map=y_plotly_colors)
fig.write_html(f"Html/histogram_{feature}.html")
plotly.io.write_image(fig, f"Img/histogram_{feature}.svg")
fig.show()

# %%
feature ="housing"
fig = plx.histogram(df, x = feature, color="class", color_discrete_map=y_plotly_colors)
fig.write_html(f"Html/histogram_{feature}.html")
plotly.io.write_image(fig, f"Img/histogram_{feature}.svg")
fig.show()

# %%
feature ="finance"
fig = plx.histogram(df, x = feature, color="class", color_discrete_map=y_plotly_colors)
fig.write_html(f"Html/histogram_{feature}.html")
plotly.io.write_image(fig, f"Img/histogram_{feature}.svg")
fig.show()

# %%
feature ="social"
fig = plx.histogram(df, x = feature, color="class", color_discrete_map=y_plotly_colors)
fig.write_html(f"Html/histogram_{feature}.html")
plotly.io.write_image(fig, f"Img/histogram_{feature}.svg")
fig.show()

# %%
feature ="health"

fig = plx.histogram(df, x = feature, color="class", color_discrete_map=y_plotly_colors)
fig.write_html(f"Html/histogram_{feature}.html")
plotly.io.write_image(fig, f"Img/histogram_{feature}.svg")
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
