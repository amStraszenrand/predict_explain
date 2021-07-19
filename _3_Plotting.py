# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as plx
import altair as alt


# %%
import _2_Preprocessing

X_train, X_test, y_train, y_test = _2_Preprocessing.X_train, _2_Preprocessing.X_test, _2_Preprocessing.y_train, _2_Preprocessing.y_test

# %%
from sklearn.manifold import TSNE

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


X_tsne = TSNE(learning_rate=100).fit_transform(X_train)
X_pca = PCA().fit_transform(X_train)

# %%




# %%
firstPlot = alt.Chart(X_tsne).mark_circle().encode(
    x = X_tsne
)
# secondPlot = alt.Chart(X_tsne).mark_circle().encode(
#     x = alt.X(X_tsne[:, 0]),
#     y = alt.Y(X_tsne[:, 1]),
#     color = y_train
# )
firstPlot + secondPlot


# %%

    print(x)
# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=1, cols=1)

df_temp = pd.concat([pd.DataFrame(X_tsne, index=y_train.index), y_train], axis=1)
for y_train_grouped, X_tsne_grouped in df_temp.groupby("species"):
    fig.add_trace(
        #go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
        go.Scatter(x=X_tsne_grouped[0], y=X_tsne_grouped[1], name=y_train_grouped, mode="markers"),
        row=1, col=1
    )
    
fig.add_trace(
        #go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
        go.Scatter(x=[0], y=[10], name="X_test", mode="markers"),
        row=1, col=1
    )
    

# fig.add_trace(
#     #go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
#     go.Scatter(x=X_test[:, 0], y=X_tsne[:, 1]),
#     row=1, col=1
# )

fig.update_layout(height=400, width=300, title_text="Side By Side Subplots")
fig.show()



# %%
fig = plx.scatter(X_pca[:, 0], X_pca[:, 1], color=y_train);
fig_go = go.Figure()
#fig.show()
fig_go = go.FigureWidget(fig)
fig_go.show()


# %%

from plotly import express as px
import numpy as np
import plotly.graph_objects as go

fig = px.imshow(np.random.rand(3,3))
fig.update_layout(coloraxis_showscale=False)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig = go.FigureWidget(fig)

# %%
