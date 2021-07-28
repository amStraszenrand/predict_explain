
import numpy as np
import pandas as pd
import plotly.express as plx

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Plot_kNN():
    
    def __init__(self, X_train, y_train, tsne_learning_rate = 100, height= 600, width = 800):
        self.tsne_learning_rate = tsne_learning_rate
        self.X, self.X_train = X_train.copy(), X_train.copy()
        self.y, self.y_train = y_train.copy(), y_train.copy()
        self.height, self.width = height, width
        X = X_train.copy()
        self.y.name, self.y_train.name = y_train.name, y_train.name

    def TSNE(self, X_predict_explain, y_predict_explain):
        self.X_predict_explain = X_predict_explain
        self.y_predict_explain = y_predict_explain
        self._prepare_X_y()
        X_tsne = TSNE(learning_rate=self.tsne_learning_rate).fit_transform(self.X)
        
        fig = self._plot(X_tsne)
        fig.update_layout(height=self.height, width=self.width, title_text="Dimensionality reduction for y_predict_explain: TSNE visualization")
        
        return fig
    
    def PCA(self, X_predict_explain, y_predict_explain):
        self.X_predict_explain = X_predict_explain
        self.y_predict_explain = y_predict_explain
        self._prepare_X_y()
        X_pca = PCA().fit_transform(self.X)
        
        fig = self._plot(X_pca)
        fig.update_layout(height=self.height, width=self.width, title_text="Dimensionality reduction for y_predict_explain: PCA visualization")
        
        return fig

    def _prepare_X_y(self):
        for i in self.y_predict_explain.index:
            self.X.loc[i] = self.X_predict_explain.loc[i]
            self.y.loc[i] = self.y_predict_explain.loc[i, "Prediction"]
            
    def _plot(self, X_plot):
        fig = make_subplots(rows=1, cols=1)
        plotly_colors = plx.colors.qualitative.Plotly

        mask_train = self.y.index.isin(self.y_train.index)        
        X_train_plot = X_plot[mask_train]
        
        _costumdata = self.X.copy()
        _costumdata["ID"] = _costumdata.index
        
        _hovertemplate = ' '.join([f'{col}: ' + '%{customdata[' + str(i) + ']}<br>' for i, col in enumerate(_costumdata)])
        
        df_temp = pd.concat([pd.DataFrame(X_train_plot, index=self.y_train.index), self.y_train], axis=1)
        for y_pred_expl_grouped, X_tsne_grouped in df_temp.groupby(self.y_train.name):
            fig.add_trace(
                go.Scatter(x=X_tsne_grouped[0], y=X_tsne_grouped[1], 
                           name=y_pred_expl_grouped, 
                           mode="markers",
                            marker={
                                "color" : plotly_colors[self.y_train.unique().tolist().index(y_pred_expl_grouped)]
                                },
                            customdata=_costumdata.loc[X_tsne_grouped.index],
                            hovertemplate=_hovertemplate
                            ),
                row=1, col=1
            )
            
            
        X_predict_explain_plot = X_plot[~mask_train]
        
        df_temp = pd.concat([pd.DataFrame(X_predict_explain_plot, index=self.y_predict_explain.index), self.y_predict_explain], axis=1)
        for y_pred_expl_grouped, X_tsne_grouped in df_temp.groupby(["Prediction", "Confidence"]):
            if y_pred_expl_grouped[1]:
                fig.add_trace(
                    go.Scatter(x=X_tsne_grouped[0], y=X_tsne_grouped[1], 
                            name=str(y_pred_expl_grouped[0]) + " (predicted)",
                            mode="markers",
                            marker={
                                "line" : {"width" : 2},
                                "color" : plotly_colors[self.y_train.unique().tolist().index(y_pred_expl_grouped[0])]
                                },
                            customdata=_costumdata.loc[X_tsne_grouped.index],
                            hovertemplate=_hovertemplate
                            ),
                    row=1, col=1
                )        
            else:
                fig.add_trace(
                    go.Scatter(x=X_tsne_grouped[0], y=X_tsne_grouped[1], 
                            name=str(y_pred_expl_grouped[0]) + " (predicted, but unsure)",
                            mode="markers",
                            marker={
                                "symbol" : "diamond",
                                "line" : {"width" : 2},
                                "color" : plotly_colors[self.y_predict_explain["Prediction"].unique().tolist().index(y_pred_expl_grouped[0])]
                                },
                            customdata=_costumdata.loc[X_tsne_grouped.index],
                            hovertemplate=_hovertemplate
                            ),
                    row=1, col=1
                )

        return fig