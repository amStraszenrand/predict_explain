import numpy as np
import pandas as pd

from scipy import stats

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors._base import _get_weights
from sklearn.utils.extmath import weighted_mode

from sklearn.utils import check_array
from sklearn.utils.validation import _num_samples

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

# %%
class my_KNeighborsClassifier(KNeighborsClassifier):
    def __init__(self,  my_predict_proba_threshold = 0.75, my_neigh_dist = None, **kwargs):
        self.model = KNeighborsClassifier(**kwargs)
        self.my_predict_proba_threshold = my_predict_proba_threshold
        self.my_neigh_dist = my_neigh_dist
        self.my_neigh_dist_median_frac = 0.5

    def fit(self, X, y):
        self.model.fit(X, y)
        
        if not self.my_neigh_dist:
            all_distances, _ =  self.kneighbors(X, n_neighbors=self.model.n_neighbors + 1, return_distance=True)
            all_not_self_referred_distances = [dist for sublist in all_distances for dist in sublist[1:]]
            self.my_neigh_dist = np.median(all_not_self_referred_distances) * self.my_neigh_dist_median_frac

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        return self.model.kneighbors(X, n_neighbors, return_distance)

    def predict_explain(self, X):
        """Raise explaination of the prediction by reporting more results.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y_predict_explain : dict shape (n_queries,4) with \
            "Prediction" : predicted target value, \
             "Confidence" : confidence of the prediction, \
             "Explanation" : explanation of the prediction, \
             "Features_Distribution": informations about the features distribution

        """
        
        try:
            feature_names = X.columns
            index_column = X.index
        except:
            feature_names = None
            
        X = check_array(X, accept_sparse='csr')
        
        neigh_dist, neigh_ind = self.kneighbors(X)
        neigh_y = self.model._y[neigh_ind]
        neigh_classes=[self.model.classes_[i] for i in neigh_y]

        y_pred = self.model.predict(X)
        y_pred_prob = self.model.predict_proba(X)

        prediction = y_pred
        
        confidence_threshold = y_pred_prob.max(axis=1) >= self.my_predict_proba_threshold
        confidence_nearestNeighbour = y_pred == [neigh_class[0] for neigh_class in neigh_classes]
        confidence = confidence_threshold & confidence_nearestNeighbour
        
        explanation = self._write_explanation(X, prediction, confidence_threshold, confidence_nearestNeighbour, neigh_classes)
        
        features_distribution = ""
        neigh_X = []
        if feature_names.any(): 
            A = self.model._fit_X[neigh_ind]
            neigh_X = [pd.DataFrame(a, columns=feature_names) for a in A]
        
        features_distribution = self._write_features_distribution(X, neigh_dist, neigh_X, feature_names)
        
        answer = pd.DataFrame({
            "Prediction" : prediction,
             "Confidence" : confidence,
             "Explanation" : explanation,
             "Features_Distribution": features_distribution
            }, index = index_column)
        
        return answer

    
    def _write_explanation(self, X,  prediction, conf_thresh, conf_nearNeigh, neigh_classes):
        
        explanation = []
        
        for i, x in enumerate(X):
            expl = f"The prediction '{prediction[i]}' is "
            expl += "quite sure: " if conf_thresh[i] & conf_nearNeigh[i] else "rather unsure: "
            
            expl += f"On one hand the {self.model.get_params()['n_neighbors']} nearest neighbours have "
            expl += "homogeneous " if conf_thresh[i] else "diverse "
            expl += f"target values ("
            class_values, class_counts = np.unique(neigh_classes[i], return_counts=True)
            for value, count in zip(class_values, class_counts):
                expl += f"{count}x value '{value}', "
            expl = expl[:-2]
            expl += "). "
            
            expl += "But " if conf_thresh[i] != conf_nearNeigh[i] else "And "
            expl += "on the other hand the nearest neighbour has "
            expl += "the same target value too." if conf_nearNeigh[i] else f"another target value ('{neigh_classes[i][0]}') as the prediction."
            
            explanation.append(expl)        
        
        return explanation

    def _write_features_distribution(self, X, neigh_dist, neigh_X, feature_names):
        
        features_distribution = []
        
        for i, x in enumerate(X):
            exact_matches = sum(neigh_dist[i] == 0)
            nearby_distances = sum(neigh_dist[i] <= self.my_neigh_dist)
            
            feat = f"The features given for predicting the target value "
            if nearby_distances == 1:
                feat += "are quite near to another observation "
            elif nearby_distances > 1:
                feat += f"are quite near to {nearby_distances} other observations "
            else:
                 feat += "are rather far from any other observations "
            feat += "already known"
            
            if exact_matches:
                feat += " (they even map exactly on the "
                feat += f"nearest neighbor" if exact_matches == 1 else f"{exact_matches} nearest neighbors). "
            else:
                feat += ". "
           
            
            if neigh_X == []:
                feat += "Specific informations about the features however cannot be given."
            else:
                scaler = MinMaxScaler()
                neigh_X_scaled = np.round(scaler.fit_transform(neigh_X[i]),2)
                x_scaled = np.round(scaler.transform(X[[i]]),2)
                metrics_distance = abs(neigh_X_scaled - x_scaled)
                
                if nearby_distances == 1:
                    metrics_distance_nearby = metrics_distance[0]
                    same_features= feature_names[np.where(metrics_distance_nearby == 0.)].values
                    diff_feature_ind = np.where(metrics_distance == np.max(metrics_distance))
                elif nearby_distances >= 1:
                    metrics_distance_nearby = metrics_distance[:nearby_distances]
                    same_features = [set(feature_names[np.where(metrics_dist == 0.)]) for metrics_dist in metrics_distance_nearby]
                    same_features = list(same_features[0].intersection(*same_features))
                    diff_feature_ind = np.where(metrics_distance == np.max(metrics_distance))
                else:
                    same_features = [set(feature_names[np.where(metrics_dist == 0.)]) for metrics_dist in metrics_distance]
                    same_features = list(same_features[0].intersection(*same_features))
                    diff_feature_ind = np.where(metrics_distance == np.max(metrics_distance))

                r = np.random.randint(len(diff_feature_ind[-1]))
                diff_feature_ind = [ind[r] for ind in diff_feature_ind]
            
                if len(same_features) == 0:
                    feat += f"No feature has "
                elif len(same_features) == 1:
                    feat += f"The feature {same_features} has "
                else:
                    feat += f"The features {same_features} have " 
                
                feat += f"the exact same values "
                if nearby_distances == 1:
                    feat += f"as this {nearby_distances} nearest neighbour has. "
                elif nearby_distances > 1:
                    feat += f"as those {nearby_distances} nearest neighbours have. "
                else:
                    feat += f"in the range of the {self.model.get_params()['n_neighbors']} nearest neighbours. "
                
                if  np.max(metrics_distance) == 0:
                    feat += f"Therefore no feature differs from any of the {self.model.get_params()['n_neighbors']} nearest neighbours!"
                else:
                    feat += f"However, the feature "
                    feat += f"'{feature_names[diff_feature_ind[-1]]}' differs at most "
                    feat += f"('{x[diff_feature_ind[-1]]}' "
                    feat += f"vs. '{float(neigh_X[i].iloc[0,diff_feature_ind[-1]])}')." if len(diff_feature_ind) == 1 else f"vs. '{neigh_X[i].iloc[diff_feature_ind[0], diff_feature_ind[1]]}') "
                    feat += f"throughout the inspected  {self.model.get_params()['n_neighbors']} nearest neighbours."
            
            features_distribution.append(feat)        
        
        return features_distribution