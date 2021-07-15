# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import neighbors

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# %%
import _0_Objects
answer = _0_Objects.Explaining_Answer()

# %%
import _2_Preprocessing

X_train, X_test, y_train, y_test = _2_Preprocessing.X_train, _2_Preprocessing.X_test, _2_Preprocessing.y_train, _2_Preprocessing.y_test
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
X_train.shape, X_test.shape

# %%
from scipy import stats
from sklearn.neighbors._base import _get_weights
from sklearn.utils.extmath import weighted_mode

from sklearn.utils import check_array
from sklearn.utils.validation import _num_samples

class my_KNeighborsClassifier(KNeighborsClassifier):
    def __init__(self, **kwargs):
        self.model = KNeighborsClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_explain(self, X, confidence_threshold = 0.75):
        """Raise explaination of the prediction by reporting more results.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.

        """
        X = check_array(X, accept_sparse='csr')
        neigh_dist, neigh_ind = self.model.kneighbors(X)
        neigh_X = self.model._fit_X[neigh_ind]
        neigh_y = self.model._y[neigh_ind]
        neigh_classes=[self.model.classes_[i] for i in neigh_y]
        y_pred = self.model.predict(X)
        y_pred_prob = self.model.predict_proba(X)
        
        answer = pd.DataFrame({
            "Prediction" : y_pred,
             "Confidence" : (y_pred_prob.max(axis=1) >= confidence_threshold) & (y_pred == neigh_y[:,0]),
             "Explanation" : list(neigh_y)
            })
        return answer
        #return  y_pred, y_pred_prob.max(axis=1) >= 0.75, neigh_y

# %%
model = my_KNeighborsClassifier(n_neighbors=4)
model.fit(X_train, y_train)
answer = model.predict_explain(X_test.iloc[0:25])
answer


# %%
pd.DataFrame({"a" : [[13,4,2],[1,2,3],[3,4,3]] })
# %%
answer = [_0_Objects.Explaining_Answer(
                prediction=y_pred[i],
                confidence = y_pred_prob[i].max() >= 0.75,
                explanation = neigh_y[i]
            ) for i in range(X.shape[0])
        ]

for i, a in enumerate(answer):
    if 2 <= sum(a.Explanation) <= 3:
        print(a.to_list)

# %%
answer =[ _0_Objects.Explaining_Answer(
                prediction=2,
                confidence = True,
                explanation = [1,1,0,0]
            ) for i in range(2)]

for i, a in enumerate(answer):
    print(a.Explanation)

# %%

for i, neigh in enumerate(answer):
    print(answer.Prediction)





# %%
    if 2 <= sum(neigh.Explanation) <= 3:
        print(neigh)



# %%

y_pred_prob = model.predict_proba(X_test.iloc[0:25])
y_pred, neigh_y = model.predict_explain(X_test.iloc[0:25])

# %%

# %%




# Predict on test set
y_pred = model.predict(X_test)

# Print accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred).round(2))
print("-----"*10)

# Print confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='YlGn');
# %%

# %%
