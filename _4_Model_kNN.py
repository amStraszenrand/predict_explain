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

# %%

# %%
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors._base import _get_weights
from sklearn.utils.extmath import weighted_mode

from sklearn.utils import check_array
from sklearn.utils.validation import _num_samples

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler



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
        
        try:
            feature_names = X.columns
        except:
            feature_names = None
            
            
        X = check_array(X, accept_sparse='csr')
        neigh_dist, neigh_ind = self.model.kneighbors(X)
        
        
        neigh_y = self.model._y[neigh_ind]
        neigh_classes=[self.model.classes_[i] for i in neigh_y]
        
        y_pred = self.model.predict(X)
        y_pred_prob = self.model.predict_proba(X)
        
        prediction = y_pred
        confidence = (y_pred_prob.max(axis=1) >= confidence_threshold) & (y_pred == neigh_y[:,0])
        
        
        if feature_names.any(): 
            A = self.model._fit_X[neigh_ind]
            
            neigh_X = [pd.DataFrame(a, columns=feature_names) for a in A]
            explanation = self._write_explanation(X, neigh_y, neigh_X)
        else:
            explanation = self._write_explanation(X, neigh_y)
        
        
        answer = pd.DataFrame({
            "Prediction" : prediction,
             "Confidence" : confidence,
             "Explanation" : explanation,
             "Distance" : list(np.round(neigh_dist, decimals=2))
            })
        return answer

    
    def _write_explanation(self, x, neigh_y, neigh_X=None):
        
        # for n in neigh_X:
        #     print(n)
        # testo = "There "
        
        # for i, n in enumerate(neigh_X):
                
            
            
        #     scaler = MinMaxScaler()
        #     X_transi = np.round(scaler.fit_transform(n),2)
        #     print("X_transi")  
        #     print(X_transi)
        #     normy = np.round(scaler.transform(X[[i]]),2)
        #     print("Normy")
        #     print(normy)
        #     print(X_transi - normy)
        return "long explanation"
        #return  y_pred, y_pred_prob.max(axis=1) >= 0.75, neigh_y

# %%
model = my_KNeighborsClassifier(n_neighbors=4, weights="uniform")
model.fit(X_train, y_train)
A=X_test.iloc[9:10]
print(A)
answer = model.predict_explain(A)
y_pred = answer["Prediction"]
answer

# %%
pd.DataFrame({"a" : [[13,4,2],[1,2,3],[3,4,3]] })

# %%
A = np.array([[1.2344522, 1.543323455], [2.4349585473483, 4.2823832783283]])
list(np.round(A,2))


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
from sklearn.preprocessing import Normalizer
from numpy import linalg as LA
# X = [[4, 1, 2, 2],
#      [1, 3, 9, 3],
#      [5, 7, 5, 1]]

X  = [[1,2],
     [1,3],
     [1,4]]

transformer = Normalizer().fit(X)  # fit does nothing.
transformer

normy = transformer.transform(X)
normy, LA.norm(normy, axis=1)
# %%
