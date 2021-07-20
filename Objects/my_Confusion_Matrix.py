import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels

def my_confusion_matrix(y_actual, y_pred, confidence):
    _confusion_matrix = confusion_matrix(y_actual, y_pred)
    _labels = unique_labels(y_actual, y_pred).tolist()

    _cols = [str(y) + "_confident" if conf else str(y) + "_unsure" for y in _labels for conf in unique_labels(confidence).tolist()]
    
    _confident_values = confusion_matrix(y_actual[confidence], y_pred[confidence])
    _confident_labels = unique_labels(y_actual[confidence], y_pred[confidence]).tolist()
    confident_values =np.zeros(shape=_confusion_matrix.shape) 
    for row in _confident_labels:
        for col in _confident_labels:
            confident_values[_labels.index(row), _labels.index(col)] = _confident_values[_confident_labels.index(row), _confident_labels.index(col)]
        
    _insecure_values = confusion_matrix(y_actual[~confidence], y_pred[~confidence])
    _insecure_labels = unique_labels(y_actual[~confidence], y_pred[~confidence]).tolist()
    insecure_values = np.zeros(shape=_confusion_matrix.shape)
    for row in _insecure_labels:
        for col in _insecure_labels:
            insecure_values[_labels.index(row), _labels.index(col)] = _insecure_values[_insecure_labels.index(row), _insecure_labels.index(col)]
    
    _data = []
    for i in range(confident_values.shape[1]):
        _data +=[insecure_values[:,i], confident_values[:,i]] 
    
    return pd.DataFrame(np.array(_data).transpose(), index = _labels, columns=_cols)