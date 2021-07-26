from abc import ABC, abstractmethod

class Interface_predict_explain(ABC):

    @abstractmethod
    def predict_explain(self):
        pass