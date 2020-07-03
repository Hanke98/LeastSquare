from abc import ABCMeta, abstractmethod
import numpy as np


class Approximator(metaclass=ABCMeta):

    NAME = 'Base'
    def __init__(self):
        pass
    
    def fit(self, X_train, y_train):
        assert len(X_train) == len(y_train)
        loss = self._run_fit(X_train, y_train)
        print('training: ')
        self._print_data(loss)

    def test(self, X_test, y_test):
        assert len(X_test) == len(y_test)
        loss = self._run_test(X_test, y_test)
        print('testing: ')
        self._print_data(loss)
        pass

    def predict(self, X_test):
        return self._run_predict(X_test)

    def _print_data(self, loss):
        print('loss: {}'.format(loss))
    
    @abstractmethod
    def _run_fit(self, X_train, y_train):
        pass

    @abstractmethod
    def _run_test(self, X_test, y_test):
        pass

    @abstractmethod
    def _run_predict(self, X_test):
        pass