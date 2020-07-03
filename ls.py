import numpy as np
from Approximator import Approximator
from DataGen import GetIrisData, GetDigitData, GetBreastCancerData


class LeastSquareApproximator(Approximator):
    
    NAME = 'LeastSqure'
    
    def __init__(self):
        super().__init__()
        self.c = []
    
    def _run_fit(self, X_train, y_train):
        H = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T)
        self.c = H.dot(y_train)
        y_pred = X_train.dot(self.c)
        loss = np.linalg.norm(y_pred - y_train)
        return loss
    
    def _run_test(self, X_test, y_test):
        y_pred = X_test.dot(self.c)
        loss = np.linalg.norm(y_pred - y_test) 
        return loss
    
    def _run_predict(self, X_test):
        y_pred = X_test.dot(self.c)
        return y_pred
        

# X_train, X_test, y_train, y_test = GetIrisData()

X_train, X_test, y_train, y_test = GetBreastCancerData()
# X_train = (X_train - X_train.mean()) / X_train.std()
# X_test = (X_test - X_test.mean()) / X_test.std()
# y_train = (y_train - y_train.mean()) / y_train.std()
# y_test = (y_test - y_test.mean()) / y_test.std()

ls_appr = LeastSquareApproximator()
ls_appr.fit(X_train, y_train)
y_pred = ls_appr.predict(X_test)
y_pred = (y_pred + 0.5).astype(np.int)
acc = 1 - np.abs(y_pred - y_test).sum() / len(y_test)
print("acc: {}".format(acc))
