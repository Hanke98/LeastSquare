import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def GetIrisData():
    iris_data = datasets.load_iris()
    return train_test_split(iris_data.data, iris_data.target, test_size=0.33, random_state=1)

def GetDigitData():
    data = datasets.load_digits()
    return train_test_split(data['data'], data['target'], test_size=0.33, random_state=1)

def GetBreastCancerData():
    data = datasets.load_breast_cancer()
    return train_test_split(data['data'], data['target'], test_size=0.33, random_state=1)