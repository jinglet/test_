from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

iris = load_iris()

iris_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'])
iris_data['target'] = iris_data['target'].map({0:'setosa', 1:'versicolor', 2:'virginica'})

x_data = iris_data.iloc[:, :-1]
y_data = iris_data.iloc[:, [-1]]