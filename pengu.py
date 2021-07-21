import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from CostFuncAndGradient import *

data = pd.read_csv("./pengu.csv")
data = data[['Fancy Words', 'Distance']]


matrix = np.array(data.values, "float")

X = matrix[:, 0]
y = matrix[:, 1]

X = X / (np.max(X))



m = np.size(y)
X = X.reshape([len(X), 1])
x = np.hstack([np.ones_like(X), X])

theta = np.zeros([2, 1])



theta, J = gradient(x, y, theta, m)


plt.figure(num='Fancy Words Used VS Distance Thrown')

plt.plot(X, y, 'c', marker=",")

plt.plot(X, x @ theta, '-')

plt.ylabel("Distance Thrown")

plt.xlabel('Fancy Words Used')



plt.title('Fancy Words VS Distance Thrown')

plt.show()  

print("Final Cost: " + str(computeCost(x, y, theta, m)))  # The final cost is 756.4073367032673

