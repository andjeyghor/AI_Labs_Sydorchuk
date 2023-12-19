import numpy as np
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.7 * X ** 2 + X + 2 + np.random.randn(m, 1)

import matplotlib.pyplot as plt

plt.scatter(X,y)
plt.show()

X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(theta_best)

X_new     = np.array([[0],[1],[2]])
X_new_b   = np.c_[np.ones((3, 1)), X_new] # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
print(y_predict)

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()



lin_reg.fit(X,y)
print("intercept & coefficient:\n", lin_reg.intercept_, lin_reg.coef_)
print("predictions:\n", lin_reg.predict(X_new))


