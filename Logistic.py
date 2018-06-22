import numpy as np
from matplotlib import pyplot as plt
from sig import sigmoid
import math

X = np.array([2,3,4,5,10,12])
Y = np.array([0,0,0,0,1,1])

plt.scatter(X,Y, s = 50, color = 'red')
#plt.show() 
m = 1000
b = 0
Learningrate = 0.001
N = len(Y)
for i in range(1000):
	y = m*X + b

	m_gradient = (-2/N)*sum(X*(Y-y))
	b_gradient = (-2/N)*sum(Y-y)

	m = m - (m_gradient*Learningrate)
	b = b - (b_gradient*Learningrate)

#print(m,b)
Y_hat = m*X + b

plt.plot(X,Y_hat,color = 'yellow')
#plt.show()

P = sigmoid(Y_hat)

plt.plot(X,P,color = 'blue')

X_pred = 11

Y_Pred = m * X_pred + b
#print(Y_Pred)

Threshold = sigmoid(Y_Pred)

if Threshold > 0.5:
	Y_Pred = 1
	plt.scatter(X_pred,Y_Pred,color = 'black')
	print 'Belongs to 1 class'
elif Threshold < 0.5:
	Y_Pred = 0
	plt.scatter(X_pred,Y_Pred,color = 'green')
	print 'Belongs to 0 class'
plt.show()


