def sigmoid(data):
	import numpy as np

	z = 1/(1+np.exp(-data))

	#print(z)

	return z