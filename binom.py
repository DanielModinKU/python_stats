#try to plot bernoulli dsitribution
import math 
import matplotlib.pyplot as plt

def binom(n, p):

	list_x = []
	list_y = []

	for i in range(n+1):
		print(i)
		list_x.append(i)
		PX = ( math.factorial(n)/(math.factorial(i)*math.factorial(n-i)) ) * (p**i) * ((1-p)**(n-i))
		list_y.append(PX)

	return list_x, list_y 


x, y = binom(100,0.01)

plt.plot(x,y, 'ro')
plt.show()