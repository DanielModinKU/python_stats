#Script for fitting logistic regression from scratch 

import numpy 
import math as m 
from sklearn import datasets #load iris data
import matplotlib.pyplot as plt

def load_data():
	iris = datasets.load_iris()
	X = iris.data[:,0]
	Y = iris.target
	return X,Y

def predict(X,B): #calculate linear prediction (ln(odds)) given data matrix X and parameter vector B
	log_odds = X.dot(B)
	return log_odds

def neg_LL(X,B,Y): # calculate negative log likelihood 
	
	#convert predicted ln(odds) to probability of Y = 1 
	log_odds = predict(X,B)
	p = numpy.exp(log_odds)/(1+numpy.exp(log_odds))

	#calculate negative log likelihood
	neg_LL = numpy.sum( (-1)*(Y * numpy.log(p) + (1-Y) * numpy.log(1-p)) )

	return(neg_LL)

def GDA(X,B,Y,alfa):			#kør gradient descent til at opdatere parametre 
	#gem gamle parametre
	B_old = B 

	#convert predicted ln(odds) to probability of Y = 1 
	log_odds = predict(X,B_old)
	p = numpy.exp(log_odds)/(1+numpy.exp(log_odds))

	#update parameters 
	for i in (range(len(B))):
		#print(len((Y - p)*X[:,i]))
		#print(X[:,i])
		B[i] = B_old[i] - alfa*numpy.sum( (Y - p)*X[:,i] )

	return(B) 

def train(X,B,Y,epoch,alfa):

	epochs = []
	neg_LogL = [] 

	for i in range(epoch):

		print('training example nr::::: '+str(i))

		B = GDA(X,B,Y,alfa)

		print(B)

		negLL = neg_LL(X,B,Y)

		epochs.append(i)
		neg_LogL.append(negLL)

	plt.plot(epochs,neg_LogL)
	plt.show()



#load data in numpy arrays 
X,Y = load_data() 

#add column of 1s to account for b0 intercept parameter 
ones = numpy.ones(X.shape[0])
X = numpy.column_stack((ones,X))

#intialise beta parameters (not important to randomly initialize, since problem is convex, so no problem with varying local minima depending on intialization as in ex NNs)
B = numpy.ones(X.shape[1]) 


train(X,B,Y,4000,0.0001)
















