#Script for fitting logistic regression from scratch 

import numpy 
import math as m 
from sklearn import datasets #load iris data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score #til at udregne accuracy 


def predict(X,B): #calculate linear prediction (ln(odds)) given data matrix X and parameter vector B
	#linear ln(odds) prediction
	log_odds = X.dot(B)
	#convert to probability
	p = 1/(1+numpy.exp(-log_odds))
	return p

def neg_LL(X,B,Y): # calculate negative log likelihood
	#calculate current prob prediction 
	p = predict(X,B)
	#small epsilon to avoid log(0) 
	epsilon = 0.0000000001
	#calculate negative log likelihood
	neg_LL =  (-1)*numpy.sum(Y*numpy.log(p+epsilon) + (1-Y)*numpy.log(1-p+epsilon))
	return(neg_LL)

def GDA(X,B,Y,alfa):			#kør gradient descent til at opdatere parametre 
	#gem gamle parametre
	B_old = B 

	#m 
	m = X.shape[0]

	#calculate probability prediction
	p = predict(X,B_old)
	#update parameters 
	for j in range(B.shape[0]):
		#parameters update
		#gradient 
		grad = (-1)*numpy.dot(X.T,(Y-p))
		#update parametrs
		B = B_old - alfa*grad
	return(B) 

def train(X,B,Y,alfa,epoch):

	epochs = []
	neg_LogL = [] 

	for i in range(epoch):

		#print('training example nr::::: '+str(i))

		B = GDA(X,B,Y,alfa)

		#print(B)

		negLL = neg_LL(X,B,Y)

		#progress display 
		if i%(epoch/10)==0:
			prog = i/epoch*100
			print('Completed: '+str(round(prog,1))+' %')
		#print(negLL)

		epochs.append(i)
		neg_LogL.append(negLL)

	plt.plot(epochs,neg_LogL)
	plt.show()

	return B



#load data in numpy arrays  (using ML stanford course data)
data = numpy.loadtxt(fname = '/Users/danielmodin/Downloads/machine-learning-ex2/ex2/ex2data1.txt', delimiter=',')

X = data[:,0:2]
Y = data[:,2]

#lav Y om til column vector 
Y  = numpy.array([Y]) 
Y = Y.T               #lav om til column vector 

#add column of 1s to account for b0 intercept parameter 
ones = numpy.ones(X.shape[0])
X = numpy.column_stack((ones,X))

#intialise beta parameters (not important to randomly initialize, since problem is convex, so no problem with varying local minima depending on intialization as in ex NNs)
ones = numpy.ones(X.shape[1]) 	#1-d array with no dimensions 
B = numpy.array([ones])			#lav et reelt matrix som er 1 row X 2 columns 
B = B.T 						#transpose til column vector 

##### MINIMIZE NEGATIVE LOG LIKELIHOOD #####
B = train(X,B,Y,0.0000115,100000) #get trained parameters
p = predict(X,B)  			 #use trained parameters to get prediciton ( in probability ) 

#evaluate accuracy 
#predict = 1 hvis p >= 0.5 og = 0 hvis p < 0.5
Y_pred = p
Y_pred[Y_pred >= 0.5] = 1
Y_pred[Y_pred < 0.5]  = 0
score = accuracy_score(Y,p)*100

print(numpy.column_stack((Y_pred,Y)))

print('parameters:::')
print(B)
print('Accuracy in percent')
print(score)



















