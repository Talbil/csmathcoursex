from numpy import *
import numpy.matlib as ml
from matplotlib import pyplot
from pylab import *
import random

#generate multidimension gauss distribution
def multgauss(mean, cov, size):
	import numpy as np
	return np.random.multivariate_normal(mean,cov,size).T
	
def distmat(X,Y):
	n = len(X)
	m = len(Y)
	xx = ml.sum(X*X, axis = 1)
	yy = ml.sum(Y*Y, axis = 1)
	xy = ml.dot(X,Y.T)
	return tile(xx, (m,1)).T+tile(yy, (n,1)) - 2*xy	
	
def init_params(centers,k):
	#setting EM algorithm initialization with kmeans algorithm
	pMiu = centers
	pPi = zeros([1,k],dtype = float)
	pSigma = zeros([len(X[0]), len(X[0]), k], dtype = float)
	#calculate the distants between X and others
	dist = distmat(X, centers)
	#allocate X to nearest point
	labels = dist.argmin(axis = 1)
	# recompute every point and parameterization
	for j in range(k):
		idx_j = (labels == j).nonzero()
		pMiu[j] = X[idx_j].mean(axis = 0)
		pPi[0, j] = 1.0 * len(X[idx_j]) / N
		pSigma[:, :, j] = cov(mat(X[idx_j]).T)
	return pMiu, pPi, pSigma

def calc_prob(k,pMiu,pSigma):
	Px = zeros([N,k], dtype = float)
	for i in range(k):
		Xshift = mat(X-pMiu[i, :])
		inv_pSigma = mat(pSigma[:, :, i]).I
		coef = pow((2*pi),(len(X[0])/2))*sqrt(linalg.det(mat(pSigma[:, :, i])))
		for j in range(N):
			tmp = (Xshift[j, :] * inv_pSigma * Xshift[j, :].T)
			Px[j, i] = 1.0 / coef*exp(-0.5*tmp)
	return Px

def pylab_plot(X,labels,iter):
	colors = array([[1, 0, 0],[0, 1 ,0], [0, 0, 1]])
	pyplot.plot(hold = False)
	pyplot.hold(True)
	labels = array(labels).ravel()
	data_colors=[colors[lbl] for lbl in labels]
	xcord = X[:, 0]
	ycord = X[:, 1]
	pyplot.scatter(xcord,ycord,c = data_colors,alpha = 0.5)
	pyplot.show()
	
	

def EM(X, k, threshold = 1e-15,maxiter = 300):
	N = len(X)
	labels = zeros(N, dtype = int)
	centers = array(random.sample(X,k))
	iter = 0
	pMiu, pPi, pSigma = init_params(centers, k)
	Lprev = float('-10000')
	pre_eps = 100000
	while iter < maxiter:
		Px = calc_prob(k,pMiu,pSigma)
		pGamma = mat(array(Px)*array(pPi))
		pGamma = pGamma / pGamma.sum(axis = 1)
		Nk = pGamma.sum(axis = 0)
		pMiu = diagflat(1 / Nk) * pGamma.T * mat(X)
		pPi = Nk /N
		pSigma = zeros([len(X[0]), len(X[0]), k],dtype = float)
		for j in range(k):
			Xshift = mat(X) - pMiu[j, :]
			for i in range(N):
				pSigmaK = Xshift[i, :].T * Xshift[i, :]
				pSigmaK = pSigmaK * pGamma[i, j] / Nk[0, j]
				pSigma[:, :, j] = pSigma[:, :, j] + pSigmaK
		
		labels = pGamma.argmax(axis = 1)
		iter += 1
		
		L = sum(log(mat(Px) * mat(pPi).T))
		cur_eps = L - Lprev
		if cur_eps < threshold:
			break
		if cur_eps > pre_eps:
			break
		pre_eps = cur_eps
		Lprev = L
		print "iter %d eps %lf" % (iter, cur_eps)
	pylab_plot(X, labels, iter)
	pyplot.show()

if __name__ == '__main__':
	mean1 = [4, 4]
	cov1 = [[4, 3], [3, 4]]
	samp1 = multgauss(mean1, cov1, 500)
	mean2 = [30, 40]
	cov2 = [[10,  0], [0, 10]]
	samp2 = multgauss(mean2,cov2,500)
	
	
	N = len(samp1[0]) + len(samp2[0])
	X = zeros((N,2))
	for i in range(N):
		if i < len(samp1[0]):
			X[i, 0] = samp1[0][i]
			X[i, 1] = samp1[1][i]
		else:
			X[i, 0] = samp2[0][i-len(samp1[0])]
			X[i, 1] = samp2[1][i-len(samp1[0])]
	pyplot.plot(X[:,0], X[:, 1],'ro')
	pyplot.show()
	EM(X,2)