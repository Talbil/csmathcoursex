import numpy as np
import random as rand
import matplotlib.pyplot as plt
def curve_fit(M = 3,label_1 = "M = 3",Reg = 0,sam = 10):
	
	lx = np.linspace(0,1,100)
	ly = np.sin(2*np.pi*lx)
	plt.plot(lx,ly,lw = 2,color = "green",label = "$sin(2\pi x)$")
	
	
	x = np.linspace(0,1,sam)
	y = np.sin(2*np.pi*x)
	
	
	gauss_y = np.empty(sam)
	for i in range(0,sam):
		gauss_y[i] = rand.gauss(y[i],0.2)	
	
	plt.plot(x,gauss_y,'bo',markersize = 10, markeredgewidth = 2,fillstyle = 'none')
	
	plt.xlabel("x")
	plt.ylabel("y")

    
	P = np.zeros((sam,M+1))
	
	for i in range(0 , M+1):
		P[:,i] = pow(x,i)	
	w = np.dot( np.dot(np.linalg.inv(np.dot(P.T , P)+Reg*np.eye(M+1)),P.T),gauss_y )
	
	ly = 0
	for i in range(0 , M+1):
		ly += pow(lx,i)*w[i]
	
	plt.plot(lx,ly,label = label_1,color = "red",lw = 2 )
	plt.legend()
	plt.show()
	return 
	
plt.figure("M = 3")
curve_fit(M = 3,label_1 = "M = 3",Reg = 0,sam = 10)
plt.figure("M = 9")
curve_fit(M = 9,label_1 = "M = 9",Reg = 0,sam = 10)
plt.figure("M = 9, 15 Samples")
curve_fit(M = 9,label_1 = "M = 9",Reg = 0,sam = 15)
plt.figure("M = 9, 100 Samples")
curve_fit(M = 9,label_1 = "M = 9",Reg = 0,sam = 100)	
plt.figure("M = 9 , 10 Samples with regularization term")
curve_fit(M = 3,label_1 = "$ln\lambda  = -18$",Reg = pow(np.e,-18),sam = 10)