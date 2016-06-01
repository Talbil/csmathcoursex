import numpy as np
import math
import pylab as pl
def Lagrange(G,r,A,b):
# solve convex quadratic programming with equality constrains by Lagrange multiplication method
# f(x) = 1/2*d.T*G*d+r.T*d     s.t. A*d = b
	m,n = A.shape
	new_K_1 = np.vstack((G,A))
	new_K_2 = np.vstack((A.T,np.zeros((m,m))))
	new_K = np.hstack((new_K_1,new_K_2))
	new_l = np.vstack((-r,b))	
	x = np.linalg.solve(new_K,new_l)
	solution = np.zeros((n,1))
	lema = np.zeros((m,1))
	for i in range(n):
		solution[i][0] = x[i][0]
	for i in range(m):
		lema[i][0] = x[i+n][0]
	return solution ,lema
	
def ESM(initial_point,H,r,G,h,A = [[0]],b = 0):
#solve convex Quadratic Programming by effective set method
#Gx <= h Ax = 0
	#Step 1: find effective set indices
	I = []
	A_r,A_c = A.shape
	if A_r == len(initial_point):
		r_A,c_A = A.shape
	else:
		r_A = 0
		c_A = 0
	m,n = G.shape
	for i in range(m):
		sum = 0
		for j in range(n):
			sum += initial_point[j][0]*G[i][j]
		if (sum - h[i][0]) < 1e-5 and (sum - h[i][0]) > -1e-5:
			I.append(i)
	iter = 0

	x = initial_point
	while True and iter < 500:	
	#Step 2 solve quadratic programming about direction 
		new_r = H.dot(x)+r
		new_A = np.zeros((r_A+len(I),n))
		new_b = np.zeros((r_A+len(I),1))

		for i in range(r_A):
			for j in range(n):
				new_A[i][j] = A[i][j]
		count = 0
		for index in I:
			for j in range(n):
				new_A[r_A+count][j] = G[index][j]
			count += 1
		d, lema = Lagrange(H,new_r,new_A,new_b)
		is_zero_vector = True
		for i in range(len(d)):
			if  np.linalg.norm(d) > 1e-5:
				is_zero_vector = False
				break
		if is_zero_vector:
		#Step 3 if d is zero vector
			is_final_solution = True
			min_idx = I[0]
			min_lema = lema[0][0]
			for i in range(len(lema)):
				if lema[i][0] < 0:
					is_final_solution = False
				if lema[i][0] < min_lema:
					min_lema = lema[i][0]
					min_idx = I[i]
			if is_final_solution:
				lemda = np.zeros((m,1))
				counter = 0
				for i in I:
					lemda[i][0] = lema[counter]
					counter += 1 
				return x ,lemda
				break
			else:
				I.remove(min_idx)
			iter += 1
		else:
		#Step 4 if d is not zero vector
			alph = 1.
			min_idx = -1
			for i in range(m):
				if I.count(i) == 0:
					k = (h[i][0]-G[i].T.dot(x))/(G[i].T.dot(d))
					if k <= alph and G[i].dot(d) > 0:
						min_idx = i
						alph  = k
			x = x+alph*d
			if min_idx != -1:
				I.append(min_idx)
			iter += 1
	return x,lema


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def gen_data_and_label(point_n = 50,mean1 = [-1,2],mean2 = [1,-1],mean3 = [5,-5],mean4 = [-6,2],cov = [[1.0,0],[0,1.0]]):
# generate 2D data and their labels(in two labels)
	X1 = np.vstack((np.random.multivariate_normal(mean1,cov,point_n),np.random.multivariate_normal(mean2,cov,point_n)))
	X2 = np.vstack((np.random.multivariate_normal(mean3,cov,point_n),np.random.multivariate_normal(mean4,cov,point_n)))
	#Y1 Y2 set shape (len(X1),1) or (len(X1))
	Y1 = np.ones(len(X1))
	Y2 = np.ones(len(X2)) * -1
	return X1,Y1,X2,Y2

def split_data(X1,Y1,X2,Y2):
	samples_num,samples_dim = X1.shape
	train_len = math.floor(samples_num * 0.9)
	test_len = samples_num - train_len
	X_train = np.vstack((X1[:train_len],X2[:train_len]))
	Y_train = np.hstack((Y1[:train_len],Y2[:train_len]))
	X_test = np.vstack((X1[train_len:],X2[train_len:]))
	Y_test = np.hstack((Y1[train_len:],Y2[train_len:]))
	return X_train,Y_train,X_test,Y_test
	

def pylab_plot_contour(X1_train, X2_train, alph, sv_y, sv_data,grid_half_len = 8):
    pl.plot(X1_train[:,0], X1_train[:,1], "ro")
    pl.plot(X2_train[:,0], X2_train[:,1], "yo")
    pl.scatter(sv_data[:,0], sv_data[:,1], s=100, c="g")
    X1, X2 = np.meshgrid(np.linspace(-grid_half_len,grid_half_len,100), np.linspace(-grid_half_len,grid_half_len,100))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = project(X ,alph, sv_data, sv_y, sv_b).reshape(X1.shape)
    pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')
    pl.axis("tight")
    pl.show()


def SVM(X, Y):
    samples_num, samples_dim = X.shape
    K = np.zeros((samples_num, samples_num))#define kernel matrix
    H = np.zeros((samples_num, samples_num))#define QP Hessian Matrix
    A = np.zeros((1,samples_num))
    for i in range(samples_num):
    	A[0, i] = Y[i]
    	for j in range(samples_num):
    		K[i, j] = gaussian_kernel(X[i], X[j])
    		H[i, j] = K[i, j]*Y[i]*Y[j]
    r = np.ones((samples_num,1)) * -1
    b = np.zeros((1,1))
    G = np.diag(np.ones(samples_num) * -1)
    h = np.zeros((samples_num,1))
    count = 0
    for i in range(samples_num):
    	if (Y[i] == 1):
    		count += 1
    initial_point = np.ones((samples_num,1))
    for i in range(samples_num):
    	if Y[i] == 1:
    		initial_point[i][0] = 1./count
    		#initial_point[i][0] *= (samples_num-count)
    	else:
    		initial_point[i][0] = 1./(samples_num-count)
    		#initial_point[i][0] *= count
    solution,lema = ESM(initial_point,H,r,G,h,A,b)
    a = np.ravel(solution)
    
    # find  Lagrange multipliers larger than zero, Support vectors have non zero lagrange multipliers
    sv = a > 1e-5
    ind = np.arange(len(a))[sv]
    alph = a[sv]
    sv_data = X[sv]
    sv_y = Y[sv]
    print "%d support vectors out of %d points" % (len(alph), samples_num)
    # Intercept
    sv_b = 0
    for n in range(len(alph)):
        sv_b += sv_y[n]
        sv_b -= np.sum(alph* sv_y * K[ind[n],sv])
    sv_b /= len(alph)
    return alph,sv_data,sv_y,sv_b

def project(X,alph,sv_data,sv_y,sv_b):
    y_predict = np.zeros(len(X))
    for i in range(len(X)):
        s = 0
        for a, data, y in zip(alph, sv_data, sv_y):
        	s += a * y * gaussian_kernel(X[i], data)
        y_predict[i] = s
    return (y_predict + sv_b)

def predict(X,alph,sv_data,sv_y,sv_b):
    return np.sign(project(X,alph,sv_data,sv_y,sv_b))


if __name__ == "__main__":
    X1, Y1, X2, Y2 = gen_data_and_label()
    #X_train, Y_train = split_train(X1, Y1, X2, Y2)
    #X_test, Y_test = split_test(X1, Y1, X2, Y2)
    X_train,Y_train,X_test,Y_test = split_data(X1,Y1,X2,Y2)
    alph,sv_data,sv_y,sv_b = SVM(X_train, Y_train)
    Y_predict = predict(X_test,alph,sv_data,sv_y,sv_b)
    correct = np.sum(Y_predict == Y_test)
    print "%d out of %d predictions correct" % (correct, len(Y_predict))
    pylab_plot_contour(X_train[Y_train==1], X_train[Y_train==-1], alph, sv_y, sv_data,15)
