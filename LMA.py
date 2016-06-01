from sympy import *
import numpy as np

# example 1
x, y, z, a = symbols('x y z a ')
f,g,h = symbols('f g h',cls = Function)
g = x**2
f = z**2
h = [f,g]
initial_val = np.array([1,2,3,4])
initial_val.shape = 4,1
sym_total = [x,y,z,a]

'''#example 2 Powell's problem
x1, x2 = symbols('x1 x2')
f,g,h = symbols('f g h',cls = Function)
f = x1
g = (10*x1)/(x1+0.1)+2*x2*x2
h = [f,g]
initial_val = np.array([3.,1.])
initial_val.shape  = 2,1
sym_total = [x1,x2]
'''


def getparameter(Func,sym_list,sym_total):
	for	func in Func:
		for sym in func.atoms(Symbol):
			sym_list.append(sym)
	sym_exi = []
	for sym in sym_total:
		if sym_list.count(sym) > 0:
			sym_exi.append(sym)
	sym_list = sym_exi[:]
	
def Jacobian(Func,valuelist,sym_total):
	#calculate Jacobian for Rn->Rm function Func
	m = len(Func)
	n = len(sym_total)
	Jac = np.zeros((m,n))
	
	dict_key_value = {}
	idx = 0
	for sym in sym_total:
		dict_key_value.update({sym:valuelist[idx][0]})
		idx += 1
		
	row = 0
	for func in Func:
		for sym in func.atoms(Symbol):
			dif = diff(func,sym)
			idx = 0
			for sym_ori in sym_total:
				if sym_ori == sym:
					break
				else:
					idx += 1
			Jac[row][idx] = dif.subs(dict_key_value)
		row += 1
	return Jac
		
def obj_Func(Func,valuelist,sym_total):
	#calculate object Function vector
	result = np.zeros((len(Func),1))
	
	dict_key_value = {}
	idx = 0
	for sym in sym_total:
		dict_key_value.update({sym:valuelist[idx][0]})
		idx += 1
	
	row = 0
	for func in Func:
		result[row][0] = func.subs(dict_key_value)
		row += 1
	return result	



def LMA(Func,initial_val,sym_total,max_iter = 300,eps1 = 1e-8,eps2 = 1e-8,to = 0.001):
	#Levenberg-Marquardt method  reference material 
	#METHODS FOR NON-LINEAR LEAST SQUARES PROBLEMS 2nd Edition, April 2004
	iter = 0
	v = 2
	x = initial_val
	J = Jacobian(Func,initial_val,sym_total)
	f = obj_Func(Func,initial_val,sym_total)
	A = J.T.dot(J)
	g = J.T.dot(f)
	
	# calculate the infinite normal of g ,not sure the correction

	gnormal_inf = np.linalg.norm(g,np.inf)
	found = (gnormal_inf <= eps1)
	max = A[0][0]
	for i in range(len(A[0])):
		if A[i][i] >= max:
			max = A[i][i]

	miu = to*max
	I = np.eye(len(A[0]), dtype = float)
	while((not(found))and(iter < max_iter)):
		iter += 1
		h = np.linalg.solve((A+miu*I),-g)
		if (np.linalg.norm(h) <= eps2*(eps2+np.linalg.norm(x))):
			found = True
		else:
			x_new = x+h
			ro = (obj_Func(Func,x,sym_total).T.dot(obj_Func(Func,x,sym_total))-obj_Func(Func,x_new,sym_total).T.dot(obj_Func(Func,x_new,sym_total)))/(h.T.dot((miu*h-g)))
			rou = ro[0][0]
			if rou > 0:
				x = x_new
				J = Jacobian(Func,x,sym_total)
				f = obj_Func(Func,x,sym_total)
				A = J.T.dot(J)
				g = J.T.dot(f)
				gnormal_inf = np.linalg.norm(g,np.inf)
				found = (gnormal_inf <= eps1)
				max = 1./3.
				if (1-(2*rou-1))**3 > max:
					max = (1-(2*rou-1))**3
				miu = miu*max
				v = 2
			else:
				miu = miu*v
				v = 2*v
	print "iteration number: "+str(iter)
	return x


#print LMA(h,initial_val,sym_total)	
print LMA(h,initial_val,sym_total,100, 1e-15, 1e-15,1)
#getparameter(h,sym_list = [],sym_total = [x,y,z])
