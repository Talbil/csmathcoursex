import csv
import numpy as np
import os
from numpy import *
import matplotlib.pyplot as plt
from PIL import Image

def find_nearest(tar_point,points):#find nearest point from points to tar_point
	distance = []
	for item in points:
		distance.append(np.linalg.norm(tar_point-item))
	location = distance.index(np.min(distance))
	return location
	
filepath = os.path.abspath("/Users/zhengzhihao/Desktop/python/optdigits_orig.tra")
file = open(filepath)
'''
csvR = csv.reader(f)

A=[]
for line in csvR:
    if int(line[64])==3 :
        for i in range(0,65):
            line[i]=int(line[i])
        line.pop()
        A.append(line)
f.close()  
  '''
data_from_file = file.readlines()
data = []
for i in range(1934):
	if int(data_from_file[53+33*i]) == 3:#get digit 3
		line = 33*i+21
		line_list = []
		for j in range(32):
			for item in data_from_file[line+j].strip():
				line_list.append(int(item))
		data.append(line_list)
A = np.array(data,dtype = np.uint8)




meanVals = mean(A,axis = 0)
meanRemoved = A-meanVals
stded = meanRemoved/std(A)
print stded.shape
covMat = cov(stded , rowvar = 0)
eigVals , eigVects = linalg.eig(mat(covMat))
eigValInd = argsort(eigVals)
eigValInd = eigValInd[:-3:-1]
redEigVects = eigVects[:,eigValInd]
finalDataMat = stded*redEigVects
finalDataMat = finalDataMat.real

n = shape(finalDataMat)[0]
xcord = []
ycord = []
for i in range(n):
	xcord.append(finalDataMat[i,0])
	ycord.append(finalDataMat[i,1])
fig = plt.figure()
ax = fig.add_subplot(111)
finalDataMat_copy = finalDataMat.T
print finalDataMat_copy.shape
minx = np.min(finalDataMat_copy[0])
maxx = np.max(finalDataMat_copy[0])
miny = np.min(finalDataMat_copy[1])
maxy = np.max(finalDataMat_copy[1])

xlist = np.linspace(minx, maxx, 7)[1:6]
ylist = np.linspace(miny,maxy,7)[1:6]
location = []
for i in range(5):
	for j in range(5):
		point = np.array([xlist[i],ylist[4-j]])
		location.append(find_nearest(point,finalDataMat))
ax.scatter(xcord,ycord,s = 30,c = 'green',marker = 'o')
r_xcord = []
r_ycord = []
x = np.ravel(finalDataMat_copy[0])
y = np.ravel(finalDataMat_copy[1])

for i in range(len(location)):
	r_xcord.append(x[location[i]])
	r_ycord.append(y[location[i]])
ax.scatter(r_xcord,r_ycord,60,color = 'red',marker = 'o')



plt.grid(True)
plt.xlabel("first principal component")
plt.ylabel("second principla component")
plt.show()


bound_r = np.zeros([3,178],dtype=np.uint8)
bound_c = np.zeros([32,3],dtype=np.uint8)

img =bound_r
for i in range(5):
	col = bound_c
	for j in range(5):
		col = np.c_[col,255*(1-A[location[5*i+j],:]).reshape(32,32)]
		col = np.c_[col,bound_c]
	img = np.r_[img,col]
	img = np.r_[img,bound_r]

windows = Image.new("RGB",(172,172))
for i in range(172):
	for j in range(172):
		windows.putpixel([i,j],(img[i,j],img[i,j],img[i,j]))
windows.rotate(90).show()