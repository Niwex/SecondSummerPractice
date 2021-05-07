import numpy as np
from scipy.linalg import norm


class GK:
    def __init__(self, n_clusters=4, max_iter=100, m=2, error=1e-6):
        super().__init__()
        self.u, self.centers, self.f = None, None, None
        self.clusters_count = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error

    def fit(self, z):
        N = z.shape[0]
        C = self.clusters_count
        centers = []

        u = np.random.dirichlet(np.ones(N), size=C)

        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()

            centers = self.next_centers(z, u)
            f = self._covariance(z, centers, u)
            dist = self._distance(z, centers, f)
            u = self.next_u(dist)
            iteration += 1

            # Stopping rule
            if norm(u - u2) < self.error:
                break

        self.f = f
        self.u = u
        self.centers = centers
        return centers

    def next_centers(self, z, u):
        um = u ** self.m
        return ((um @ z).T / um.sum(axis=1)).T

    def _covariance(self, z, v, u):
        um = u ** self.m

        denominator = um.sum(axis=1).reshape(-1, 1, 1)
        temp = np.expand_dims(z.reshape(z.shape[0], 1, -1) - v.reshape(1, v.shape[0], -1), axis=3)
        temp = np.matmul(temp, temp.transpose((0, 1, 3, 2)))
        numerator = um.transpose().reshape(um.shape[1], um.shape[0], 1, 1) * temp
        numerator = numerator.sum(0)

        return numerator / denominator

    def _distance(self, z, v, f):
        dif = np.expand_dims(z.reshape(z.shape[0], 1, -1) - v.reshape(1, v.shape[0], -1), axis=3)
        determ = np.power(np.linalg.det(f), 1 / self.m)
        det_time_inv = determ.reshape(-1, 1, 1) * np.linalg.pinv(f)
        temp = np.matmul(dif.transpose((0, 1, 3, 2)), det_time_inv)
        output = np.matmul(temp, dif).squeeze().T
        return np.fmax(output, 1e-8)

    def next_u(self, d):
        power = float(1 / (self.m - 1))
        d = d.transpose()
        denominator_ = d.reshape((d.shape[0], 1, -1)).repeat(d.shape[-1], axis=1)
        denominator_ = np.power(d[:, None, :] / denominator_.transpose((0, 2, 1)), power)
        denominator_ = 1 / denominator_.sum(1)
        denominator_ = denominator_.transpose()

        return denominator_

    def predict(self, z):
        if len(z.shape) == 1:
            z = np.expand_dims(z, axis=0)

        dist = self._distance(z, self.centers, self.f)
        if len(dist.shape) == 1:
            dist = np.expand_dims(dist, axis=0)

        u = self.next_u(dist)
        return np.argmax(u, axis=0)
    

import numpy as np
import pandas as pd


class MakeClusterName:
    normData = []
    def __init__(self, clusterAlghoritm, eps = 0.1, shablon1 ='',shablon2=' ',shablon3 = ''):
        self.a = clusterAlghoritm
        self.eps = eps
        self.namesX = []
        self.namesY = []
        self.shablonStart = shablon1
        self.shablonBetween = shablon2
        self.shablonEnd = shablon3
        
    def makeNames(self,data,X_names, Y_names , Normalize = False):
        a = self.a
        n_clusters = a.clusters_count
        maxX = max(data[:,0])
        minX = min(data[:,0])
        maxY = max(data[:,1])
        minY = min(data[:,1])
        
        m = np.zeros(2)
        m[0]= int(len(X_names))
        m[1]= int(len(Y_names))
        granici_x , granici_y = self._zone(maxX, minX, maxY, minY, m)
        if Normalize == False:
            self.namesX,self.namesY = self._getname(a,data, granici_x,granici_y,n_clusters, X_names, Y_names)
        if Normalize == True:
            self.namesX, self.namesY = self._getNormName(a,data, n_clusters, X_names, Y_names)
        return self.namesX,self.namesY 
        
    def _zone(self,max_x,min_x,max_y,min_y, m): 
        shag_x = round((max_x-min_x)/m[0], 1)
        shag_y = round((max_y-min_y)/m[1], 1)
        granici_x = [[0]*2 for i in range(int(m[0]))]
        granici_y = [[0]*2 for i in range(int(m[1]))]
        zonex = min_x
        zoney = min_y
        
        for j in range(2):
            if j == 0: 
                for x in granici_x:
                    x[0] = zonex
                    x[1] = zonex+shag_x
                    zonex = zonex+shag_x
            if j == 1:
                for y in granici_y:
                    y[0] = zoney
                    y[1] = zoney+shag_y
                    zoney = zoney+shag_y
        return granici_x , granici_y           
        
    def _getname(self,algorithm,data, granici_x, granici_y, n_clusters, name_x, name_y):
        numofVectors = data.shape[0]
        names_x = [0]*n_clusters
        names_y = [0]*n_clusters
        
        for j in range(int(n_clusters)):
            counter_x = np.zeros(len(granici_x))
            counter_y = np.zeros(len(granici_y))
            u_x =np.zeros(len(granici_x))
            u_y =np.zeros(len(granici_y))
            for i in range(numofVectors):            
                if algorithm.u[j][i]>0.5:
                    l = 0
                    while l < len(granici_x):
                        if data[i][0]>granici_x[l][0] and data[i][0]<=granici_x[l][1]:
                            counter_x[l]+=1
                        l+=1
                    l = 0
                    while l < len(granici_y):
                        if data[i][1]>granici_y[l][0] and data[i][1]<=granici_y[l][1]:
                            counter_y[l]+=1
                        l+=1
            
            for i in range(int(counter_x.shape[0])):
                u_x[i] = counter_x[i]/sum(counter_x)
            for i in range(int(counter_y.shape[0])):
                u_y[i] = counter_y[i]/sum(counter_y)
                
            names_x[j] = self._nameOnerow(u_x,name_x)
            names_y[j] = self._nameOnerow(u_y,name_y)      
        return names_x, names_y
    
    def _nameOnerow(self,u_x , name_x):
        names = ''
        print("u_x ", u_x)
        for i in range(u_x.shape[0]):
            
            if u_x[i]<self.eps:
                continue
                
            else:
                k = i
                maxim = i
                while k < u_x.shape[0]:
                    if u_x[maxim]<u_x[k]:
                        maxim = k
                    k+=1
                if maxim != u_x.shape[0]-1 and maxim != 0:
                    if abs(u_x[maxim] - u_x[maxim+1])< self.eps and u_x[maxim+1]> u_x[maxim-1]:
                        names = name_x[maxim]+' and '+name_x[maxim+1]
                        break
                    elif abs(u_x[maxim] - u_x[maxim-1])< self.eps and u_x[maxim+1]< u_x[maxim-1]:
                        names = name_x[maxim-1]+' and '+name_x[maxim]
                        break
                    else:
                        names = name_x[maxim]
                        break
                elif maxim == 0:
                    if abs(u_x[maxim] - u_x[maxim+1])< self.eps:
                        names = name_x[maxim]+' and '+name_x[maxim+1]
                        break
                    else:
                        names = name_x[maxim]
                        break
                elif maxim == u_x.shape[0]-1:
                    if abs(u_x[maxim] - u_x[maxim-1])< self.eps:
                        names = name_x[maxim-1]+' and '+name_x[maxim]
                        break
                    else:
                        names = name_x[maxim]
                        break
        return names
    
    def showCluster(self,data,nameX ='0',nameY ='0'):
        a = self.a
        n_clusters = self.a.clusters_count
        names_x = self.namesX
        names_y = self.namesY
        if nameX !='0' and nameY !='0':
            for i in range(n_clusters):
                if nameX == names_x[i] and nameY == names_y[i]:
                    self._printCluster(a,data,i)
        elif nameX !='0' and nameY =='0':
            for i in range(n_clusters):
                if nameX == names_x[i]:
                    self._printCluster(a,data,i)
        elif nameX =='0' and nameY !='0':
            for i in range(n_clusters):
                if nameY == names_y[i]:
                    self._printCluster(a,data,i)
        else:
            print ('Enter "nameX" or "nameY" ')
    
    def _normalizeByVika(self, data):
        normData = pd.DataFrame(data)
        test_df_norm = (normData[0] - normData[0].min()) / (normData[0].max() - normData[0].min())
        test_df_norm = pd.DataFrame(test_df_norm)
        test_df_norm[1] = (normData[1] - normData[1].min()) / (normData[1].max() - normData[1].min())
        self.normData = test_df_norm.values
        return np.array(test_df_norm.values)
    
    def _getNormName(self, a,data, n_clusters, name_x, name_y):
        dataNorm = self._normalizeByVika(data)
        numofVectors = data.shape[0]
        sumCounter =[0]*n_clusters
        names_x = [0]*n_clusters
        names_y = [0]*n_clusters
        granici_x = [[0, 0.4],[0.3, 0.7],[0.6, 1]]
        granici_y = [[0, 0.4],[0.3, 0.7],[0.6, 1]]
        for j in range(int(n_clusters)):
            for i in range(numofVectors):
                if a.u[j][i]>0.5:
                    sumCounter[j] += 1
            
        for j in range(int(n_clusters)):
            counter_x = np.zeros(len(granici_x))
            counter_y = np.zeros(len(granici_y))
            u_x =np.zeros(len(granici_x))
            u_y =np.zeros(len(granici_y))
            for i in range(numofVectors):            
                if a.u[j][i]>0.5:
                    l = 0
                    while l < len(granici_x):
                        if dataNorm[i][0]>=granici_x[l][0]:
                            if dataNorm[i][0]<=granici_x[l][1]:
                                counter_x[l]+=1

                        l+=1
                    l = 0
                    while l < len(granici_y):
                        if dataNorm[i][1]>=granici_y[l][0]:
                            if dataNorm[i][1]<=granici_y[l][1]:
                                counter_y[l]+=1

                        l+=1
            
            for i in range(int(counter_x.shape[0])):
                u_x[i] = counter_x[i]/sumCounter[j]

            for i in range(int(counter_y.shape[0])):
                u_y[i] = counter_y[i]/sumCounter[j]
                
            names_x[j] = self._nameOnerow(u_x,name_x)
            names_y[j] = self._nameOnerow(u_y,name_y)      
        return names_x, names_y
    
    def _printCluster(self,a,data,clusterNummber):
        print('cluster № ',clusterNummber, ':')
        print('cluster name - ',self.shablonStart, self.namesX[clusterNummber],self.shablonBetween, self.namesY[clusterNummber], self.shablonEnd, sep='')
        for i in range(data.shape[0]):
            if a.u[clusterNummber][i]>0.5:
                print(data[i])
        
        
                
        