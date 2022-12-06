__all__=['HopfieldNetwork']
from typing import Union
import numpy as np
class HopfieldNetwork:
    def __init__(self, networkShape:Union[None, int]=None) -> None:
        self._W:Union[None, np.ndarray[np.ndarray[float]]]=None
        self._theta:Union[None, np.ndarray[float]]=None
        if networkShape is int: 
            self._W=np.zeros((networkShape, networkShape))

    def fit(self, trainData:np.ndarray[np.ndarray[float]])->None:
        self._W=np.zeros((trainData.shape[1], trainData.shape[1]))
        self._theta=np.zeros(trainData.shape[1])
        scaler=np.ones(trainData.shape[0])
        S=0
        for i in range(trainData.shape[0]):
            for j in range(i):
                if (trainData[i]==trainData[j]).all() or (trainData[i]!=trainData[j]).all():
                    scaler[i]/=8
            S+=scaler[i]
        for i in range(trainData.shape[0]):
            data2d=np.array([trainData[i]])
            self._W+=(data2d.transpose()@data2d)*scaler[i]
        self._W-=S*(np.identity(trainData.shape[1]))
        for i in range(self._W.shape[0]):
            biasInterval=[-99999999, 99999999]
            for data in trainData:
                if (res:=self._W[i]@data)*data[i]>=0:
                    if res>=0 and biasInterval[0] < -res: biasInterval[0]= -res
                    if res<=0 and biasInterval[1] > -res: biasInterval[1]= -res
                else:
                    if res<0 and biasInterval[0] < -res: biasInterval[0]= -res
                    elif res < 0: print(data)
                    if res>0 and biasInterval[1] > -res: biasInterval[1]= -res
                    elif res > 0: print(data)
            if biasInterval[0]<=biasInterval[1]: self._theta[i]=(9*biasInterval[0]+15*biasInterval[1])/24

    def predict(self, testData:np.ndarray[Union[np.ndarray[int], int]], lim:int=-1)->np.ndarray[Union[np.ndarray[int], int]]:
        if needFlatten:=testData.ndim==1:
            testData=np.array([testData])
        for data in testData:
            counter=0
            while True:
                hater, ext = -1, 0
                for idx in range(self._W.shape[0]):
                    if (res:=self._W[idx]@data+self._theta[idx])>0 and data[idx]==-1:
                        if abs(res)>ext: hater, ext=idx, abs(res)
                    if res<0 and data[idx]==1:
                        if abs(res)>ext: hater, ext=idx, abs(res)
                if hater!=-1: data[hater]*=-1
                else: break
        return testData.flatten() if needFlatten else testData
    
    def next(self, data:np.ndarray[int]):
        hater, ext = -1, 0
        for idx in range(self._W.shape[0]):
            if (res:=self._W[idx]@data+self._theta[idx])>0 and data[idx]==-1:
                if abs(res)>ext: hater, ext=idx, abs(res)
            if res<0 and data[idx]==1:
                if abs(res)>ext: hater, ext=idx, abs(res)
        if hater!=-1: data[hater]*=-1
        return data, hater
    
    def compare(self, data, b):
        for idx in range(self._W.shape[0]):
            if (res:=self._W[idx]@data+self._theta[idx])*b[idx]<0:
                print(idx, res)
        print()
        

if __name__ == "__main__":
    model = HopfieldNetwork()
    testData = np.array([[1, -1, 1, -1, 1]])
    model.fit(testData)
    print(model)
    print(model._W)
    print(model.predict(np.array([1, -1, -1, -1, -1])))
