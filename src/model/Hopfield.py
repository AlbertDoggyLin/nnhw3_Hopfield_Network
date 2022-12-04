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
        if self._W is None:
            self._W=np.zeros((trainData.shape[1], trainData.shape[1]))
            self._theta=np.zeros(trainData.shape[1])
        tmp=[]
        for i in range(trainData.shape[0]):
            for j in range(i-1, -1, -1):
                if (trainData[i]==trainData[j]).all() or (trainData[i]!=trainData[j]).all():
                    break
            else: tmp.append(trainData[i])
        trainData=np.array(tmp)
        for data in trainData:
            data2d=np.array([data])
            self._W+=data2d.transpose()@data2d
        self._W-=trainData.shape[0]*(np.identity(trainData.shape[1]))
        for i in range(self._W.shape[0]):
            biasInterval=[-99999999, 99999999]
            for data in trainData:
                if (res:=self._W[i]@data*data[i])>=0:
                    if res>=0 and biasInterval[0] < -res: biasInterval[0]= -res
                    if res<=0 and biasInterval[1] > -res: biasInterval[1]= -res
                else:
                    if res<0 and biasInterval[0] < -res: biasInterval[0]= -res
                    elif res < 0: print(data)
                    if res>0 and biasInterval[1] > -res: biasInterval[1]= -res
                    elif res > 0: print(data)
            if biasInterval[0]==1 or biasInterval[1]==-1: self._theta[i]=biasInterval[1] if biasInterval[0]==1 else biasInterval[0]
        print(self._theta)

    def predict(self, testData:np.ndarray[Union[np.ndarray[int], int]], lim:int=-1)->np.ndarray[Union[np.ndarray[int], int]]:
        if needFlatten:=testData.ndim==1:
            testData=np.array([testData])
        for data in testData:
            counter=0
            while True and counter!=lim:
                hater, ext = -1, 0
                for idx in range(self._W.shape[0]):
                    if (res:=self._W[idx]@data+0*self._theta[idx])>0 and data[idx]==-1:
                        if abs(res)>ext: hater, ext=idx, abs(res)
                    if res<0 and data[idx]==1:
                        if abs(res)>ext: hater, ext=idx, abs(res)
                if hater!=-1: data[hater]*=-1
                else: break
                counter+=1
        return testData.flatten() if needFlatten else testData
    
    def next(self, data:np.ndarray[int]):
        hater, ext = -1, 0
        for idx in range(self._W.shape[0]):
            if (res:=self._W[idx]@data+0*self._theta[idx])>0 and data[idx]==-1:
                if abs(res)>ext: hater, ext=idx, abs(res)
            if res<0 and data[idx]==1:
                if abs(res)>ext: hater, ext=idx, abs(res)
        if hater!=-1: data[hater]*=-1
        return data

if __name__ == "__main__":
    model = HopfieldNetwork()
    testData = np.array([[1, -1, 1, -1, 1]])
    model.fit(testData)
    print(model)
    print(model._W)
    print(model.predict(np.array([1, -1, -1, -1, -1])))
