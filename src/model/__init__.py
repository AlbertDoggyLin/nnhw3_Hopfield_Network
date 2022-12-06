__all__=['test', 'readHopfieldData', 'addNoise']
import numpy as np

def readHopfieldData(datasetPath:str)->tuple[list[list[int]], int]:
    dataset, lineSize=[[]], 0
    with open(datasetPath) as file:
        for line in file:
            if line[-1]=='\n': line= line[0:-1]
            if len(line):
                dataset[-1]+=[1 if i=='1' else -1 for i in line]
                lineSize=len(line)
            else: dataset.append([])
    return dataset, lineSize

def addNoise(num:int, data:np.ndarray[int]):
    data=np.array(data)
    eidx=np.random.choice(data.shape[0], num, replace=False)
    data[eidx]*=-1
    return data

def test():
    datasetPath='dataset/Basic_Training.txt'
    testDataPath='dataset/Basic_Testing.txt'
    import numpy as np
    dataset, _ = readHopfieldData(datasetPath)
    dataset = np.array(dataset)
    from .Hopfield import HopfieldNetwork
    model = HopfieldNetwork()
    model.fit(dataset)
    testData, size = readHopfieldData(testDataPath)
    testData = np.array(testData)
    for data in testData: data=addNoise(30, data)
    for data in testData:
        for j in range(data.shape[0]//size):
            for k in data[j*size:(j+1)*size]:
                if k==1:print(1, end='')
                else: print(' ', end='')
            print()
        print()
    testData = np.array(testData)
    res = model.predict(testData, lim=-1)
    for i, m in zip(res, dataset):
        for j in range(res.shape[1]//size):
            for k in i[j*size:(j+1)*size]:
                if k==1:print(1, end='')
                else: print(' ', end='')
            print()
        print()


if __name__=="__main__":
    datasetPath='dataset/Basic_Training.txt'
    testDataPath='dataset/Basic_Testing.txt'
    import numpy as np
    dataset, _ = readHopfieldData(datasetPath)
    dataset = np.array(dataset)
    from Hopfield import HopfieldNetwork
    model = HopfieldNetwork()
    model.fit(dataset)
    testData, size = readHopfieldData(testDataPath)
    testData = np.array(testData)
    # for data in testData: data=addNoise(30, data)
    # for data in testData:
    #     for j in range(data.shape[0]//size):
    #         for k in data[j*size:(j+1)*size]:
    #             if k==1:print(1, end='')
    #             else: print(' ', end='')
    #         print()
    #     print()
    res = model.predict(testData, lim=-1)
    for i, m in zip(res, dataset):
        for j in range(res.shape[1]//size):
            for k in i[j*size:(j+1)*size]:
                if k==1:print(1, end='')
                else: print(' ', end='')
            print()
        print()