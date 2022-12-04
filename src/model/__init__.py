def readHopfieldData(datasetPath:str)->list[list[int]]:
    dataset, lineSize=[[]], 0
    with open(datasetPath) as file:
        for line in file:
            if line[-1]=='\n': line= line[0:-1]
            if len(line):
                dataset[-1]+=[1 if i=='1' else -1 for i in line]
                lineSize=len(line)
            else: dataset.append([])
    return dataset, lineSize


if __name__=="__main__":
    datasetPath='dataset/Bonus_Training.txt'
    testDataPath='dataset/Bonus_Testing.txt'
    import numpy as np
    dataset, _ = readHopfieldData(datasetPath)
    dataset = np.array(dataset)
    from Hopfield import HopfieldNetwork
    model = HopfieldNetwork()
    model.fit(dataset)
    testData, size = readHopfieldData(testDataPath)
    testData = np.array(testData)
    # eidx=np.random.choice(testData.shape[1], 30, replace=False)
    # for data in testData:
    #     data[eidx]*=-1
    #     for j in range(data.shape[0]//size):
    #         for k in data[j*size:(j+1)*size]:
    #             if k==1:print(1, end='')
    #             else: print(' ', end='')
    #         print()
    #     print()
    res = model.predict(testData, lim=-13)

    # for data in testData:
    #     for i in range(30):
    #         data=model.next(data)
    #         if i%3==0: 
    #             for j in range(data.shape[0]//size):
    #                 for k in data[j*size:(j+1)*size]:
    #                     if k==1:print(1, end='')
    #                     else: print(' ', end='')
    #                 print()
    #         print()
    for i, m in zip(res, dataset):
        for j in range(res.shape[1]//size):
            for k in i[j*size:(j+1)*size]:
                if k==1:print(1, end='')
                else: print(' ', end='')
            print()
        print()
        for j in range(res.shape[1]//size):
            for k in m[j*size:(j+1)*size]:
                if k==1:print(1, end='')
                else: print(' ', end='')
            print()
        print();print()