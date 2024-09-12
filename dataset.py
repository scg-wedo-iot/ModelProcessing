import numpy as np
import random

def splitDataAndLabel(datas, labels, listUniqueLabel, percSplit, is_2d=False):
    # split train test
    # percSplit = 70

    # odd even
    # nImgs = datas.shape[0]
    # indexTrainAll = np.zeros(nImgs,dtype=bool)
    # posTrainAll = []
    # x = input of model = image
    dictClassInfo = {'class':[], 'n_train':[], 'n_valid':[], 'n_test':[]}
    for iclass in listUniqueLabel:
        '''
        ref         1 1 1 5 2 3 3 3 1 1
        i           0 1 2 3 4 5 6 7 8 9 
        class=1   
        indexClass  1 1 1 0 0 0 0 0 1 1 
        posClass    0 1 2           8 9
        '''
        indexClass = (labels == iclass)
        posClass = np.where(indexClass)
        posClass = np.array(posClass)
        nImgAll = sum(indexClass)

        nImgTrain = int(percSplit[0]*nImgAll)
        nImgValid = int(percSplit[1]*nImgAll)
        nImgTest = nImgAll - (nImgTrain + nImgValid)

        posTrain = random.sample(sorted(set(posClass[0, :])), nImgTrain)
        posClassFil = np.setdiff1d(posClass,posTrain)
        posValid = random.sample(sorted(set(posClassFil)), nImgValid)
        posTest = np.setdiff1d(posClassFil,posValid)

        if iclass==listUniqueLabel[0]:
            posTrainAll = posTrain
            posValidAll = posValid
            posTestAll = posTest
        else:
            posTrainAll = np.concatenate([posTrainAll,posTrain],axis=0)
            posValidAll = np.concatenate([posValidAll,posValid],axis=0)
            posTestAll = np.concatenate([posTestAll,posTest],axis=0)

        dictClassInfo['class'].append(iclass)
        dictClassInfo['n_train'].append(nImgTrain)
        dictClassInfo['n_valid'].append(nImgValid)
        dictClassInfo['n_test'].append(nImgTest)

    posTrainAll = np.sort(posTrainAll)
    x_train = datas[posTrainAll,:,:,:]
    y_train = labels[posTrainAll]

    posValidAll = np.sort(posValidAll)
    x_valid = datas[posValidAll,:,:,:]
    y_valid = labels[posValidAll]

    posTestAll = np.sort(posTestAll)
    x_test = datas[posTestAll,:,:,:]
    y_test = labels[posTestAll]

    if is_2d:
        x_train = x_train.astype(float) / 255.0
        x_train = np.squeeze(x_train, axis=3)
        x_valid = x_valid.astype(float) / 255.0
        x_valid = np.squeeze(x_valid, axis=3)
        x_test = x_test.astype(float) / 255.0
        x_test = np.squeeze(x_test, axis=3)

        y_train = y_train.astype(int)
        y_valid = y_valid.astype(int)
        y_test = y_test.astype(int)


    return x_train, y_train, x_valid, y_valid, x_test, y_test, posTrainAll, posValidAll, posTestAll, dictClassInfo
