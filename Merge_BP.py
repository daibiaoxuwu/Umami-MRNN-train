# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import random
import os
import math


def Train(data, label, feature_num, names,set_epochs=2500):
    DOPCA = False



    # divide testset and trainset
    index1 = [i for i in range(289)]
    random.shuffle(index1)
    index2 = [i + 289 for i in range(237)]
    random.shuffle(index2)
    trainindex = index1[60:]
    trainindex.extend(index2[50:])
    testindex = index1[0:60]
    testindex.extend(index2[0:50])
    testdata = data.take(testindex, 0)
    testlabel = label.take(testindex, 0)
    testnames = [names[item] for item in testindex]
    traindata = data.take(trainindex, 0)
    trainlabel = label.take(trainindex, 0)
    trainnames = [names[item] for item in trainindex]

    # for i in range(len(trainlabel)):
    #   if trainlabel[i][0]=='F':
    #     for j in range(2):
    #       traindata=np.vstack((traindata,traindata[i,:]))
    #       trainlabel.append(trainlabel[i])




    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(feature_num,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)


    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    model.fit(traindata, trainlabel, epochs=set_epochs, batch_size=64, validation_split=1 / 5, shuffle=True)

    predictions = model.predict(testdata)

    #m2=tflearn.DNN(layer)

    acc_num = 0
    for i in range(len(testdata)):
        a = predictions[i][0]
        b = testlabel[i]
        c = sum(testdata[i])
        # if (b==40) != (a>=40):print(i,a,b,c,names[i])

        if (b == 40) == (a >= 20):
            acc_num += 1
        if (b == 40) != (a >= 20):
            print(i, a, b, c, testnames[i])
            #dict[testnames[i]]+=1
    print("ACC=", acc_num / 110)
    negP = sorted([predictions[j][0] for j in range(60)])
    posP = sorted([predictions[j + 60][0] for j in range(50)])

    # find the best Threshold and acc
    bestacc = 0
    bestmax = 0
    bestmin = 0
    accList = []
    maxList = []
    minList = []
    for i in range(60):
        err = 0
        min = 0
        for j in range(49, -1, -1):
            if posP[j] < negP[i]:
                min = posP[j]
                break
            else:
                err += 1
        acc = (110 - err - i) / 110
        if bestacc < acc:
            bestacc = acc
            bestmax = negP[i]
            bestmin = min
        if acc >= 0.9:
            accList.append(acc)
            maxList.append(negP[i])
            minList.append(min)

    for i in range(len(accList)):
        print("acc=", accList[i], "upbound=", maxList[i], "lowbound=", minList[i])
    print("--------------------------------------------")
    print("Bestacc=", bestacc, "upbound=", bestmax, "lowbound=", bestmin)
