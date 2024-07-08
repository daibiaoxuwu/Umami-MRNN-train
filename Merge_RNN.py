# coding:utf-8
__author__ = 'jmh081701'

import tensorflow as tf
import numpy as np
import math
import random

def RNN(TESTPERCENT=0.2):


    df=pd.read_excel('PeptideSequence.xlsx',sheet_name='Umami',dtype='object')

    raw_data=df.to_numpy()
    raw_data=np.delete(raw_data,[0,1],1)
    steplist=[]
    for i in range(np.shape(raw_data)[0]):
        j=0
        while j<np.shape(raw_data)[1]:
            if not isinstance(raw_data[i,j],str):
               if np.isnan(raw_data[i,j]):
                   for k in range(np.shape(raw_data)[1]-j):
                    raw_data[i,j+k]='0000000000'
                   break

            j+=1
        j-=1
        steplist.append(j)

    data=np.array([
        [[float(raw_data[examples,step][feature]) for feature in range(len(raw_data[0,0]))] for step in range(np.shape(raw_data)[1])] for examples in range(np.shape(raw_data)[0])
    ])



    raw_data=df.to_numpy()

    umamidata_len=raw_data.shape[0]
    umamidata_featurenum=np.shape(data)[1]

    names = df['Peptide Sequence'].values.tolist()


    ls = []
    exampleSheet = df[list(df.keys())[1]]
    # get threshold
    for i in range(exampleSheet.shape[0]):
        d = exampleSheet.iloc[i]

        d2 = 0
        if type(d) is str:
            d = d.split('>')[-1].split('<')[-1].split(' ')[0].split('m')[0]
            if len(d) == 0:
                d2 = 10
            else:
                d2 = float(d)
        else:
            if math.isnan(d):
                d2 = 10
            else:
                d2 = d
        ls.append(d2)




    ##########bitter data##########
    df=pd.read_excel('PeptideSequence.xlsx',sheet_name='Bitter',dtype='object')

    raw_data=df.to_numpy()
    raw_data=np.delete(raw_data,0,1)
    steplist=[]
    for i in range(np.shape(raw_data)[0]):
        j=0
        while j<np.shape(raw_data)[1]:
            if not isinstance(raw_data[i,j],str):
               if np.isnan(raw_data[i,j]):
                   for k in range(np.shape(raw_data)[1]-j):
                    raw_data[i,j+k]='0000000000'
                   break

            j+=1
        j-=1
        steplist.append(j)

    data_bitter=np.array([
        [[float(raw_data[examples,step][feature]) for feature in range(len(raw_data[0,0]))] for step in range(np.shape(raw_data)[1])] for examples in range(np.shape(raw_data)[0])
    ])
    unshuffle_bitterdata=data_bitter

    bitterdata_len=raw_data.shape[0]
    bitterdata_featurenum=np.shape(data_bitter)[1]

    # if bitterdata_featurenum>umamidata_featurenum:
    expand_data=data
    # elif bitterdata_featurenum<umamidata_featurenum:
    #     expand_data=data_bitter
    # else:
    #     expand_data=0

    if ~isinstance(expand_data,int):
        expand_column=abs(bitterdata_featurenum-umamidata_featurenum)
        expand_pad=np.array([
            [[0 for feature in range(len(raw_data[0,0]))] for column in range(expand_column)] for examples in range(np.shape(expand_data)[0])
        ])
        #expand_data=np.stack((expand_data,expand_pad),axis=0)
        #expand=np.insert(expand_data,15,values=expand_pad,axis=1)
        data=np.hstack((expand_data,expand_pad))
    unshuffle_umamidata = data
    raw_data=df.to_numpy()
    names_bitter = df['Peptide Sequence'].values.tolist()


    #divide test and train
    #umami
    index=[i for i in range(umamidata_len)]
    random.shuffle(index)
    divideflag=int(umamidata_len*TESTPERCENT)
    testx=data.take(index[0:divideflag],0)
    testy=[ls[examples] for examples in index[0:divideflag]]
    trainx=data.take(index[divideflag:-1],0)
    trainy=[ls[example] for example in index[divideflag:-1]]

    #bitter
    index=[i for i in range(bitterdata_len)]
    random.shuffle(index)
    divideflag=int(bitterdata_len*TESTPERCENT)
    testx_b=data_bitter.take(index[0:divideflag],0)
    testx=np.vstack((testx,testx_b))
    testy.extend([40 for examples in index[0:divideflag]])
    testy=np.array(testy)
    trainx=np.vstack((trainx,data_bitter.take(index[divideflag:-1],0)))
    trainy.extend([40 for example in index[divideflag:-1]])
    trainy=np.array(trainy)



    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((39, 10)),
        tf.keras.layers.Masking(mask_value=0),
        tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        tf.keras.layers.Dropout(rate=0.01),
        tf.keras.layers.SimpleRNN(20),
        tf.keras.layers.Dropout(rate=0.01),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError,
                  optimizer=tf.keras.optimizers.Adam(0.01))
    record=model.fit(trainx,trainy,epochs=75,verbose=1,validation_split=1 / 5, shuffle=True)
    model.evaluate(testx,testy)
    prediction=model.predict(testx)
    err=0
    for i in range(len(testy)):
        if ~((prediction[i]>20)==(testy[i]==40)):
            err+=1
        print(prediction[i],' ',testy[i])
    Acc=(np.shape(testy)[0]-err)/np.shape(testy)[0]
    print("Finish training RNN, Acc=",Acc)

    partial_output = [layer.output for layer in model.layers[1:]]
    visual_model = tf.keras.models.Model(inputs=model.inputs, outputs=partial_output)

    bitter_featuremap = visual_model.predict(unshuffle_bitterdata)[2]
    umami_featuremap=visual_model.predict(unshuffle_umamidata)[2]

    return bitter_featuremap,umami_featuremap

