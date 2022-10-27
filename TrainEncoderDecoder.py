
from  Utils import *
from MyModels import *
from sklearn.model_selection import train_test_split
import numpy as np
from random import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from matplotlib import pyplot as plt

from tensorflow.keras import backend as K


n_classes = 7
classNames = ['CoV-2 (B)', 'CoV-2 (B.1.1.7)', 'CoV-2 (B.1.351)', 'CoV-2 (B.1.617.2)', 'CoV-2 (C.37)', 'CoV-2 (P.1)', 'Normal']
all_data_class1 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B)\ncbi_dataset\data\genomic.fna',0)
all_data_class2 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B.1.1.7)\ncbi_dataset\data\genomic.fna',1)
all_data_class3 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B.1.351)\ncbi_dataset\data\genomic.fna',2)
all_data_class4 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B.1.617.2)\ncbi_dataset\data\genomic.fna',3)
all_data_class5 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (C.37)\ncbi_dataset\data\genomic.fna',4)
all_data_class6 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (P.1)\ncbi_dataset\data\genomic.fna',5)

all_data=[]
for itm in all_data_class1:
    all_data.append(itm)
# for itm in all_data_class2:
#     all_data.append(itm)
# for itm in all_data_class3:
#     all_data.append(itm)
# for itm in all_data_class4:
#     all_data.append(itm)
# for itm in all_data_class5:
#     all_data.append(itm)
# for itm in all_data_class6:
#     all_data.append(itm)

shuffle(all_data)

x=[]
y=[]
for itm in all_data:
    x.append(itm[0])
    y.append(np.array(itm[1])) 

x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.2)
x_train=np.asarray(x_train,dtype=np.float)
x_test=np.asarray(x_test,dtype=np.float)
y_train=np.asarray(y_train)
y_test=np.asarray(y_test)
encoded = to_categorical([y_train])
y_train = np.squeeze(encoded)
encoded = to_categorical([y_test])
y_test = np.squeeze(encoded)


inputs2 = tf.keras.layers.Input(shape=(12000, ))
model = EncoderDecoder()
model.build(inputs2.shape)
print (model.summary())
opt = Adam(lr=0.0001, amsgrad = True)
loss1 = 'binary_crossentropy'
loss2 = 'mean_squared_error'

def hammingloss(y_true,y_pred):
    return K.mean(y_true*(1-y_pred)+(1-y_true)*y_pred)

# def hammingloss1(y_true,y_pred):

#     tmp = K.abs(y_true-y_pred)
#     return K.mean(K.cast(K.greater(tmp,0.5),dtype=float))


model.compile(loss=loss2, optimizer=opt)
history = model.fit(x_train.reshape(-1,12000), x_train.reshape(-1,12000) ,batch_size=10,epochs=100,shuffle=True, validation_split=0.3)
y_scores= model.predict(x_test.reshape(-1,12000))
model.save(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Code\COVID19-CNN-LSTM\COVID19VariantClassification\my_CNN1D7ClassModel',save_format='tf')


saveGeneratedSeq(y_scores.reshape(-1,3000,4))
