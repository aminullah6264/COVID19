import tensorflow as tf
from  Utils import *
import numpy as np
from random import shuffle
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split



# Recreate the exact same model purely from the file
new_model = tf.keras.models.load_model(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Code\COVID19-CNN-LSTM\COVID19VariantClassification\my_CNN1D7ClassModel')


n_classes = 7
classNames = ['CoV-2 (B)', 'CoV-2 (B.1.1.7)', 'CoV-2 (B.1.351)', 'CoV-2 (B.1.617.2)', 'CoV-2 (C.37)', 'CoV-2 (P.1)', 'CoV-2 (B.1.525)']

all_data_class1 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\SARS-CoV-2 (B.1.525)\ncbi_dataset\data\genomic.fna',0)
# all_data_class1 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B)\ncbi_dataset\data\genomic.fna',0)
all_data=[]
for itm in all_data_class1:
    all_data.append(itm)

x=[]
y=[]
for itm in all_data:
    x.append(itm[0])
    y.append(np.array(itm[1])) 

x=np.asarray(x,dtype=np.float)

y_scores= new_model.predict(x)

ent1 = cal_entropy(y_scores)

all_data_class1 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B.1.1.7)\ncbi_dataset\data\genomic.fna',1)
all_data=[]
for itm in all_data_class1:
    all_data.append(itm)

x=[]
y=[]
for itm in all_data:
    x.append(itm[0])
    y.append(np.array(itm[1])) 

x=np.asarray(x,dtype=np.float)

y_scores= new_model.predict(x)

ent2 = cal_entropy(y_scores)

aa = ent1 > 0.15
numberss = np.count_nonzero(aa)
print(numberss)

aa = ent2 > 0.15
numberss = np.count_nonzero(aa)
print(numberss)

import matplotlib.pyplot as plt

x = np.arange(0,len(ent1),1)


plt.scatter(x, ent1, label="Eta")
plt.scatter(x, ent2, label="Alpha")
plt.xlabel('Test Samples')
plt.ylabel('Uncertainty Value')
plt.legend(loc="upper left")
plt.show()


