
from  Utils import *
from MyModels import *
from sklearn.model_selection import train_test_split
import numpy as np
from random import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from matplotlib import pyplot as plt
from datetime import datetime
now = datetime.now() # current date and time

n_classes = 7
classNames = ['CoV-2 (B)', 'CoV-2 (B.1.1.7)', 'CoV-2 (B.1.351)', 'CoV-2 (B.1.617.2)', 'CoV-2 (C.37)', 'CoV-2 (P.1)', 'CoV-2 (B.1.525)']


############ nucleotide seq data

# all_data_class1 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B)\ncbi_dataset\data\genomic.fna',0)
# all_data_class2 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.1.7)\ncbi_dataset\data\genomic.fna',1)
# all_data_class3 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.351)\ncbi_dataset\data\genomic.fna',2)
# all_data_class4 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.617.2)\ncbi_dataset\data\genomic.fna',3)
# all_data_class5 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (C.37)\ncbi_dataset\data\genomic.fna',4)
# all_data_class6 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (P.1)\ncbi_dataset\data\genomic.fna',5)
# all_data_class7 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.525)\ncbi_dataset\data\genomic.fna',6)


########### Data used for seq generation Test
# all_data_class8 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (C.1.2)\ncbi_dataset\data\genomic.fna',7)
# all_data_class9 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (P.2)\ncbi_dataset\data\genomic.fna',8)
# all_data_class10 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.427)\ncbi_dataset\data\genomic.fna',9)



############ protein seq data

all_data_class1 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B)\ncbi_dataset\data\protein.faa',0)
all_data_class2 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.1.7)\ncbi_dataset\data\protein.faa',1)
all_data_class3 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.351)\ncbi_dataset\data\protein.faa',2)
all_data_class4 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.617.2)\ncbi_dataset\data\protein.faa',3)
all_data_class5 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (C.37)\ncbi_dataset\data\protein.faa',4)
all_data_class6 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (P.1)\ncbi_dataset\data\protein.faa',5)
all_data_class7 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.525)\ncbi_dataset\data\protein.faa',6)

########### Data used for seq generation Test
# all_data_class8 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (C.1.2)\ncbi_dataset\data\protein.faa',7)
# all_data_class9 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (P.2)\ncbi_dataset\data\protein.faa',8)
# all_data_class10 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.427)\ncbi_dataset\data\protein.faa',9)



all_data=[]
for itm in all_data_class1:
    all_data.append(itm)
for itm in all_data_class2:
    all_data.append(itm)
for itm in all_data_class3:
    all_data.append(itm)
for itm in all_data_class4:
    all_data.append(itm)
for itm in all_data_class5:
    all_data.append(itm)
for itm in all_data_class6:
    all_data.append(itm)
for itm in all_data_class7:
    all_data.append(itm)
shuffle(all_data)

x=[]
y=[]
for itm in all_data:
    x.append(itm[0])
    y.append(np.array(itm[1])) 



print(len(x), len(y))

x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.2)
x_train=np.asarray(x_train,dtype=np.float)
x_test=np.asarray(x_test,dtype=np.float)
y_train=np.asarray(y_train)
y_test=np.asarray(y_test)
encoded = to_categorical([y_train])
y_train = np.squeeze(encoded)
encoded = to_categorical([y_test])
y_test = np.squeeze(encoded)



model = Conv1DModel()
model.build(x_train.shape)
print (model.summary())
opt = Adam(lr=0.0003,amsgrad = True)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=32,epochs=100,shuffle=True, validation_split=0.3)
y_scores= model.predict(x_test)


# filename = 'nucleotide'
filename = 'protein'

plot_Acc_Loss(history, filename)
plot_ROC(y_test,y_scores, classNames, filename)
plot_confusion_matrix(y_test,y_scores, classNames, filename)


# fileName = './Classification_nucleotide_Model'
fileName = './Classification_pro_Model'
print(fileName)
model.save(fileName,save_format='tf')
