from numpy.testing._private.utils import print_assert_equal
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling1D, Convolution2D
from tensorflow.keras.layers import BatchNormalization, MaxPooling1D, Conv1D, UpSampling2D, Conv1DTranspose
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.python.keras.backend import dropout

from tensorflow.python.keras.layers.advanced_activations import Softmax
from  Utils import *
from random import shuffle
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
import tensorflow as tf




n_classes = 6
classNames = ['CoV-2 (B)', 'CoV-2 (B.1.1.7)', 'CoV-2 (B.1.351)', 'CoV-2 (B.1.617.2)', 'CoV-2 (C.37)', 'CoV-2 (P.1)']
all_data_class1 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B)\ncbi_dataset\data\genomic.fna',0)
# all_data_class2 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B.1.1.7)\ncbi_dataset\data\genomic.fna',1)
# all_data_class3 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B.1.351)\ncbi_dataset\data\genomic.fna',2)
# all_data_class4 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B.1.617.2)\ncbi_dataset\data\genomic.fna',3)
# all_data_class5 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (C.37)\ncbi_dataset\data\genomic.fna',4)
# all_data_class6 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (P.1)\ncbi_dataset\data\genomic.fna',5)
# all_data_class7 = read_seq_new(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\GRCh38_latest_genomic.fna\GRCh38_latest_genomic.fna',6)

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
# for itm in all_data_class7:
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



# Set the dimensions of the noise
z_dim = 100

nch = 500

# # Generator
# adam = Adam(lr=0.0002, beta_1=0.5)

# g = Sequential()
# g.add(Dense(nch*3*4, input_dim=z_dim))
# g.add(Reshape((nch*3, 4)))
# g.add(BatchNormalization())
# g.add(Activation(LeakyReLU(alpha=0.2)))
# g.add(UpSampling1D(size=2))
# g.add(Conv1D(int(nch/8),4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
# g.add(BatchNormalization())
# g.add(Activation(LeakyReLU(alpha=0.2)))
# g.add(Conv1D(4,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
# g.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# g.summary()

# d = Sequential()
# d.add(Conv1D(int(nch/4),4,padding='same', input_shape=(3000, 4), activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
# d.add(Conv1D(int(nch/8),4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
# d.add(BatchNormalization())
# d.add(Activation(LeakyReLU(alpha=0.2)))
# d.add(Conv1D(int(nch/10),4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
# d.add(Activation(LeakyReLU(alpha=0.2)))
# d.add(Flatten())
# d.add(Dense(112, activation=LeakyReLU(alpha=0.2)))
# d.add(Dense(1, activation='sigmoid'))
# d.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# d.summary()

# d.trainable = False
# inputs = Input(shape=(z_dim, ))
# hidden = g(inputs)
# output = d(hidden)
# gan = Model(inputs, output)
# gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# gan.summary()


# Set the dimensions of the noise
z_dim = 100

nch = 100

# Generator
adam = Adam(lr=0.0002, beta_1=0.5)

g = Sequential()
g.add(Dense(nch*4, input_dim=z_dim))
g.add(Reshape((nch, 4)))
g.add(BatchNormalization())
g.add(UpSampling1D(2))
g.add(Conv1D(128,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
g.add(Conv1D(256,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
g.add(BatchNormalization())
g.add(UpSampling1D(5))
g.add(Conv1D(256,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
g.add(Conv1D(128,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
g.add(Conv1D(128,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
g.add(BatchNormalization())
g.add(UpSampling1D(10))
g.add(Conv1D(256,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
g.add(Conv1D(128,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
g.add(Conv1D(64,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
g.add(BatchNormalization())
g.add(UpSampling1D(3))
g.add(Dropout(0.5))
g.add(Conv1D(32,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
g.add(Conv1D(16,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
g.add(Conv1D(4,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
g.add(Softmax(axis=2))
g.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
g.summary()

d = Sequential()
d.add(Conv1D(128,4,padding='same', input_shape=(30000, 4), activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
d.add(MaxPooling1D(2))
d.add(Conv1D(64,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
d.add(BatchNormalization())
d.add(MaxPooling1D(2))
d.add(Conv1D(32,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
d.add(MaxPooling1D(2))
d.add(Conv1D(32,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
d.add(MaxPooling1D(2))
d.add(Conv1D(16,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
d.add(MaxPooling1D(2))
d.add(Conv1D(8,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform()))
d.add(Flatten())
d.add(Dropout(0.5))
d.add(Dense(112, activation=LeakyReLU(alpha=0.2)))
d.add(Dense(1, activation='sigmoid'))
d.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
d.summary()

d.trainable = False
inputs = Input(shape=(z_dim, ))

hidden = g(inputs)
output = d(hidden)
gan = Model(inputs, output)
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
gan.summary()



def plot_loss(losses):
    """
    @losses.keys():
        0: loss
        1: accuracy
    """
    d_loss = [v[0] for v in losses["D"]]
    g_loss = [v[0] for v in losses["G"]]
    
    plt.figure(figsize=(10,8))
    plt.plot(d_loss, label="Discriminator loss")
    plt.plot(g_loss, label="Generator loss")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    
    plt.savefig('DisGenLoss.png')

    plt.figure(figsize=(10,8))
    plt.plot(d_loss, label="Discriminator loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('DisLoss.png')

    plt.figure(figsize=(10,8))
    plt.plot(g_loss, label="Generator loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('GenLoss.png')
    
# def plot_generated(n_ex=10, dim=(1, 10), figsize=(12, 2)):
#     noise = np.random.normal(0, 1, size=(n_ex, z_dim))
#     generated_seq = g.predict(noise)
#     generated_seq = generated_seq.reshape(generated_seq.shape[0], 3000, 4)
#     plt.figure(figsize=figsize)
#     for i in range(generated_seq.shape[0]):
#         plt.subplot(dim[0], dim[1], i+1)
#         plt.imshow(generated_seq[i, :, :], interpolation='nearest', cmap='gray_r')
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()

# Set up a vector (dict) to store the losses
losses = {"D":[], "G":[]}
samples = []

def train(epochs=100, plt_frq=1, BATCH_SIZE=10):
    batchCount = int(x_train.shape[0] / BATCH_SIZE)
    print('Epochs:', epochs)
    print('Batch size:', BATCH_SIZE)
    print('Batches per epoch:', batchCount)
    
    for e in range(1, epochs+2):
        if e == 1 or e%plt_frq == 0:
            print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batchCount)):  # tqdm_notebook(range(batchCount), leave=False):
            # Create a batch by drawing random index numbers from the training set
            Orignal_seq = x_train[np.random.randint(0, x_train.shape[0], size=BATCH_SIZE)]
            Orignal_seq = Orignal_seq.reshape(Orignal_seq.shape[0], Orignal_seq.shape[1], Orignal_seq.shape[2])
            # Create noise vectors for the generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            
            # Generate the images from the noise
            generated_seq = g.predict(noise)

            # generated_seq = generated_seq[np.where(generated_seq==np.max(generated_seq, axis=2))] = 1 
            for i in range(generated_seq.shape[0]):
                gg = generated_seq[i,:,:]
                b = np.zeros_like(gg)
                b[np.arange(len(gg)), gg.argmax(1)] = 1
                generated_seq[i,:,:] = b

            # print(generated_seq.shape, generated_seq[0,0,:])


            samples.append(generated_seq)

            
            saveGeneratedSeq(generated_seq[0,:,:])


            X = np.concatenate((Orignal_seq, generated_seq))
            # Create labels
            y = np.zeros(2*BATCH_SIZE)
            y[:BATCH_SIZE] = 1  # One-sided label smoothing

            # Train discriminator on generated images
            d.trainable = True
            d_loss = d.train_on_batch(X, y)

            # Train generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            y2 = np.ones(BATCH_SIZE)
            d.trainable = False
            g_loss = gan.train_on_batch(noise, y2)

        # Only store losses from final batch of epoch
        print("Epoch", e, "Generator Loss", g_loss)
        print("Epoch", e, "Discrminator loss", d_loss)
        losses["D"].append(d_loss)
        losses["G"].append(g_loss)

        # Update the plots
        # if e == 1 or e%plt_frq == 0:
        #     plot_generated()
    plot_loss(losses)

train()