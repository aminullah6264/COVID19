
from operator import xor
from  Utils import *
from MyModels import *
from sklearn.model_selection import train_test_split
import numpy as np
from random import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from matplotlib import pyplot as plt
import tensorflow as tf

from tensorflow.keras import backend as K
from keras.callbacks import History 
history = History()

############ protein seq data

all_data_class1 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B)\ncbi_dataset\data\protein.faa',0)
all_data_class2 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.1.7)\ncbi_dataset\data\protein.faa',1)
# all_data_class2 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.351)\ncbi_dataset\data\protein.faa',2)
# all_data_class2 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.617.2)\ncbi_dataset\data\protein.faa',3)
# all_data_class2 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (C.37)\ncbi_dataset\data\protein.faa',4)
# all_data_class2 = read_seq_new_pro(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (P.1)\ncbi_dataset\data\protein.faa',5)

x=[]
y=[]
for itm in all_data_class1:
    x.append(itm[0])

for itm in all_data_class2:
    y.append(itm[0])

x_train=np.asarray(x, dtype=np.float)
y_train=np.asarray(y, dtype=np.float)

if y_train.shape[0] > x_train.shape[0]:
    y_train= y_train[0:x_train.shape[0],:,:]
else:
    x_train= x_train[0:y_train.shape[0],:,:]
print(x_train.shape, y_train.shape)





class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 350

encoder_inputs = tf.keras.layers.Input(shape=(7800,))
x = tf.keras.layers.Dense(6000, activation='relu')(encoder_inputs)
x = tf.keras.layers.Dense(3000, activation='relu')(x)
x = tf.keras.layers.Dense(2000, activation='relu')(x)
x = tf.keras.layers.Dense(1500, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(700, activation='relu')(x)
z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()



latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))
x = tf.keras.layers.Dense(700, activation='relu')(latent_inputs)
x = tf.keras.layers.Dense(1500, activation='relu')(x)
x = tf.keras.layers.Dense(2000, activation='relu')(x)
x = tf.keras.layers.Dense(3000, activation='relu')(x)
x = tf.keras.layers.Dense(6000, activation='relu')(x)
decoder_outputs = tf.keras.layers.Dense(7800, activation='relu')(x)
decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x , y = data
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(y, reconstruction), axis=-1
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

x_train = x_train.reshape(-1,7800)
y_train = y_train.reshape(-1,7800)



vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())


history = vae.fit(x_train, y_train, epochs=100, batch_size=64,  callbacks=[history])


plot_Gen_Loss(history,'Generator_pro')

gen_seqs = []
for i in range(100):
    noise = np.random.randn(1, latent_dim)
    gen_seqs.append(vae.decoder.predict(noise))
gen_seqs = np.array(gen_seqs)
saveGeneratedSeq_Pro(gen_seqs.reshape(-1,300,26))
fileName = 'VAEseqGen_Pro_Model'
vae.save(fileName,save_format='tf')




