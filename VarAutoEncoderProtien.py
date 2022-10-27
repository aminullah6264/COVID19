
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


n_classes = 7
classNames = ['CoV-2 (B)', 'CoV-2 (B.1.1.7)', 'CoV-2 (B.1.351)', 'CoV-2 (B.1.617.2)', 'CoV-2 (C.37)', 'CoV-2 (P.1)', 'Normal']
all_data_class1 = read_seq_new_pro(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B)\ncbi_dataset\data\protein.faa',0)
# all_data_class2 = read_seq_new_pro(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B.1.1.7)\ncbi_dataset\data\protein.faa',1)
# all_data_class3 = read_seq_new_pro(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B.1.351)\ncbi_dataset\data\protein.faa',2)
# all_data_class4 = read_seq_new_pro(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (B.1.617.2)\ncbi_dataset\data\protein.faa',3)
# all_data_class5 = read_seq_new_pro(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (C.37)\ncbi_dataset\data\protein.faa',4)
# all_data_class6 = read_seq_new_pro(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Dataset\CovidVariantsDataset\SARS-CoV-2 (P.1)\ncbi_dataset\data\protein.faa',5)

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

x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.01)
x_train=np.asarray(x_train,dtype=np.float)
x_test=np.asarray(x_test,dtype=np.float)
y_train=np.asarray(y_train)
y_test=np.asarray(y_test)
encoded = to_categorical([y_train])
y_train = np.squeeze(encoded)
encoded = to_categorical([y_test])
y_test = np.squeeze(encoded)

print(x_train.shape)

saveGeneratedSeq_Pro(x_train.reshape(-1,300,26))



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
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=-1
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
print(x_train.shape)
x_train = x_train.reshape(-1,7800)

print(x_train.shape)


vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())


vae.fit(x_train, epochs=1, batch_size=128)

# noise = np.random.randn(100, latent_dim)


# gen_seqs= vae.decoder.predict(noise)
gen_seqs = []
for i in range(100):
    noise = np.random.randn(1, latent_dim)
    gen_seqs.append(vae.decoder.predict(noise))
gen_seqs = np.array(gen_seqs)
saveGeneratedSeq_Pro(gen_seqs.reshape(-1,300,26))
# vae.save(r'D:\3. Imam University Research\COVID-19 RNA Analysis\Code\COVID19-CNN-LSTM\COVID19VariantClassification\VAEseqGen_Pro',save_format='tf')





