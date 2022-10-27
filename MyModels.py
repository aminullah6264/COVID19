import tensorflow as tf
from keras_self_attention import SeqSelfAttention
import numpy as np

class Conv1DModel(tf.keras.Model):

  def __init__(self):
    super(Conv1DModel, self).__init__()
    self.conv1 = tf.keras.layers.Conv1D(128,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
    self.conv2 = tf.keras.layers.Conv1D(64,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
    self.pool1 = tf.keras.layers.MaxPooling1D(4)
    self.conv3 = tf.keras.layers.Conv1D(32,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
    self.pool2 = tf.keras.layers.MaxPooling1D(4)
    self.conv4 = tf.keras.layers.Conv1D(16,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
    self.dense1 = tf.keras.layers.Dense(7, activation=tf.nn.softmax)

  def call(self, inputs, training=False):
    x = self.conv1(inputs)
    x = self.conv2(x)
    x = self.pool1(x)
    x = self.conv3(x)
    x = self.pool2(x)
    x = self.conv4(x)
    x = tf.keras.layers.Attention()([x,x])
    x = tf.keras.layers.Flatten()(x)
    if training:
        x = tf.keras.layers.Dropout(0.5)(x)
    x = self.dense1(x)
    return x


class LSTMModel(tf.keras.Model):

  def __init__(self):
    super(LSTMModel, self).__init__()
    self.LSTM1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))
    self.LSTM2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))
    self.dense1 = tf.keras.layers.Dense(7, activation=tf.nn.softmax)

  def call(self, inputs, training=False):
    x = tf.reshape(inputs, [-1, 30, 100*4])
    x = self.LSTM1(x)
    x = self.LSTM2(x)
    x = tf.keras.layers.Attention()([x,x])
    x = tf.keras.layers.Flatten()(x)
    if training:
        x = tf.keras.layers.Dropout(0.5)(x)
    x = self.dense1(x)
    return x

  
class DeeperBind(tf.keras.Model):

  def __init__(self):
    super(DeeperBind, self).__init__()
    self.LSTM1 = tf.keras.layers.LSTM(30, return_sequences=True)
    self.LSTM2 = tf.keras.layers.LSTM(20, return_sequences=True)
    self.dense1 = tf.keras.layers.Dense(7, activation=tf.nn.softmax)

  def call(self, inputs, training=False):
    x = inputs[:,0:600,]
    x = tf.reshape(inputs, [-1, 20, 30*4])
    x = self.LSTM1(x)
    x = self.LSTM2(x)
    x = tf.keras.layers.Flatten()(x)
    if training:
        x = tf.keras.layers.Dropout(0.5)(x)
    x = self.dense1(x)
    return x



class MyGenerator(tf.keras.Model):

  def __init__(self):
    super(MyGenerator, self).__init__()

    self.z_dim = 100
    self.nch = 500


    self.dense1 = tf.keras.layers.Dense(self.nch*3*4, input_dim=100, activation='relu')
    self.upsample1 = tf.keras.layers.UpSampling1D(size=2)
    self.conv1 = tf.keras.layers.Conv1D(128,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
    self.conv2 = tf.keras.layers.Conv1D(4,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
    
    

  def call(self, inputs, training=False):

    x = self.dense1(inputs)
    x = tf.keras.layers.Reshape((self.nch*3, 4))(x)
    x = self.upsample1(x)
    x = self.conv1(x)
    x = self.conv2(x)
    
    return x

class MyDiscriminator(tf.keras.Model):

  def __init__(self):
    super(MyDiscriminator, self).__init__()


    self.conv1 = tf.keras.layers.Conv1D(10,4, input_shape=(3000, 4), padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
    self.pool1 = tf.keras.layers.MaxPooling1D(2)
    self.conv2 = tf.keras.layers.Conv1D(5,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
    self.pool2 = tf.keras.layers.MaxPooling1D(2)
    self.conv3 = tf.keras.layers.Conv1D(4,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())


    self.dense1 = tf.keras.layers.Dense(128, activation='relu')
    self.dense2 = tf.keras.layers.Dense(1, activation='relu')

  def call(self, inputs, training=False):
    
    x = self.conv1(inputs)
    # print(x.shape)
    x = self.pool1(x)
    # print(x.shape)
    x = self.conv2(x)
    # print(x.shape)
    x = self.pool2(x)
    # print(x.shape)
    x = self.conv3(x)
    # print(x.shape)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    # print(x.shape)

    x = self.dense1(x)
    # print(x.shape)
    x = self.dense2(x)
    # print(x.shape)
    
    return x

class EncoderDecoder(tf.keras.Model):

  def __init__(self):
    super(EncoderDecoder, self).__init__()
    self.Edense1 = tf.keras.layers.Dense(9000, activation='relu')
    self.Edense2 = tf.keras.layers.Dense(6000, activation='relu')
    self.Edense3 = tf.keras.layers.Dense(3000, activation='relu')
    self.Edense4 = tf.keras.layers.Dense(1500, activation='relu')
    self.Edense5 = tf.keras.layers.Dense(700, activation='relu')

    self.Ddense1 = tf.keras.layers.Dense(1500, activation='relu')
    self.Ddense2 = tf.keras.layers.Dense(3000, activation='relu')
    self.Ddense3 = tf.keras.layers.Dense(6000, activation='relu')
    self.Ddense4 = tf.keras.layers.Dense(9000, activation='relu')
    self.Ddense5 = tf.keras.layers.Dense(12000, activation='relu')

    


  def call(self, inputs, training=False):
    # x = tf.keras.layers.Flatten()(inputs)
    x = self.Edense1(inputs)
    x = self.Edense2(x)
    x = self.Edense3(x)
    x = self.Edense4(x)
    x = self.Edense5(x)

    if training:
      x = tf.keras.layers.Dropout(0.5)(x)

    x = self.Ddense1(x)
    x = self.Ddense2(x)
    x = self.Ddense3(x)
    x = self.Ddense4(x)
    x = self.Ddense5(x)
    # x = tf.keras.layers.Reshape((3000,4))

    return x


# class EncoderDecoder(tf.keras.Model):

#   def __init__(self):
#     super(EncoderDecoder, self).__init__()
#     self.conv1 = tf.keras.layers.Conv1D(128,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
#     self.pool1 = tf.keras.layers.MaxPooling1D(2)
#     self.conv2 = tf.keras.layers.Conv1D(64,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
#     self.pool2 = tf.keras.layers.MaxPooling1D(2)
#     self.conv3 = tf.keras.layers.Conv1D(32,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
#     self.pool3 = tf.keras.layers.MaxPooling1D(2)
#     self.conv4 = tf.keras.layers.Conv1D(16,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
#     self.dense1 = tf.keras.layers.Dense(7, activation=tf.nn.softmax)


#     self.upsample1 = tf.keras.layers.UpSampling1D(size=2)
#     self.dconv1 = tf.keras.layers.Conv1D(32,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
#     self.dconv2 = tf.keras.layers.Conv1D(64,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
#     self.upsample2 = tf.keras.layers.UpSampling1D(2)
#     self.dconv3 = tf.keras.layers.Conv1D(64,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
#     self.dconv4 = tf.keras.layers.Conv1D(32,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
#     self.upsample3 = tf.keras.layers.UpSampling1D(2)
#     self.dconv5 = tf.keras.layers.Conv1D(16,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
#     self.dconv6 = tf.keras.layers.Conv1D(4,4,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform())
    

#   def call(self, inputs, training=False):
#     x = self.conv1(inputs)
#     x = self.pool1(x)
#     x = self.conv2(x)
#     x = self.pool2(x)
#     x = self.conv3(x)
#     x = self.pool3(x)
#     x = self.conv4(x)
#     # x = tf.keras.layers.Attention()([x,x])

#     # x = tf.keras.layers.Flatten()(x)
    
#     # if training:
#     #     x = tf.keras.layers.Dropout(0.5)(x)
#     # x = self.dense1(x)


#     xx = self.upsample1(x)
#     xx = self.dconv1(xx)
#     xx = self.dconv2(xx)
#     xx = self.upsample2(xx)
#     xx = self.dconv3(xx)
#     xx = self.dconv4(xx)
#     xx = self.upsample3(xx)
#     xx = self.dconv5(xx)
#     xx = self.dconv6(xx)

#     xx = tf.keras.layers.Softmax(axis=-1)(xx)
#     # print(xx[0,0,:])
#     # for i in range(xx.shape[0]):
#     #   gg = xx[i,:,:]
#     #   b = np.zeros_like(gg)
#     #   b[np.arange(len(gg)), gg.argmax(1)] = 1
#     #   xx[i,:,:] = b.dtype('float32')
#     # print(xx.shape)

#     return xx


# model = Conv1DModel()
# model.build((None, 2500,4))
# print (model.summary(line_length=None, positions=None, print_fn=None))



