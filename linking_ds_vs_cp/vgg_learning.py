from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.compat.v1.keras import backend as K
import tensorflow.keras.callbacks as cb
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np


class LearningRateDecay:
	def plot(self, epochs, title="Learning Rate Schedule"):
		# compute the set of learning rates for each corresponding
		# epoch
		lrs = [self(i) for i in epochs]
		# the learning rate schedule
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(epochs, lrs)
		plt.title(title)
		plt.xlabel("Epoch #")
		plt.ylabel("Learning Rate")

class PolynomialDecay(LearningRateDecay):
	def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
		# store the maximum number of epochs, base learning rate,
		# and power of the polynomial
		self.maxEpochs = maxEpochs
		self.initAlpha = initAlpha
		self.power = power
	def __call__(self, epoch):
		# compute the new learning rate based on polynomial decay
		decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
		alpha = self.initAlpha * decay
		# return the new learning rate
		return float(alpha)

def save_history(out_file, history):
    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss', fontsize=8)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy', fontsize=8)
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='val')
    pyplot.legend()
    pyplot.savefig(out_file)
    plt.figure().clear()
    pyplot.close()
    pyplot.cla()
    pyplot.clf()
    
    
def vgg16_learning(imgs, labels, samples, batch, epochs, file_name, n_classes=10):   
    # setting learning rate scheduler
    schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-4, power=1)
    callbacks = [LearningRateScheduler(schedule)]

    # defining layers
    input = Input(shape=(imgs.shape[1],imgs.shape[2],imgs.shape[3]),name = 'image_input')
    arch = VGG16(include_top=False, input_shape=(K.int_shape(input)[1],K.int_shape(input)[2], K.int_shape(input)[3]),weights='imagenet')
    
    # transfer learning/fine tuning 
    for layer in arch.layers[:-4]:
        layer.trainable = False

    output_vgg16_conv = arch(preprocess_input(input))
    flat = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(flat)
    x = Dense(4096, activation='relu', name='fc2')(x)
    output = Dense(n_classes, activation='softmax', name='predictions')(x)

    # defining model end encoded model
    model = Model(input, output)
    enc_model = Model(input, flat) 

    # compiling and training the model
    sgd = optimizers.SGD(lr=1e-1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    history = model.fit(imgs[samples], to_categorical(labels[samples],n_classes), batch_size=batch, epochs=epochs, verbose=0, validation_split=0.2, shuffle=True, callbacks=callbacks)
    save_history(file_name, history)

    class_labels = model.predict(imgs)
    class_labels = model.predict(imgs)

    return class_labels

