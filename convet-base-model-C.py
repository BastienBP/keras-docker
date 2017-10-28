import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import *
from keras.metrics import *

# Setting callback report & save best report
conv_name = "Convet-Model-C_96-kernel(3*3)-relu_96-kernel(3*3)-relu-stride-2-MaxPool(3*3)_192-kernel(3*3)-relu_192-kernel(3*3)-relu-stride-2_MaxPool(3*3)_192-kernel(3*3)_192-kernel(1*1)_10-kernel(1*1)_global-average-pool_10-softmax"
logdir = "/home/models-logs/convnet/" # A changer vers ton repo de log pour tensorboard
tensorboard_callback = TensorBoard(logdir + conv_name, write_graph=True)
checkpoint = ModelCheckpoint('best-model-Conv-base-model-C.h5', monitor='val_loss', verbose=0, save_best_only= True, mode='auto')

# import CIFAR-10 dataset
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# model build

model = Sequential()

model.add(Conv2D(96, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))

model.add(Conv2D(96, (3, 3), activation='relu', padding='same' , strides = 2))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Dropout(0.25))


model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(192, (3, 3), activation='relu', padding='same', strides=2))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(192, (1, 1), activation='relu', padding='valid'))

model.add(Conv2D(10, (1, 1), activation='relu', padding='valid'))

model.add(GlobalAveragePooling2D())

model.add(Dense(10, activation='softmax'))


# Print model architecture

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics= [categorical_accuracy])

model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data= (x_test, y_test), callbacks=[tensorboard_callback, checkpoint])


# Evaluate the model
scores = model.evaluate(x_test, y_test, verbose = 0)
print('Final accuracy:', str(scores[1] * 100), '%')

# Model results analysis
prediction = model.predict(x_test, batch_size=32)
label_pred = np.argmax(prediction, axis=1)

correct = (label_pred == y_test)
print('Number of correct classification : ', str(sum(correct)))


# Show mis-classifications
incorrect = (correct == False)

# get image with wrong classification
mis_image = x_test[incorrect]

# get wrong prediction for those images
labels_err = label_pred[incorrect]

# get the true label for those images
labels_true = y_test[incorrect]

#from matplotlib import pyplot as plt

#def plot_images(images, labels_true, class_names, labels_pred=None):

    #Create a figure with sub-plots
 #   fig, axes = plt.subplots(3, 3, figsize = (8, 8))

    #Adjust the vertical spacing
   # if label_pred is None:
   #     hspace = 0.2
   # else:
  #      hspace = 0.5

  #  fig.subplots_adjust(hspace = hspace, wspace = 0.3)

  #  for i, ax in enumerate(axes.flat):

        #for now don't crash when less than 9 images
      #  if i < len(images):
            #Plot the image
          #  ax.imshow(images[i], interpolation='spline16')

            #Name of the correct class
          #  labels_true_name = class_names[labels_true[i]]

            # Show true and predicted classes
         #   if labels_pred in None:
       #             xlabel = "Correct: " + labels_true_name
#            else:
                # Name of predicted class
#                labels_pred_name = class_names[labels_pred[i]]
#                xlabel = "Correct: " + labels_true_name + "\nPredicted : " + labels_pred_name

            #Show the class on the x axis
#            ax.set_xlabel(xlabel)

        # Remove ticks from the plot
#        ax.set_xticks([])
#        ax.set_yticks([])

    # Show the plot
 #   plt.show()
