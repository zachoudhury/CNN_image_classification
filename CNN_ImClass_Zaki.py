#Develop CNN Model for identifying cracks on road pavement using TensorFlow


import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # This line was added to avoid an error that resulted in restarting kernel
os.chdir("E:/Machine Learning/AssignImClass")

# Avoid OOM errors by setting GPU Memory Consumption Growth when GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')


for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)#limit memory growth

print(tf.config.list_physical_devices('GPU'))


#Load Data
import numpy as np
from matplotlib import pyplot as plt

data = tf.keras.utils.image_dataset_from_directory('PhotosCrack', batch_size=32, shuffle=True, image_size=(256, 256))
#'PhotosCrack' was the name of the main image data folder that contained 
#two image folders with labeled images
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()


#Scale Data

data = data.map(lambda x,y: (x/255, y))

data.as_numpy_iterator().next()

#Split Data
train_size = int(len(data)*.75)
val_size = int(len(data)*.15)
test_size = int(len(data)*.10)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#Build Deep Learning Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(padding='same'))
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D(padding='same'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()

#Train and fit
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

#Plot Performance
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#Evaluate the model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()},Accuracy:{acc.result().numpy()}')

#Test model using an image
import cv2
img = cv2.imread('cr1.jpg')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))

print(yhat)

if yhat > 0.5: 
    print("Predicted class is Yes")
else:
    print("Predicted class is No")

#Save the Model

model.save(os.path.join('model1','imageclassifier.h5'))

#Load model to classify image
from tensorflow.keras.models import load_model
new_model = load_model('model1/imageclassifier.h5')
new_model.predict(np.expand_dims(resize/255, 0))
#here the image tested is the image in "#Test model using an image" section



