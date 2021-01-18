import tensorflow as tf
from tensorflow.keras import layers, models, utils
# Helper libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

target_trainset = 'train_images__224x224.npy'
target_testset = 'test_images__224x224.npy'
resize_to = 224

### import data
train_images = np.load(target_trainset)
print('Shape of train_images: ', train_images.shape)
train_labels = np.load('train_labels.npy')
train_labels = utils.to_categorical(list(train_labels))
print('Shape of train_labels: ', train_labels.shape)
print(train_labels[0], train_labels[-1])
test_images = np.load(target_testset)
print('Shape of test_images: ', test_images.shape)


### split val data
training_idx = np.random.randint(train_images.shape[0], size=int(train_images.shape[0]*0.8))
val_idx = np.random.randint(train_images.shape[0], size=int(train_images.shape[0]-int(train_images.shape[0]*0.8)))
training_x, val_x = train_images[training_idx,:], train_images[val_idx,:]
training_y, val_y = train_labels[training_idx], train_labels[val_idx]

### reshape
training_x = training_x.reshape(training_x.shape[0], resize_to, resize_to, 1)
val_x = val_x.reshape(val_x.shape[0], resize_to, resize_to, 1)
test_images = test_images.reshape(test_images.shape[0], resize_to, resize_to, 1)
print('Shape of training_x: ', training_x.shape)
print('Shape of val_y: ', val_y.shape)


### model build
#### reference: https://blog.gtwang.org/programming/tensorflow-core-keras-api-cnn-tutorial/
input_shape = (resize_to, resize_to, 1)
num_classes = 42
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])


### model train
history = model.fit(training_x, training_y,
          batch_size=32,
          epochs=20,
          verbose=1,
          validation_data=(val_x, val_y))


### validation set performance plot
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
val_loss, val_acc = model.evaluate(val_x, val_y, verbose=2)
print('\nvalidated accuracy:', val_acc)


### prediction
#probability_model = tf.keras.Sequential([model,
#                                         tf.keras.layers.Softmax()])
predictions = model.predict(test_images)
print('first prediction_proba: ', predictions[0])
print('first prediction: ', np.argmax(predictions[0]))
predictions_digit = [np.argmax(x) for x in predictions]
df_submit = pd.read_csv('dims_records_all_test_pic.csv')
df_submit['category'] = predictions_digit
df_submit.drop(['height','width'], axis=1, inplace=True)
print(df_submit.head())
df_submit.to_csv('submission_200704_5.csv', index=False)


