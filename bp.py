import tensorflow as tf
import keras as kr
from keras.src.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from sklearn import metrics

#GPU
NAME = "Sigmoid"
gpu_list = tf.config.list_physical_devices('GPU')

print("Num GPUs Available: ", len(gpu_list))
#tf.debugging.set_log_device_placement(True)

tf.config.experimental.set_memory_growth(gpu_list[0], True)

IMAGE_WIDTH = 60
IMAGE_HEIGHT = 60
BATCH_SIZE = 16

#            0        1       2       3        4        5
CLASSES = ["down", "front", "left", "none", "right", "up"]
NUM_OF_CLASSES = len(CLASSES)
    
'''
MODEL
'''
model = kr.Sequential()
model.add(kr.layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))

model.add(kr.layers.Rescaling(1./255)) #Normalizacia

model.add(kr.layers.Conv2D(filters=1024, kernel_size=3, activation='relu', padding='same'))

model.add(kr.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
model.add(kr.layers.MaxPool2D((2, 2)))
model.add(kr.layers.Dropout(0.1))

model.add(kr.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))

model.add(kr.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(kr.layers.MaxPool2D((2, 2)))
model.add(kr.layers.Dropout(0.1))

model.add(kr.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(kr.layers.MaxPool2D((2, 2)))
model.add(kr.layers.Dropout(0.1))

model.add(kr.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(kr.layers.MaxPool2D((2, 2)))
model.add(kr.layers.Dropout(0.1))

model.add(kr.layers.Flatten())

model.add(kr.layers.Dense(units=32, activation='relu'))
model.add(kr.layers.Dense(units=16, activation='relu'))
model.add(kr.layers.Dense(units=NUM_OF_CLASSES, activation='softmax'))

model.compile(
    loss=kr.losses.SparseCategoricalCrossentropy(),
    optimizer=kr.optimizers.Adam(),
    metrics=['accuracy']
)

model.summary()

'''
DATASET
'''

dataset_path = "./organized_images"

train_dataset:tf.data.Dataset = kr.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset:tf.data.Dataset = kr.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

weights_filepath = NAME + "-model.weights.h5"
weights_save = ModelCheckpoint(
    filepath=weights_filepath,
    save_best_only=True,
    save_weights_only=True,
    monitor='val_accuracy',
    save_freq='epoch',
    mode='max',
    verbose=1
)

early_stopping = kr.callbacks.EarlyStopping(monitor="val_loss", patience=10)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=100,
    verbose=1,
    callbacks=[weights_save]
)

model.save(NAME + "-whole_model.keras")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['training', 'validation'], loc='upper left')
plt.savefig(NAME + "-history.jpg")
plt.show()

y_= model.predict(validation_dataset)
y_pred = tf.argmax(y_, axis=1)
y_true = tf.concat([y for x, y in validation_dataset], axis=0)
con_mat = metrics.confusion_matrix(y_true, y_pred)
cm_plt = metrics.ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=CLASSES)
cm_plt.plot()
plt.savefig(NAME + "-confusion_matrix.jpg")
plt.show()
