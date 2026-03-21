import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

#-----
#Data Preparation
#-----

#path to split data
BASE_DIR = 'Tomato_Leaf_Data_Split'

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR,'val')
TEST_DIR = os.path.join(BASE_DIR,'test')

#Constant image parameters
IMG_SIZE = (128,128)
BATCH_SIZE = 32

#Normalising pixel values (0-255 -> 0-1)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,       
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    shuffle = False
)


#----
#Building the CNN model version - 001
#----

model = models.Sequential([
    layers.Conv2D(32, (3,3,), activation= 'relu', input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(128, activation= 'relu'),
    layers.Dropout(0.5),

    layers.Dense(train_generator.num_classes, activation= 'softmax')
])


#----
#Compiling the model
#----

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']

)

#----
#Training the model
#----

history = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs = 5
)

#----
#Evaluating the model on the test set
#----

test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


#----
#Saving the model
#----

model.save('tomato_leaf_disease_model.h5')
print("Model saved as 'tomato_leaf_disease_model.h5'")

print(train_generator.class_indices)