import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0  
from tensorflow.keras.applications.efficientnet import preprocess_input

import sklearn.metrics as metrics
import numpy as np
import os

#-----
#Data Preparation
#-----

#path to split data
BASE_DIR = r'C:\Users\Menon\OneDrive\Documents\LeafVision-ML\data\Tomato_Leaf_Data_Split'

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR,'val')
TEST_DIR = os.path.join(BASE_DIR,'test')

#Constant image parameters
IMG_SIZE = (128,128)
BATCH_SIZE = 32

#Normalising pixel values (0-255 -> 0-1)
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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


#Base Model - EfficientNetB0

base_model = EfficientNetB0(
    weights = 'imagenet',
    include_top = False,
    input_shape = (128,128,3)
)

base_model.trainable = False



model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation= 'relu'),

    layers.Dropout(0.5),

    layers.Dense(
        train_generator.num_classes,
          activation= 'softmax'
        )
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
    epochs = 10
)

#----
#Evaluating the model on the test set
#----

test_loss, test_accuracy = model.evaluate(test_generator)
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


cm = metrics.confusion_matrix(
    true_classes, predicted_classes
    )

print("Confusion Matrix:")
print(cm)

#----
#Saving the model
#----

model.save('leafvision_efficientnet_v1.keras')
print("Model saved as 'leafvision_efficientnet_v1.keras'")

print(train_generator.class_indices)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


from sklearn.metrics import classification_report

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=test_generator.class_indices.keys()
    )
)



 