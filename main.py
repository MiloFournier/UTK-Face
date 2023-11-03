#######################################################################################################################
# Table of contents:
    # Imports
    # Initialisation
    # Models
    # Predictions
    # Results
#######################################################################################################################

#######################################################################################################################
# Imports
#######################################################################################################################
import os
import cv2
import numpy as np
from sklearn import metrics
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten

#######################################################################################################################
# Initialisation
#######################################################################################################################
# Path to the images
path = "UTKFace"
images  = []
ages = []
genders = []
# For each image in the directory
for image in os.listdir(path):
    # We get the 1st element of the file name -> age
    age = image.split("_")[0]

    # We get the 2nd element of the file name -> gender
    gender = image.split("_")[1]

    # We read the image
    image = cv2.imread(str(path) + "/" + str(image))

    # We add each element to their corresponding array
    ages.append(np.array(age))
    genders.append(np.array(gender))
    images.append(np.array(image))

ages = np.array(ages, dtype=np.int64)
genders = np.array(genders, dtype=np.uint64)
images = np.array(images)

# We then split the data in train & test categories
# One for age and the other for gender because the predicted output is not the same
x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, ages ,random_state=42)
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, genders, random_state=42)

#######################################################################################################################
# Models
#######################################################################################################################
def run_age_model(nb_epochs, run):
    if run:
        # We create the age model
        age_model = Sequential()

        # We start with 2D Convolution, 128 nodes and ReLU activations up to 512 nodes
        age_model.add(Conv2D(128, kernel_size=3, activation="relu", input_shape=(200, 200, 3)))
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Conv2D(128, kernel_size=3, activation="relu"))
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Conv2D(256, kernel_size=3, activation="relu"))
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Conv2D(512, kernel_size=3, activation="relu"))
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Flatten())
        age_model.add(Dropout(0.2))
        age_model.add(Dense(512, activation="relu"))

        # Output is only 1 node (age)
        age_model.add(Dense(1, activation="linear", name="age"))

        # Regression problem -> mse loss
        age_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        print(age_model.summary())

        age_model.fit(x_train_age, y_train_age, validation_data=(x_test_age, y_test_age), epochs=nb_epochs)

        # Save the model so we don't always have to run it
        age.model.save("age_model.keras")

def run_gender_model(nb_epochs, run):
    if run:
        # We create the gender model
        gender_model = Sequential()

        # We start with 2D Convolution, 36 nodes and ReLU activations up to 512 nodes
        gender_model.add(Conv2D(36, kernel_size=3, activation="relu", input_shape=(200, 200, 3)))
        gender_model.add(MaxPool2D(pool_size=3, strides=2))

        gender_model.add(Conv2D(64, kernel_size=3, activation="relu"))
        gender_model.add(MaxPool2D(pool_size=3, strides=2))

        gender_model.add(Conv2D(128, kernel_size=3, activation="relu"))
        gender_model.add(MaxPool2D(pool_size=3, strides=2))

        gender_model.add(Conv2D(256, kernel_size=3, activation="relu"))
        gender_model.add(MaxPool2D(pool_size=3, strides=2))

        gender_model.add(Conv2D(512, kernel_size=3, activation="relu"))
        gender_model.add(MaxPool2D(pool_size=3, strides=2))

        gender_model.add(Flatten())
        gender_model.add(Dropout(0.2))
        gender_model.add(Dense(512, activation="relu"))

        # Output is only 1 node (gender)
        gender_model.add(Dense(1, activation="sigmoid", name="gender"))

        # Classification problem -> binary crossentropy loss
        gender_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        print(gender_model.summary())

        gender_model.fit(
            x_train_gender,
            y_train_gender,
            validation_data=(x_test_gender, y_test_gender),
            epochs=nb_epochs
        )

        # Save the model so we don't always have to run it
        gender_model.save("gender_model.keras")

#######################################################################################################################
# Predictions
#######################################################################################################################
# Here we can run the models
run_gender_model(nb_epochs=50, run=True)
run_age_model(nb_epochs=50, run=False)

# Here we chose which model to predict from
my_model = load_model("gender_model.keras", compile=False)
# my_model = load_model("age_model.keras", compile=False)

# And here we can predict the output
predictions = my_model.predict(x_test_gender)
y_pred = (predictions >= 0.5).astype(int)[:, 0]

#######################################################################################################################
# Results
#######################################################################################################################
# We print the accuracy of the model
print(f"Accuracy = {metrics.accuracy_score(y_test_gender, y_pred)}")
