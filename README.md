# UTK-Face

## Deep Learning training for age and gender detection on the UTK-Face dataset

### Summary

| Section | Description |
| ------- | ----------- |
| 1. Project Description | Quick introduction |
| 2. File Structure | Dataset to download & structure |
| 3. Gender Model | Describes this classification model |
| 4. Age Model | Describes this regression model |
| 5. Epochs | How to change the epochs parameter |
| 6. Predictions | How to choose which model to use |

---
### 1. Project Description
These 2 models were engineered to train for **age** and **gender** detection over **23,000 images**.

The outputs are predicted with **87% accuracy**.

---
### 2. File Structure
You can download the dataset [here](https://susanqq.github.io/UTKFace/).
I have personally downloaded the Aligned&Cropped Faces one.

Once the dataset downloaded, you just have to put the data in a directory called UTKFace so you have this:
<p align="center">
  <img width="228" alt="image" src="https://github.com/MiloFournier/UTK-Face/assets/132404970/365566d4-ad87-4d63-93d5-ee5b55f0355b">
</p>

---
### 3. Gender Model
This model mainly uses Conv2D, ReLU and MaxPool2D. I started with 36 nodes:
```py
gender_model.add(Conv2D(36, kernel_size=3, activation="relu", input_shape=(200, 200, 3)))
gender_model.add(MaxPool2D(pool_size=3, strides=2))
```
and ended with 512 nodes:
```py
gender_model.add(Conv2D(512, kernel_size=3, activation="relu"))
gender_model.add(MaxPool2D(pool_size=3, strides=2))
```
I drop 20% of the input to avoid overfitting:
```py
gender_model.add(Dropout(0.2))
gender_model.add(Dense(512, activation="relu"))
```
Being a binary classification problem, I use a sigmoid activation layer with a binary cross-entropy loss function:
```py
gender_model.add(Dense(1, activation="sigmoid", name="gender"))
gender_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

---
### 4. Age Model
This model also mainly uses Conv2D, ReLU and MaxPool2D. I started with 36 nodes:
```py
age_model.add(Conv2D(128, kernel_size=3, activation="relu", input_shape=(200, 200, 3)))
age_model.add(MaxPool2D(pool_size=3, strides=2))
```
and ended with 512 nodes:
```py
age_model.add(Conv2D(512, kernel_size=3, activation="relu"))
age_model.add(MaxPool2D(pool_size=3, strides=2))
```
After dropping 20% of the input, I use the mean squared error loss function because this time, it is a regression problem:
```py
age_model.add(Dense(1, activation="linear", name="age"))
age_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
```

---
### 5. Epochs
As of now, the model only runs the gender model with 50 epochs. It is possible to change these parameters here:
```py
run_gender_model(nb_epochs=50, run=True)
run_age_model(nb_epochs=50, run=False)
```

---
### 6. Predictions
So we don't always have to run the models, they are saved in "gender_model.keras" or "age_model.keras".

You can then choose the model from which you want to predict outputs:
```py
my_model = load_model("gender_model.keras", compile=False)
# my_model = load_model("age_model.keras", compile=False)
```
