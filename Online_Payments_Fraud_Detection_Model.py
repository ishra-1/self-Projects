# -*- coding: utf-8 -*-


! pip install -q kaggle
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle competitions download -c spaceship-titanic

! unzip spaceship-titanic.zip

import pandas as pd
df = pd.read_csv('train.csv')
df

df['PassengerGroup'] = df['PassengerId'].str[0:4]
df = df.drop('PassengerId', axis = 1)
df

df = df.drop('Name', axis = 1)
df = df.dropna()
df

df['HomePlanet'].unique()

cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in cols:
  df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
df

df['CabinDeck'] = df['Cabin'].str[0]
df['CabinNum'] = df['Cabin'].str[2:-2:1]
df['CabinSide'] = df['Cabin'].str[-1]
df = df.drop('Cabin', axis = 1)
df

df['Destination_num'] = df['Destination'].astype('category').cat.codes
df['CryoSleep_num'] = df['CryoSleep'].astype('category').cat.codes
df['VIP_num'] = df['VIP'].astype('category').cat.codes
df['HomePlanet_num'] = df['HomePlanet'].astype('category').cat.codes
df['CabinDeck_num'] = df['CabinDeck'].astype('category').cat.codes
df['CabinSide_num'] = df['CabinSide'].astype('category').cat.codes
df = df.drop(['Destination', 'CryoSleep', 'VIP', 'HomePlanet', 'CabinDeck', 'CabinSide'], axis = 1)

cols = ['PassengerGroup', 'CabinNum']
for col in cols:
  df[col] = df[col].astype(int)
  df[col] = ((df[col]) - (df[col].min())) / ((df[col].max()) - (df[col].min()))
df

df["Transported"] = df["Transported"].map({True:1, False:0})
df['Transported'].value_counts()

X = df.drop('Transported', axis = 1)
y = df['Transported']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"""Neural Network"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(14, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])
optim = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy', 'precision'])
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=400, max_depth=8, verbose=1, criterion = 'entropy', min_samples_leaf=4)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
train_random_for = model.predict(X_train)
test_random_for = model.predict(X_test)
train_accuracy = accuracy_score(y_train, train_random_for)
test_accuracy = accuracy_score(y_test, test_random_for)
print('Train accuracy:', train_accuracy)
print('Test accuracy:', test_accuracy)



"""Using XGBoost

"""

from xgboost import XGBClassifier
model = XGBClassifier(
    depth = 200,
    iterations=2000,
    learning_rate=0.2,
    verbose=1,
    reg_alpha = 0.5,
    reg_lambda=1,
    early_stopping_rounds=10
)
model.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)])

print(accuracy_score(y_test, model.predict(X_test)))
print(accuracy_score(y_train, model.predict(X_train)))

"""Roughly same accuracy on test data, choosing neural network model and training on complete input dataset"""

final_model = Sequential([
    Dense(14, activation='relu', input_shape=(X.shape[1],)),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])
optim = tf.keras.optimizers.Adam(learning_rate=0.001)
final_model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy', 'precision'])
final_model.fit(X, y, epochs=26, batch_size=32)

df1 = pd.read_csv('test.csv')

df1['PassengerGroup'] = df1['PassengerId'].str[0:4]
df1 = df1.drop('PassengerId', axis = 1)
df1

cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in cols:
  df1[col] = (df1[col] - df1[col].min()) / (df1[col].max() - df1[col].min())
df1

df1['CabinDeck'] = df1['Cabin'].str[0]
df1['CabinNum'] = df1['Cabin'].str[2:-2:1]
df1['CabinSide'] = df1['Cabin'].str[-1]
df1 = df1.drop('Cabin', axis = 1)
df1

df1['Destination_num'] = df1['Destination'].astype('category').cat.codes
df1['CryoSleep_num'] = df1['CryoSleep'].astype('category').cat.codes
df1['VIP_num'] = df1['VIP'].astype('category').cat.codes
df1['HomePlanet_num'] = df1['HomePlanet'].astype('category').cat.codes
df1['CabinDeck_num'] = df1['CabinDeck'].astype('category').cat.codes
df1['CabinSide_num'] = df1['CabinSide'].astype('category').cat.codes
df1 = df1.drop(['Destination', 'CryoSleep', 'VIP', 'HomePlanet', 'CabinDeck', 'CabinSide'], axis = 1)
df1
