import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Dropout
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# baseline
model1 = Sequential()
model1.add(Flatten())
model1.add(Dense(64, activation='relu'))
model1.add(Dense(10, activation='softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history1 = model1.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_test, y_test))
model1.save('my_model1.h5')
model1.summary()
print(model1.evaluate(X_test, y_test))

epochs = range(10)
loss1 = history1.history['loss']
val_loss1 = history1.history['val_loss']
plt.plot(epochs, loss1, 'r', label='training loss ANN')
plt.plot(epochs, val_loss1, 'b', label='validation loss ANN')
plt.legend()
plt.show()

model_simpan = load_model('my_model1.h5')
pred = model_simpan.predict(X_test)
print('label actual:', np.argmax(y_test[30]))
print('label prediction:', np.argmax(pred[30]))

# a. L1 (rate 0.0001)
model2 = Sequential()
model2.add(Flatten())
model2.add(Dense(64, activation='relu', kernel_regularizer=l1(0.0001)))
model2.add(Dense(10, activation='softmax'))
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history2 = model2.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_test, y_test))
model2.summary()
print(model2.evaluate(X_test, y_test))

plt.plot(epochs, history2.history['loss'], 'r', label='training loss L1')
plt.plot(epochs, history2.history['val_loss'], 'b', label='validation loss L1')
plt.legend()
plt.show()

# b. L2 (rate 0.001)
model3 = Sequential()
model3.add(Flatten())
model3.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model3.add(Dense(10, activation='softmax'))
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history3 = model3.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_test, y_test))
model3.summary()
print(model3.evaluate(X_test, y_test))

plt.plot(epochs, history3.history['loss'], 'r', label='training loss L2')
plt.plot(epochs, history3.history['val_loss'], 'b', label='validation loss L2')
plt.legend()
plt.show()

# c. Dropout (rate 0.3)
model4 = Sequential()
model4.add(Flatten())
model4.add(Dense(64, activation='relu'))
model4.add(Dropout(0.3))
model4.add(Dense(10, activation='softmax'))
model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history4 = model4.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_test, y_test))
model4.summary()
print(model4.evaluate(X_test, y_test))

plt.plot(epochs, history4.history['loss'], 'r', label='training loss dropout')
plt.plot(epochs, history4.history['val_loss'], 'b', label='validation loss dropout')
plt.legend()
plt.show()

# d. Early stopping
model5 = Sequential()
model5.add(Flatten())
model5.add(Dense(64, activation='relu'))
model5.add(Dense(10, activation='softmax'))
model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history5 = model5.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_test, y_test), callbacks=[stop])
model5.summary()
print(model5.evaluate(X_test, y_test))

epochs5 = range(len(history5.history['loss']))
plt.plot(epochs5, history5.history['loss'], 'r', label='training loss early stopping')
plt.plot(epochs5, history5.history['val_loss'], 'b', label='validation loss early stopping')
plt.legend()
plt.show()

# e. L1 (0.0001) + Dropout (0.3)
model6 = Sequential()
model6.add(Flatten())
model6.add(Dense(64, activation='relu', kernel_regularizer=l1(0.0001)))
model6.add(Dropout(0.3))
model6.add(Dense(10, activation='softmax'))
model6.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history6 = model6.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_test, y_test))
model6.summary()
print(model6.evaluate(X_test, y_test))

plt.plot(epochs, history6.history['loss'], 'r', label='training loss L1+dropout')
plt.plot(epochs, history6.history['val_loss'], 'b', label='validation loss L1+dropout')
plt.legend()
plt.show()
