import numpy as np
import pandas as pd

ratings = pd.read_csv('../data/bookcrossing.csv',sep=',',error_bad_lines=False,encoding='latin-1')
users = pd.read_csv('../data/BX-Users.csv',sep=';',error_bad_lines=False,encoding='latin-1')
counts = ratings.groupby("User-ID").filter(lambda x: len(x) > 2000)

ratings['Offered'] = pd.Series(0, index=ratings.index)
#ratings['Offered'] = ratings.loc[ratings['Book-Rating'] > 7]
#ratings['Offered'] = ratings.loc[ratings['Book-Rating'] > 7] = 1
ratings['Offered'] = np.where(ratings['Book-Rating'] > 7, 1, ratings['Book-Rating'])
ratings['Offered'] = np.where(ratings['Offered'] != 1, 0, ratings['Offered'])

merged = ratings.merge(users, left_on='User-ID', right_on='User-ID', how='inner')
merged = merged.dropna()
                        
y = merged.iloc[0:50000,3].values
merged = merged.drop(['Offered'], axis=1)
x = merged.as_matrix()
#x = x[:,[2,4]]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
x[:, 1] = le.fit_transform(x[:, 1])
x[:, 3] = le.fit_transform(x[:, 3])
#
x_sample = x[0:50000,:]
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0,1,3])
x_sample = onehotencoder.fit_transform(x_sample).toarray()
x = x_sample
 
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.regularizers import l2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=0)

model = Sequential()

##Hidden Layer-1
model.add(Dense(100,activation='relu',input_dim=43035,kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3, noise_shape=None, seed=None))
#
##Hidden Layer-2
model.add(Dense(100,activation = 'relu',kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3, noise_shape=None, seed=None))
#
##Output layer
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_output = model.fit(x_train,y_train,epochs=10,batch_size=100,verbose=1,validation_data=(x_test,y_test),)
#
print('Training Accuracy : ' , np.mean(model_output.history["acc"]))
print('Validation Accuracy : ' , np.mean(model_output.history["val_acc"]))
#
## Plot training & validation accuracy values
#plt.plot(model_output.history['acc'])
#plt.plot(model_output.history['val_acc'])
#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
#
## Plot training & validation loss values
#plt.plot(model_output.history['loss'])
#plt.plot(model_output.history['val_loss'])
#plt.title('model_output loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
#
#y_pred = model.predict(x_test)
#rounded = [round(x[0]) for x in y_pred]
#y_pred1 = np.array(rounded,dtype='int64')
#confusion_matrix(y_test,y_pred1)
#precision_score(y_test,y_pred1)
#model.save("Calssifier.h5")
