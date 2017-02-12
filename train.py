import pandas as pd
import numpy as np
import math
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from keras.regularizers import l1, activity_l2, l2
from keras.callbacks import EarlyStopping, Callback
from keras.preprocessing import sequence
from sklearn.preprocessing import MinMaxScaler

#For reproducibility
np.random.seed(7)

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.best_epoch = None
		self.min_loss = 100000
	def on_epoch_end(self, epoch, logs={}):
		if self.min_loss > logs.get('val_loss'):
			self.min_loss = logs.get('val_loss')
			self.best_epoch = epoch


def train_val_split(X):
	#Split data into train and test: 70%-30% split
	random.seed(7)
	ind_1 = np.array(range(0,X.shape[0]))
	ind_val_1 = np.array(random.sample(ind_1, int(0.3*X.shape[0])))
	ind_train_1 = np.delete(ind_1, ind_val_1)
	random.seed(7)
	ind_2 = ind_train_1
	ind_val_2 = np.array(random.sample(ind_2, int(0.3*X.shape[0])))
	ind_train_2 = np.delete(ind_1, ind_val_2)
	x_1 = X[ind_train_1,0:X.shape[1]-1]
	x_val_1 = X[ind_val_1,0:X.shape[1]-1]
	y_1 = X[ind_train_1, -1]
	y_val_1 = X[ind_val_1, -1]
	x_2 = X[ind_train_2,0:X.shape[1]-1]
	x_val_2 = X[ind_val_2,0:X.shape[1]-1]
	y_2 = X[ind_train_2, -1]
	y_val_2 = X[ind_val_2, -1]
	return (x_1, x_val_1, y_1, y_val_1, x_2, x_val_2, y_2, y_val_2)




#load projectiles.csv
ip = pd.read_csv('projectiles.csv', index_col=None, header=None)
ip = np.array(ip)



##########plot for x and y for different projectiles############
#Uncomment this to plot horizontal dist x and vertical dist y as a function of time t
"""
import matplotlib.pyplot as plt

f, p = plt.subplots(4, sharex=True)
#f.subplots_adjust(hspace=0.3)
p[0].set_title('Plot of x vs time-index')
p[1].set_title('Plot of y vs time-index')
p[2].set_title('Plot of log(x) vs log(time-index')
p[3].set_title('Plot of log(y) vs log(time-index')
z = np.where(ip[:,0] == 0)[0]
for i in xrange(len(z)-1):
	p[0].plot(ip[z[i]:z[i+1],0], ip[z[i]:z[i+1],1])
	p[1].plot(ip[z[i]:z[i+1],0], ip[z[i]:z[i+1],2])
	p[2].plot(np.log(ip[z[i]:z[i+1],0]), np.log(ip[z[i]:z[i+1],1]))
	p[3].plot(np.log(ip[z[i]:z[i+1],0]), np.log(ip[z[i]:z[i+1],2]))

plt.show()
"""

############################ Train for x ##################################

#train with features: [t, x(t1), t*x(t1)]
X = []
for i in ip:
	if i[0] == 1:
		x = i[1]
	elif i[0] != 0:
		X.append([i[0], x, x*i[0], i[1]])


X = np.array(X)
#Train for Horizontal distance

x_1, x_val_1, y_1, y_val_1, x_2, x_val_2, y_2, y_val_2 = train_val_split(X)

#data values are not to large or small, no high variance, hence there is no need to normalize 
##########Build model-1#############

#Model 1
model1 = Sequential()
model1.add(Dense(1, input_dim=x_1.shape[1], activation='linear'))
model1.compile(loss='mse', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', patience=8)
history = LossHistory()
batch_size = 10
model1.fit(x_1, y_1, nb_epoch=500, batch_size=batch_size, validation_data=(x_val_1, y_val_1), shuffle=True, callbacks=[early_stopping, history])

print "Minimum validation loss is - mean square error: ", history.min_loss

w_m1 = 1-(pow(history.min_loss, 0.5)/np.mean(y_val_1))
print "RMSE percent of mean of output: ", pow(history.min_loss, 0.5)*100/np.mean(y_val_1) 

#get the best epoch and re run training on entire data
model1 = Sequential()
model1.add(Dense(1, input_dim=x_1.shape[1], activation='linear'))
model1.compile(loss='mse', optimizer='adam')
model1.fit(np.append(x_1,x_val_1,axis=0), np.append(y_1,y_val_1,axis=0), nb_epoch=history.best_epoch, batch_size=batch_size, shuffle=True)

##############predict path for projectile shot at an angle of 45 degrees and velocity of 10m/s
x_t1 = 0.707106781187
x_pred_1 = []
for i in xrange(2,101):
	x_pred_1.append(model1.predict(np.array([[x_t1, i, x_t1*i]]))[0][0])






##########Build model-2#############

#Model 2
model2 = Sequential()
model2.add(Dense(1, input_dim=x_2.shape[1], activation='linear'))
model2.compile(loss='mse', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', patience=8)
history = LossHistory()
batch_size = 10
model2.fit(x_2, y_2, nb_epoch=500, batch_size=batch_size, validation_data=(x_val_2, y_val_2), shuffle=True, callbacks=[early_stopping, history])

print "Minimum validation loss is - mean square error: ", history.min_loss
w_m2 = 1-(pow(history.min_loss, 0.5)/np.mean(y_val_2))
print "RMSE percent of mean of output: ", pow(history.min_loss, 0.5)*100/np.mean(y_val_2) 

#get the best epoch and re run training on entire data
model2 = Sequential()
model2.add(Dense(1, input_dim=x_1.shape[1], activation='linear'))
model2.compile(loss='mse', optimizer='adam')
model2.fit(np.append(x_2,x_val_2,axis=0), np.append(y_2,y_val_2,axis=0),  nb_epoch=history.best_epoch, batch_size=batch_size, shuffle=True)

##############predict path for projectile shot at an angle of 45 degrees and velocity of 10m/s
x_t2 = 0.707106781187
x_pred_2 = []
for i in xrange(2,101):
	x_pred_2.append(model2.predict(np.array([[x_t2, i, x_t2*i]]))[0][0])




###### Take average of results from model1 and model2
w_m1 = w_m1/(w_m1+w_m2)
w_m2 = 1-w_m1
x_pred = w_m1 * np.array(x_pred_1) + w_m2 * np.array(x_pred_2)


 



####################With log transformation#####################################

#Transform data with log - both input and output
X = []
for i in ip:
	if i[0] == 1:
		x = math.log(i[1])
	elif i[0] != 0:
		X.append([math.log(i[0]), x, math.log(i[1])])

X = np.array(X)

x_1, x_val_1, y_1, y_val_1, x_2, x_val_2, y_2, y_val_2 = train_val_split(X)

#data values are not to large or small, no high variance, hence there is no need to normalize 
##########Build model-1#############

#Model 1
model1 = Sequential()
model1.add(Dense(1, input_dim=x_1.shape[1], activation='linear'))
model1.compile(loss='mse', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', patience=8)
history = LossHistory()
batch_size = 10
model1.fit(x_1, y_1, nb_epoch=500, batch_size=batch_size, validation_data=(x_val_1, y_val_1), shuffle=True, callbacks=[early_stopping, history])

####predict for x_val and get mse in non-log transformed output.
y_val_pred = pow(math.e, model1.predict(x_val_1)[:,0])
y_val_actual = pow(math.e, y_val_1)


print "Minimum validation loss is - mean square error: ",  np.mean(pow(y_val_pred - y_val_actual,2)) 
w_m1 = 1- (pow( np.mean(pow(y_val_pred - y_val_actual,2)) , 0.5)/np.mean(y_val_actual))
print "RMSE percent of mean of output: ", pow( np.mean(pow(y_val_pred - y_val_actual,2)) , 0.5)*100/np.mean(y_val_actual) 

#get the best epoch and re run training on entire data
model1 = Sequential()
model1.add(Dense(1, input_dim=x_1.shape[1], activation='linear'))
model1.compile(loss='mse', optimizer='adam')
model1.fit(np.append(x_1,x_val_1,axis=0), np.append(y_1,y_val_1,axis=0), nb_epoch=history.best_epoch, batch_size=batch_size, shuffle=True)

##############predict path for projectile shot at an angle of 45 degrees and velocity of 10m/s
x_t1 = 0.707106781187
x_t1_log = math.log(x_t1)
x_pred_1 = []
for i in xrange(2,101):
	x_pred_1.append(pow(math.e,model1.predict(np.array([[math.log(i), x_t1_log]]))[0][0]))





##########Build model-2#############

#Model 2
model2 = Sequential()
model2.add(Dense(1, input_dim=x_2.shape[1], activation='linear'))
model2.compile(loss='mse', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', patience=8)
history = LossHistory()
batch_size = 10
model2.fit(x_2, y_2, nb_epoch=500, batch_size=batch_size, validation_data=(x_val_2, y_val_2), shuffle=True, callbacks=[early_stopping, history])

####predict for x_val and get mse in non-log transformed output.
y_val_pred = pow(math.e, model2.predict(x_val_2)[:,0])
y_val_actual = pow(math.e, y_val_2)

print "Minimum validation loss is - mean square error: ",  np.mean(pow(y_val_pred - y_val_actual,2)) 
w_m2 = 1 - (pow( np.mean(pow(y_val_pred - y_val_actual,2)) , 0.5)/np.mean(y_val_actual))
print "RMSE percent of mean of output: ", pow( np.mean(pow(y_val_pred - y_val_actual,2)) , 0.5)*100/np.mean(y_val_actual) 

#get the best epoch and re run training on entire data
model2 = Sequential()
model2.add(Dense(1, input_dim=x_1.shape[1], activation='linear'))
model2.compile(loss='mse', optimizer='adam')
model2.fit(np.append(x_2,x_val_2,axis=0), np.append(y_2,y_val_2,axis=0), nb_epoch=history.best_epoch, batch_size=batch_size, shuffle=True)

##############predict path for projectile shot at an angle of 45 degrees and velocity of 10m/s
x_t1 = 0.707106781187
x_t1_log = math.log(x_t1)
x_pred_2 = []
for i in xrange(2,101):
	x_pred_2.append(pow(math.e,model2.predict(np.array([[math.log(i), x_t1_log]]))[0][0]))



###### Take average of results from model1 and model2
w_m1 = w_m1/(w_m1+w_m2)
w_m2 = 1-w_m1
x_pred_log = w_m1*np.array(x_pred_1) + w_m2*np.array(x_pred_2)




###############################################################################################################################################################



######Train for vertical distance - y ################

#Train with features: (t, (t*0.1 * t*0.1), y(t1), y(t1)*t*0.1) 
X = []

for i in ip:
        if i[0] == 1:
                y = i[2]
        elif i[0] != 0:
                X.append([i[0], pow(0.1*i[0],2), y, y*0.1*i[0], i[2]])


X = np.array(X)




x_1, x_val_1, y_1, y_val_1, x_2, x_val_2, y_2, y_val_2 = train_val_split(X)

#data values are not to large or small, no high variance, hence there is no need to normalize 
##########Build model-1#############

#Model 1
model1 = Sequential()
model1.add(Dense(1, input_dim=x_1.shape[1], activation='linear'))
model1.compile(loss='mse', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', patience=8)
history = LossHistory()
batch_size = 1
model1.fit(x_1, y_1, nb_epoch=500, batch_size=batch_size, validation_data=(x_val_1, y_val_1), shuffle=True, callbacks=[early_stopping, history])

print "Minimum validation loss is - mean square error: ", history.min_loss
w_m1 = 1 - (pow(history.min_loss, 0.5)/np.mean(y_val_1))
print "RMSE percent of mean of output: ", pow(history.min_loss, 0.5)*100/np.mean(y_val_1) 

#get the best epoch and re run training on entire data
model1 = Sequential()
model1.add(Dense(1, input_dim=x_1.shape[1], activation='linear'))
model1.compile(loss='mse', optimizer='adam')

model1.fit(np.append(x_1,x_val_1,axis=0), np.append(y_1,y_val_1,axis=0), nb_epoch=history.best_epoch, batch_size=batch_size, shuffle=True)

##############predict path for projectile shot at an angle of 45 degrees and velocity of 10m/s
y_t1 = 0.658106781187
y_pred_1 = []
for i in xrange(2,101):
	pred = model1.predict(np.array([[i, pow(i*0.1,2), y_t1, y_t1*0.1*i]]))[0][0]
	y_pred_1.append(pred)






##########Build model-2#############

#Model 2
model2 = Sequential()
model2.add(Dense(1, input_dim=x_2.shape[1], activation='linear'))
model2.compile(loss='mse', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', patience=8)
history = LossHistory()
batch_size = 1
model2.fit(x_2, y_2, nb_epoch=500, batch_size=batch_size, validation_data=(x_val_2, y_val_2), shuffle=True, callbacks=[early_stopping, history])

print "Minimum validation loss is - mean square error: ", history.min_loss
w_m2 =  1 - (pow(history.min_loss, 0.5)/np.mean(y_val_2))
print "RMSE percent of mean of output: ", pow(history.min_loss, 0.5)*100/np.mean(y_val_2) 

#get the best epoch and re run training on entire data
model2 = Sequential()
model2.add(Dense(1, input_dim=x_1.shape[1], activation='linear'))
model2.compile(loss='mse', optimizer='adam')
model2.fit(np.append(x_2,x_val_2,axis=0), np.append(y_2,y_val_2,axis=0),  nb_epoch=history.best_epoch, batch_size=batch_size, shuffle=True)


##############predict path for projectile shot at an angle of 45 degrees and velocity of 10m/s
y_t1 = 0.658106781187
y_pred_2 = []
for i in xrange(2,101):
	pred = model2.predict(np.array([[i, pow(i*0.1,2), y_t1, y_t1*0.1*i]]))[0][0]
	y_pred_2.append(pred)




###### Take average of results from model1 and model2
w_m1 = w_m1/(w_m1+w_m2)
w_m2 = 1-w_m1
y_pred = w_m1*np.array(y_pred_1) + w_m2*np.array(y_pred_2)

break_index = 0
for i in xrange(0,len(y_pred)):
	if y_pred[i] < 0:
		break_index = i
		break

y_pred = y_pred[0:break_index]



#########################################################################################################################


#### Combine x and y and writing into .csv file #########

result = [[0,0.0, 0.0], [1, x_t1, y_t1]]

# having x values predicted by model trained with output and input being log transformed
result_log = [[0,0.0, 0.0], [1, x_t1, y_t1]]

for i in xrange(0,len(y_pred)):
	result.append([i+2, x_pred[i], y_pred[i]])
	result_log.append([i+2, x_pred_log[i], y_pred[i]])


result = np.array(result)
result_log = np.array(result_log)



print "writing to result_lin_reg.csv"
result = pd.DataFrame({"[time_index]": result[:,0], "[x]": result[:,1], "[y]" : result[:,2]})
result.to_csv('result_lin_reg.csv',index=False)



print "writing to result_lin_reg_log.csv"
result_log = pd.DataFrame({"[time_index]": result_log[:,0], "[x]": result_log[:,1], "[y]" : result_log[:,2]})
result_log.to_csv('result_lin_reg_log.csv',index=False)



############################################################################################################### LSTM ##############################################################################################################

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def create_dataset(ip, window_size, index):
	i = 0
	train = []
	while(i<ip.shape[0]):
		if (ip[i,0] == 0):
			ind = i
		if(ip[i,0] > 1):
			train.append(list(ip[max(ind,i-window_size):i+1,index]))
		i = i+1
	return np.array(sequence.pad_sequences(train, dtype='float32', maxlen=window_size))


####### Train for horizontal distance - X ############
window_size = 3
X = create_dataset(ip, window_size, 1)

#scaling is optional because of the range of data values.
#scaler_x = MinMaxScaler(feature_range=(0, 1))
#X[:,:-1] = scaler_x.fit_transform(X[:,:-1])
#scaler_y = MinMaxScaler(feature_range=(0, 1))
#X[:,-1] = scaler_y.fit_transform(X[:,-1])



x_1, x_val_1, y_1, y_val_1, x_2, x_val_2, y_2, y_val_2 = train_val_split(X)

x_1 = np.reshape(x_1, (x_1.shape[0], 1, x_1.shape[1]))
x_val_1 = np.reshape(x_val_1, (x_val_1.shape[0], 1, x_val_1.shape[1]))
x_2 = np.reshape(x_2, (x_2.shape[0], 1, x_2.shape[1]))
x_val_2 = np.reshape(x_val_2, (x_val_2.shape[0], 1, x_val_2.shape[1]))

# create and fit the LSTM network for model1

model1 = Sequential()
model1.add(LSTM(16, input_shape=x_1.shape[1:]))
model1.add(Dense(1))
model1.compile(loss='mean_squared_error', optimizer='adam')



history = LossHistory()
batch_size = 1
early_stopping = EarlyStopping(monitor='val_loss', patience=8)

model1.fit(x_1, y_1, nb_epoch=500, batch_size=batch_size,  validation_data=(x_val_1, y_val_1), verbose=2, shuffle=True, callbacks=[early_stopping, history])

print "Minimum validation loss is - mean square error: ", history.min_loss
w_m1 = 1 -  (pow(history.min_loss, 0.5)/np.mean(y_val_1))
print "RMSE percent of mean of output: ", pow(history.min_loss, 0.5)*100/np.mean(y_val_1) 


#get the best epoch and re run training on entire data
model1 = Sequential()
model1.add(LSTM(16, input_shape=x_1.shape[1:]))
model1.add(Dense(1))
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.fit(np.append(x_1,x_val_1,axis=0), np.append(y_1,y_val_1,axis=0), nb_epoch=history.best_epoch, batch_size=batch_size, verbose=2, shuffle=True)


##############predict path for projectile shot at an angle of 45 degrees and velocity of 10m/s
x_t1 = 0.707106781187
x_pred_1 = []
seq = np.array([sequence.pad_sequences([[0, x_t1]], dtype='float32', maxlen=window_size-1)])
for i in xrange(2,101):
	pred = model1.predict(seq)[0][0]
	temp = np.zeros((1,window_size-1))
	temp[0,-1] = pred
	temp[0,0:-1] = seq[0][0][1:]
	seq[0] = temp
	x_pred_1.append(pred)



# create and fit the LSTM network for model2

model2 = Sequential()
model2.add(LSTM(16, input_shape=x_2.shape[1:]))
model2.add(Dense(1))
model2.compile(loss='mean_squared_error', optimizer='adam')



history = LossHistory()
batch_size = 1
early_stopping = EarlyStopping(monitor='val_loss', patience=8)

model2.fit(x_2, y_2, nb_epoch=500, batch_size=batch_size,  validation_data=(x_val_2, y_val_2), verbose=2, shuffle=True, callbacks=[early_stopping, history])

print "Minimum validation loss is - mean square error: ", history.min_loss
w_m2 = 1 - (pow(history.min_loss, 0.5)/np.mean(y_val_2))
print "RMSE percent of mean of output: ", pow(history.min_loss, 0.5)*100/np.mean(y_val_2) 


#get the best epoch and re run training on entire data
model2 = Sequential()
model2.add(LSTM(16, input_shape=x_2.shape[1:]))
model2.add(Dense(1))
model2.compile(loss='mean_squared_error', optimizer='adam')
model2.fit(np.append(x_2,x_val_2,axis=0), np.append(y_2,y_val_2,axis=0), nb_epoch=history.best_epoch, batch_size=batch_size, verbose=2, shuffle=True)


##############predict path for projectile shot at an angle of 45 degrees and velocity of 10m/s
x_t1 = 0.707106781187
x_pred_2 = []
seq = np.array([sequence.pad_sequences([[0, x_t1]], dtype='float32', maxlen=window_size-1)])
for i in xrange(2,101):
        pred = model2.predict(seq)[0][0]
        temp = np.zeros((1,window_size-1))
        temp[0,-1] = pred
        temp[0,0:-1] = seq[0][0][1:]
        seq[0] = temp
        x_pred_2.append(pred)


###### Take average of results from model1 and model2
w_m1 = w_m1/(w_m1+w_m2)
w_m2 = 1 - w_m1
x_pred = w_m1*np.array(x_pred_1) + w_m2*np.array(x_pred_2)


############For vertical distance##################


window_size = 5
X = create_dataset(ip, window_size, 2)

x_1, x_val_1, y_1, y_val_1, x_2, x_val_2, y_2, y_val_2 = train_val_split(X)

x_1 = np.reshape(x_1, (x_1.shape[0], 1, x_1.shape[1]))
x_val_1 = np.reshape(x_val_1, (x_val_1.shape[0], 1, x_val_1.shape[1]))
x_2 = np.reshape(x_2, (x_2.shape[0], 1, x_2.shape[1]))
x_val_2 = np.reshape(x_val_2, (x_val_2.shape[0], 1, x_val_2.shape[1]))

# create and fit the LSTM network for model1

model1 = Sequential()
model1.add(LSTM(8, input_shape=x_1.shape[1:]))
model1.add(Dense(1))
model1.compile(loss='mean_squared_error', optimizer='adam')



history = LossHistory()
batch_size = 1
early_stopping = EarlyStopping(monitor='val_loss', patience=8)

model1.fit(x_1, y_1, nb_epoch=500, batch_size=batch_size,  validation_data=(x_val_1, y_val_1), verbose=2, shuffle=True, callbacks=[early_stopping, history])

print "Minimum validation loss is - mean square error: ", history.min_loss
w_m1 = 1 - (pow(history.min_loss, 0.5)/np.mean(y_val_1))
print "RMSE percent of mean of output: ", pow(history.min_loss, 0.5)*100/np.mean(y_val_1) 


#get the best epoch and re run training on entire data
model1 = Sequential()
model1.add(LSTM(8, input_shape=x_1.shape[1:]))
model1.add(Dense(1))
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.fit(np.append(x_1,x_val_1,axis=0), np.append(y_1,y_val_1,axis=0), nb_epoch=history.best_epoch, batch_size=batch_size, verbose=2, shuffle=True)


##############predict path for projectile shot at an angle of 45 degrees and velocity of 10m/s
y_t1 = 0.658106781187
y_pred_1 = []
seq = np.array([sequence.pad_sequences([[0, y_t1]], dtype='float32', maxlen=window_size-1)])
for i in xrange(2,101):
        pred = model1.predict(seq)[0][0]
        temp = np.zeros((1,window_size-1))
        temp[0,-1] = pred
        temp[0,0:-1] = seq[0][0][1:]
        seq[0] = temp
        y_pred_1.append(pred)


# create and fit the LSTM network for model2

model2 = Sequential()
model2.add(LSTM(8, input_shape=x_2.shape[1:]))
model2.add(Dense(1))
model2.compile(loss='mean_squared_error', optimizer='adam')



history = LossHistory()
batch_size = 1
early_stopping = EarlyStopping(monitor='val_loss', patience=8)

model2.fit(x_2, y_2, nb_epoch=500, batch_size=batch_size,  validation_data=(x_val_2, y_val_2), verbose=2, shuffle=True, callbacks=[early_stopping, history])

print "Minimum validation loss is - mean square error: ", history.min_loss
w_m2 = 1 - (pow(history.min_loss, 0.5)/np.mean(y_val_2))
print "RMSE percent of mean of output: ", pow(history.min_loss, 0.5)*100/np.mean(y_val_2) 


#get the best epoch and re run training on entire data
model2 = Sequential()
model2.add(LSTM(8, input_shape=x_2.shape[1:]))
model2.add(Dense(1))
model2.compile(loss='mean_squared_error', optimizer='adam')
model2.fit(np.append(x_2,x_val_2,axis=0), np.append(y_2,y_val_2,axis=0), nb_epoch=history.best_epoch, batch_size=batch_size, verbose=2, shuffle=True)


##############predict path for projectile shot at an angle of 45 degrees and velocity of 10m/s
y_t1 = 0.658106781187
y_pred_2 = []
seq = np.array([sequence.pad_sequences([[0, y_t1]], dtype='float32', maxlen=window_size-1)])
for i in xrange(2,101):
        pred = model2.predict(seq)[0][0]
        temp = np.zeros((1,window_size-1))
        temp[0,-1] = pred
        temp[0,0:-1] = seq[0][0][1:]
        seq[0] = temp
        y_pred_2.append(pred)




###### Take average of results from model1 and model2
w_m1 = w_m1/(w_m1+w_m2)
w_m2 = 1 - w_m1
y_pred = w_m1*np.array(y_pred_1) + w_m2*np.array(y_pred_2)

break_index = 0
for i in xrange(0,len(y_pred)):
	if y_pred[i] < 0:
		break_index = i
		break

y_pred = y_pred[0:break_index]




#### Combine x and y and writing into .csv file #########

result = [[0,0.0, 0.0], [1, x_t1, y_t1]]

for i in xrange(0,len(y_pred)):
	result.append([i+2, x_pred[i], y_pred[i]])


result = np.array(result)

print "writing to result_lstm.csv"
result = pd.DataFrame({"[time_index]": result[:,0], "[x]": result[:,1], "[y]" : result[:,2]})
result.to_csv('result_lstm.csv',index=False)
