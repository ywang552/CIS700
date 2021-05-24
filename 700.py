import numpy as np
from keras.models import Sequential
from keras.layers import Dense
np.set_printoptions(precision=4,suppress=True)
import math
LIMIT = 10
import matplotlib.pyplot as plt

def GRN_gen(dim, lw,high):
    reso = np.zeros([dim,dim])
    limit = [None]*dim
    for i in range(dim):
        tmp = np.random.randint(lw,high+1)
        tmpa = np.random.choice(dim,tmp,replace=False)
        limit[i] = np.array(tmpa)
    for i in range(len(reso)):
        for j in range(len(reso)):
            if(i in limit[j]):
                reso[i,j] = np.random.uniform(-1,1)
    return reso
def generator(init, GRN, tstep):
    dim = len(init)
    reso = np.ndarray((tstep,dim))
    reso[0] = init
    for i in range(1,tstep):
        reso[i] = np.matmul(GRN,reso[i-1])
        for j in range(len(reso[i])):
            if(abs(reso[i,j]) > LIMIT):
                reso[i,j] = LIMIT
    return reso

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def norm(data,tstep, dim):
    max = [-1000]*len(data[0])
    min = [1000]*len(data[0])
    ranger = [None]*len(data[0])
    for t in range(tstep):
        for i in range(dim):
            if(data[t,i]> max[i]):  
                max[i] = data[t,i]
            if(data[t,i]< min[i]):
                min[i] = data[t,i]
    
    for i in range(len(max)):
        ranger[i] = max[i] - min[i]
    for t in range(tstep):
        for i in range(dim):
            
            data[t,i] = (data[t,i] - min[i])/ranger[i]
    
    return data
dim = 10
tstep = 100
ts = int(tstep/10*8)

init = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10])
GRN = GRN_gen(dim, 2, 5)

data = generator(init,GRN,tstep)

data = norm(data,tstep,dim)


training = data[:ts]
testing = data[ts:]

training_x = training[:-1]
training_y = training[1:]

testing_x = testing[:-1]
testing_y = testing[1:]
print(len(training_x),len(training_y))
print(len(testing_x),len(testing_y))
model = Sequential()
model.add(Dense(dim*4, input_dim=dim, activation='relu'))
model.add(Dense(dim*8, activation='relu'))
model.add(Dense(dim, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history1 = model.fit(training_x, training_y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
_, accuracy = model.evaluate(training_x, training_y)
print('Accuracy: %.2f' % (accuracy*100))



model = Sequential()
model.add(Dense(dim*2, input_dim=dim, activation='relu'))
model.add(Dense(dim*4, activation='relu'))
model.add(Dense(dim, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history2 = model.fit(training_x, training_y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)

model = Sequential()
model.add(Dense(dim*4, input_dim=dim, activation='relu'))
model.add(Dense(dim*2, activation='relu'))
model.add(Dense(dim, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history3 = model.fit(training_x, training_y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)

predictions = model.predict(testing_x)
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.plot(history3.history['accuracy'])
plt.plot(history3.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['4-8_train', '4-8_test','2-4_train','2-4_test', '4-2_train', '4-2_test'], loc='upper left')
plt.show()
