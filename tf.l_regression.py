import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import pandas as pd 

plt.rcParams['figure.figsize'] = (10, 6)
x = np.arange(0.0,0.5,0.1)
a = 1
b = 0
y = a*x +b
plt.plot(x,y)
plt.xlabel("X")
plt.ylabel("Regression")
plt.show()
dataset = pd.read_csv('file.csv')
#print(dataset.head())

x_train = np.asanyarray(dataset[['Cause']])
y_train = np.asanyarray(dataset[['State']])
#print(x_train)

#initialize a random variable for graph
a = tf.Variable(20.0)
b = tf.Variable(30.2)

def h(x):
    y = a*x + b
    return y

loss_object = tf.keras.losses.MeanSquaredLogarithmicError()

learning_rate = 0.01
training_data = []
loss_values = []
a_values = []
b_values = []
trianing_epoch = 200

for epoch in range(trianing_epoch):
    with tf.GradientTape() as tape:
        y_pred = h(x_train)
        loss_value = loss_object(y_train,y_pred)
        loss_values.append(loss_value)
        
        gradients = tape.gradient(loss_values,[b,a]) 

        a_values.append(a.numpy())
        b_values.append(b.numpy())
        b.assign_sub(gradients[0]*learning_rate)
        a.assign_sub(gradients[1]*learning_rate)
        if epoch % 5 == 0:
            training_data.append([a.numpy(),b.numpy()])

# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.plot(loss_values,'ro')
#plt.show()
plt.scatter(x_train, y_train, color='green')
for a,b in zip(a_values[0:len(a_values)], b_values[0:len(b_values)]):
    plt.plot(x_train,a*x_train+b, color='red', linestyle='dashed')
plt.plot(x_train,a_values[-1]*x_train+b_values[-1], color='black')

final = mpatches.Patch(color='Black', label='Final')
estimates = mpatches.Patch(color='Red', label='Estimates')
data = mpatches.Patch(color='Green', label='Data Points')

plt.legend(handles=[data, estimates, final])

plt.show()












