#!/usr/bin/env python
# coding: utf-8

# # Q2

# In[5]:


import numpy as np
def generate_data(size, mean, std):
    random_data = np.random.normal(mean, std, size)
    data = random_data - np.mean(random_data)
    final_data = data * (std/np.std(data)) + mean
    return final_data
#class one
x1 = generate_data(100,2.0,1)
y1 = generate_data(100,1.0,0.1)
#class two
x2 = generate_data(100,-1.0,0.4)
y2 = generate_data(100,2.0,0.4)

import matplotlib.pyplot as plt
plt.scatter(x1, y1, label='class1')
plt.scatter(x2, y2, label='class2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[129]:


#Adaline

w1, w2, b = 0.1, 0.1, 0.1
learning_rate = 0.0004

target_ = np.concatenate([np.ones(len(x1))*1, np.ones(len(x2))*-1])

X1_ = np.concatenate([x1, x2])
X2_ = np.concatenate([y1, y2])

shuffler = np.random.permutation(len(target_))
X1 = X1_[shuffler]
X2 = X2_[shuffler]
target = target_[shuffler]

def calculate_net(i):
    return w1*X1[i] + w2*X2[i] + b

cost_values = []
def is_finished():
    for i in range(len(target)):
#         print(target[i], calculate_net(i))
        cost = 0.5 * (target[i] - calculate_net(i))**2
        cost_values.append(cost)
        if cost > 0.1:
            return False
    return True

epochs = 0
while not is_finished():
    epochs += 1
    if epochs == 30:
        break
    for i in range(len(target)):
        net = calculate_net(i)
        w1 += learning_rate*(target[i] - net)*X1[i]
        w2 += learning_rate*(target[i] - net)*X2[i]
        b += learning_rate*(target[i] - net)


# In[130]:


plt.scatter(x1, y1, label='class1')
plt.scatter(x2, y2, label='class2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
x = np.linspace(-1,3,100)
y = (-w1*x -b)/w2
plt.plot(x,y)
plt.show()
x = np.linspace(0,epochs,len(cost_values))
plt.scatter(x, cost_values)
plt.xlabel('epochs')
plt.ylabel('cost')
plt.show()


# In[164]:


#class one
x1 = generate_data(100,2.0,1)
y1 = generate_data(100,1.0,0.1)
#class two
x2 = generate_data(20,-1.0,0.4)
y2 = generate_data(20,2.0,0.4)

import matplotlib.pyplot as plt
plt.scatter(x1, y1, label='class1')
plt.scatter(x2, y2, label='class2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[175]:


#Adaline

w1, w2, b = 0.1, 0.1, 0.1
learning_rate = 0.0004

target_ = np.concatenate([np.ones(len(x1))*1, np.ones(len(x2))*-1])

X1_ = np.concatenate([x1, x2])
X2_ = np.concatenate([y1, y2])

shuffler = np.random.permutation(len(target_))
X1 = X1_[shuffler]
X2 = X2_[shuffler]
target = target_[shuffler]

def calculate_net(i):
    return w1*X1[i] + w2*X2[i] + b

cost_values = []
def is_finished():
    for i in range(len(target)):
#         print(target[i], calculate_net(i))
        cost = 0.5 * (target[i] - calculate_net(i))**2
        cost_values.append(cost)
        if cost > 0.1:
            return False
    return True

epochs = 0
while not is_finished():
    epochs += 1
    if epochs == 25:
        break
    for i in range(len(target)):
        net = calculate_net(i)
        w1 += learning_rate*(target[i] - net)*X1[i]
        w2 += learning_rate*(target[i] - net)*X2[i]
        b += learning_rate*(target[i] - net)


# In[176]:


plt.scatter(x1, y1, label='class1')
plt.scatter(x2, y2, label='class2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
x = np.linspace(-1,1.5,100)
y = (-w1*x -b)/w2
plt.plot(x,y)
plt.show()
x = np.linspace(0,epochs,len(cost_values))
plt.scatter(x, cost_values)
plt.xlabel('epochs')
plt.ylabel('cost')
plt.show()

