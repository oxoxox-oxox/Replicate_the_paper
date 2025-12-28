import math
import numpy as np

# easy mode: no hidden layer, only input layer and output layer.
# use a real number other than a vector as the input.

#Relu function

def relu(x):
    if x>0:
        return x
    else:
        return 0




# initialise w, b and learning rate 
w1 = 1
b1 = 1

w2 = 1
b2 = 1

learning_rate = 0.001

input = 1
real_output = 23

h = relu(input*w1 + b1)
output = h*w2 + b2

loss = (output-real_output)**2

for i in range(100):
    if (input*w1 + b1) < 0:
        w1 = w1
        b1 = b1
    else:
        w1 = w1 - learning_rate* (2*(output - real_output)*w2*input)
        b1 = b1 - learning_rate* (2*(output - real_output)*w2)

    w2 = w2 - learning_rate* (2*(output-real_output)*h)
    b2 = b2 - learning_rate* (2*(output-real_output))

    h = relu(input*w1 + b1)
    output = h*w2 + b2
    loss = (output-real_output)**2
    print(loss)



