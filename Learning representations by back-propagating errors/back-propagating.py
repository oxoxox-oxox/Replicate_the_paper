import math
import numpy as np

# easy mode: no hidden layer, only input layer and output layer.
# use a real number other than a vector as the input.




# initialise w, b and learning rate 
w = 1
b = 1
learning_rate = 3

input = 1
real_output = 0.874

z = input*w + b
output = 1/(1+math.e ** (-z))

loss = (output-real_output)**2

for i in range(100):
    w = w - learning_rate* (2*(output-real_output)*output*(1-output)*input)
    b = b - learning_rate* (2*(output-real_output)*output*(1-output))
    z = input*w + b
    output = 1/(1+math.e ** (-z))
    loss = (output-real_output)**2
    print(loss)

print(w,b)

