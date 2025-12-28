import math
import numpy as np

# 1. easy mode: have input layer and output layer. only one neuron in each layer
# use a real number other than a vector as the input.

# #sigmod function

# def sigmod(x):
#     return 1/(1 + math.e**(-x))




# # initialise w, b and learning rate 
# w1 = 1
# b1 = 1

# w2 = 1
# b2 = 1

# learning_rate = 0.001

# input = 1
# real_output = 0.855

# h = input*w1 + b1
# output = sigmod(h*w2 + b2)

# loss = (output-real_output)**2

# # back-propagating

# for i in range(100):
#     w1 = w1 - learning_rate* (2*(output - real_output)*w2*input*output*(1-output))
#     b1 = b1 - learning_rate* (2*(output - real_output)*w2*output*(1-output))

#     w2 = w2 - learning_rate* (2*(output-real_output)*h*output*(1-output))
#     b2 = b2 - learning_rate* (2*(output-real_output)*output*(1-output))

#     h = input*w1 + b1
#     output = sigmod(h*w2 + b2)
#     loss = (output-real_output)**2
#     print(loss)



# 2. recuisive net work(replaced by freeforward neuron network)



#sigmod function

def sigmod(x):
    return 1/(1 + math.e**(-x))


# three time input

x1 = 1
x2 = 2
x3 = 3

w1 = 1
w2 = 1
w3 = 1
w = 1

wh2 = 1
wh3 = 1
wh = 1

a = 0.05    #learning rate

b1 = 0
b2 = 0
b3 = 0
b = 0

h2 = sigmod(w1*x1 + b1)
h3 = sigmod(w2*x2 + h2*wh2 + b2)

t = 0.875   #target number
y = sigmod(w3*x3 + h3*wh3 + b3)
Loss = (y-t)**2


# back-propagating

for i in range(100):

    # calculate the gredient

    dL_dy = 2*(y-t)

    dy_dsigmod3 = y*(1-y)
    dsigmod3_dw3 = x3
    dsigmod3_dwh3 = h3
    dsigmod3_db3 = 1
    dsigmod3_dh3 = wh3


    dh3_dsigmod2 = h3*(1-h3)
    dsigmod2_dw2 = x2
    dsigmod2_dwh2 = h2
    dsigmod2_db2 = 1
    dsigmod2_dh2 = wh2

    dh2_dsigmod1 = h2*(1-h2)
    dsigmod1_dw1 = x1
    dsigmod1_db1 = 1

    #renew the weight and bias

    kw1 = dL_dy*dy_dsigmod3*dsigmod3_dh3*dh3_dsigmod2*dsigmod2_dh2*dh2_dsigmod1*dsigmod1_dw1
    kw2 = dL_dy*dy_dsigmod3*dsigmod3_dh3*dh3_dsigmod2*dsigmod2_dw2
    kw3 = dL_dy*dy_dsigmod3*dsigmod3_dw3

    kb1 = dL_dy*dy_dsigmod3*dsigmod3_dh3*dh3_dsigmod2*dsigmod2_dh2*dh2_dsigmod1*dsigmod1_db1
    kb2 = dL_dy*dy_dsigmod3*dsigmod3_dh3*dh3_dsigmod2*dsigmod2_db2
    kb3 = dL_dy*dy_dsigmod3*dsigmod3_db3

    kwh2 = dL_dy*dy_dsigmod3*dsigmod3_dh3*dh3_dsigmod2*dsigmod2_dwh2
    kwh3 = dL_dy*dy_dsigmod3*dsigmod3_dwh3


    avgkw = (kw1+kw2+kw3)/3
    avgkwh = (kwh2+kwh3)/2
    avgb = (kb1+kb2+kb3)/3

    w = w - a*avgkw
    w1=w2=w3=w

    b = b - a*avgb
    b1=b2=b3 = b

    wh = wh - a*avgkwh
    wh2 = wh3 = wh


    # forward-propagating again


    h2 = sigmod(w1*x1 + b1)
    h3 = sigmod(w2*x2 + h2*wh2 + b2)

    y = sigmod(w3*x3 + h3*wh3 + b3)
    Loss = (y-t)**2

    # 改进的打印输出
    if i % 10 == 0:
        print(f"Iteration {i}: Loss = {Loss:.6f}, Output = {y:.6f}")




