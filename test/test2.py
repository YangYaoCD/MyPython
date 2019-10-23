import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import imageio
from test.lr_utils import load_dataset

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    sum1=Y * np.log(A)
    sum2=(1 - Y) * np.log(1 - A)
    cost = (-1. / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A), axis=1)  # 按行相加
    dw = (1. / m) * np.dot(X, ((A - Y).T))  # dw就是损失函数对w的求导
    db = (1. / m) * np.sum(A - Y, axis=1)  # axis=0按列相加，axis=1按行相加
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)  # squeeze函数的作用是去掉维度为1的维,在这就是将一个一维变成一个数字
    assert (cost.shape == ())
    grads = {"dw": dw,"db": db}
    return grads, cost

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        prediction=0
        if(A[0][i]>0.5):
            prediction=1
        Y_prediction[0][i]=prediction
    assert (Y_prediction.shape == (1, m))
    return Y_prediction

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - np.dot(learning_rate, dw)
        b = b - np.dot(learning_rate, db)
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
num_px=train_set_x_orig.shape[1]
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
# print(d["costs"])
# print(np.squeeze(d["costs"]))
# plt.plot(np.squeeze(d["costs"]))

# index = 5
# plt.imshow(test_set_x[:,index].reshape((train_set_x_orig.shape[1], train_set_x_orig.shape[1], 3)))
# plt.show()

my_image = "}C%1[A3IG8O1IC_D@}``%F8.png"   # change this to the name of your image file
fname = "images/" + my_image
image = np.array(imageio.imread(fname))
from scipy import misc
size=(num_px,num_px)
my_image = np.array(Image.fromarray(image).resize(size)).reshape((1, num_px*num_px*3)).T
# my_image = np.array(Image.fromarray(image).resize(train_set_x_orig.shape[1],train_set_x_orig.shape[1])).reshape((1, train_set_x_orig.shape[1]*train_set_x_orig.shape[1]*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
plt.show()
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
