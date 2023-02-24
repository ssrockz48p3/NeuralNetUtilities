import numpy as np
from math import exp


inp = np.array([[1.65, 1.65, 1.65],
                [1.875, 1.875, 1.875]])


def relu(dot):
    out = np.zeros(shape=np.shape(dot))
    for i in range(len(dot)):
        out[i] = np.maximum(dot[i], 0)
    return out


def softmax(dot_list):
    softmax_arr = np.array([])
    for dot in dot_list:
        tot = sum([exp(x) for x in dot])
        out = np.array([exp(x)/tot for x in dot])
        softmax_arr = np.concatenate((softmax_arr, out))
    return softmax_arr.reshape(len(dot_list), -1)


def sigmoid(dot):
    out = np.zeros(shape=np.shape(dot))


class Errors:
    def __init__(self, y_rl, y_pred):
        self.y_rl = y_rl
        self.y_pred = y_pred

    def mean_errors(self, type="RMSE"):
        total_err = 0

        if type == "MSE":
            for i in range(len(self.y_rl)):
                total_err = total_err + ((y_rl[i]-y_pred[i])**2)
        elif type == "MAE":
            for i in range(len(self.y_rl)):
                total_err = total_err + abs(y_rl[i]-y_pred[i])
        elif type == "RMSE":
            for i in range(len(self.y_rl)):
                total_err = total_err + (((y_rl[i]**2)-(y_pred[i]**2))**(1/2))

        mean_err = total_squared_err/len(y_rl)
        return mean_err


