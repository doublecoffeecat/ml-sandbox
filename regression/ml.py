import numpy as np
import datetime


class CostFunction:
    def cost(self, y, y_hat):
        return 1/(2*len(y)) * np.sum((y_hat - y)**2)

    def gradient(self, X: np.ndarray, y: np.array, y_hat: np.array):
        m = len(y)
        return 1/m * np.sum((y_hat - y) * X), 1/m * np.sum(y_hat - y)


class LinearRegression:
    cost_value = 0

    def __init__(self, cost_function_obj=None, learning_rate=0.01, epsilon=0.01, n_iterations=1000):
        print('ml v19')
        self.w = None
        self.b = None
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_iterations = n_iterations

        if cost_function_obj is None:
            self.cost_function = CostFunction()
        else:
            self.cost_function = cost_function_obj

    def f(self, X, w, b):
        return w @ X + b


    def gradient_descent(self, X, y, w_init, b_init):
        w = w_init
        b = b_init
        l = self.learning_rate
        e = self.epsilon

        y_hat = self.f(X, w, b)
        for i in range(1000):
            gradient_w, gradient_b = self.cost_function.gradient(X, y, y_hat)
            print('gradient_w', gradient_w)
            print('gradient_b', gradient_b)
            w = w - l * gradient_w
            b = b - l * gradient_b
            print('w', w)
            print('b', b)

            y_hat_new = self.f(X, w, b)
            print('y_hat_new', y_hat_new)

            if np.sum(np.abs(y_hat_new - y_hat)) < e:
                print('Converged')
                print(np.sum(np.abs(y_hat_new - y_hat)))
                break
            y_hat = y_hat_new
        return w, b

    def fit(self, X: np.ndarray, y: np.array, w_init:np.ndarray = None, b_init: float = None) -> (np.ndarray, float):
        print('X', X)
        print('X shape', X.shape)
        if w_init is None:
            w_init = np.zeros((1, X.shape[1]))
        if b_init is None:
            b_init = 0
        self.w, self.b = self.gradient_descent(X, y, w_init, b_init)
        return self.w, self.b


    def predict(self, x):
        if self.w is None or self.b is None:
            raise Exception('Model not trained yet, run fit method first.')

        return self.f(x, self.w, self.b)
