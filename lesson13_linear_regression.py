
def predict(alpha:  float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha


def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    return predict(alpha, beta, x_i) - y_i

from lesson3 import Vector
from typing import Tuple
from lesson4 import correlation, standard_deviation, mean


def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))


def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]

assert least_squares_fit(x, y) == (-5, 3)

from lesson4 import num_friends_good, daily_minutes_good, de_mean
import matplotlib.pyplot as plt
alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
assert 22.9 < alpha < 23.0
assert 0.9 < beta < 0.905

plt.scatter(num_friends_good, daily_minutes_good)
plt.title("Model linear regression")
plt.xlabel("# friends")
plt.ylabel('daily minutes')
plt.ylim(0, 100)
prediction = [predict(alpha, beta, friend) for friend in num_friends_good]
plt.plot(num_friends_good, prediction, color='red')
plt.show()


def total_sum_of_squares(y: Vector) -> float:
    return sum(v ** 2 for v in de_mean(y))


def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) / total_sum_of_squares(y))


rsq = r_squared(alpha, beta, num_friends_good, daily_minutes_good)
assert 0.328 < rsq < 0.330

import random
import tqdm
from lesson7_Gradient_Descent import gradient_step

num_epochs = 10000
random.seed(0)
guess = [random.random(), random.random()]
learning_rate = 0.00001
with tqdm.trange(num_epochs) as t:
    for _ in t:
        alpha, beta = guess
        grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                     for x_i, y_i in zip(num_friends_good, daily_minutes_good))
        grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
                     for x_i, y_i in zip(num_friends_good, daily_minutes_good))

        loss = sum_of_sqerrors(alpha, beta,
                               num_friends_good, daily_minutes_good)
        t.set_description(f"loss: {loss:.3f}")
        guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)
    alpha, beta = guess
    assert 22.9 < alpha < 23.0
    assert 0.9 < beta < 0.905