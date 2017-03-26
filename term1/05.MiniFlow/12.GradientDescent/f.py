"""
Given the starting point of any `x` gradient descent
should be able to find the minimum value of x for the
cost function `f` defined below.
"""
import random
from gd import gradient_descent_update


def f(x):
    """
    Quadratic function.

    It's easy to see the minimum value of the function
    is 5 when is x=0.
    """
    return x**2 + 5


def df(x):
    """
    Derivative of `f` with respect to `x`.
    """
    return 2*x


"""
Notes:

Setting the learning_rate to 0.1 should result in
x -> 0 and f(x) -> 5 if you've implemented gradient descent correctly.

Play around with different values for the learning rate.
Try very small values, values close to 1, above 1, etc. What happens?
"""
# Random number better 0 and 10,000. Feel free to set x whatever you like.
x = random.randint(0, 10000)
# TODO: Set the learning rate
learning_rate = 0.5
epochs = 100

for i in range(epochs+1):
    cost = f(x)
    gradx = df(x)
    print("EPOCH {}: Cost = {:.3f}, x = {:.3f}".format(i, cost, gradx))
    x = gradient_descent_update(x, gradx, learning_rate)
