from lesson3 import Vector, dot


def step_function(x: float) -> float:
    return 1.0 if x >= 0 else 0.0


def perceptron_output(weights: Vector, bias:  float, x: Vector) -> float:
    calculation = dot(weights, x) + bias
    return step_function(calculation)


import math


def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))


def neuron_output(weights: Vector, inputs: Vector) -> float:
    return sigmoid(dot(weights, inputs))


from typing import List


def feed_forward(neural_network: List[List[Vector]],
                 input_vector: Vector) -> List[Vector]:
    outputs: List[Vector] = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]
        outputs.append(output)
        input_vector = output
    return outputs


def sqerror_gradients(network: List[List[Vector]],
                      input_vector: Vector,
                      target_vector: Vector) -> List[List[Vector]]:

    hidden_outputs, outputs = feed_forward(network, input_vector)
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]
    output_grads = [[output_deltas[i] * hidden_output
                     for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]

    hidden_deltas = [hidden_output * (1 - hidden_output) *
                     dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]

    return [hidden_grads, output_grads]



import random


random.seed(0)

xs = [[0., 0], [0., 1], [1.0, 0], [1., 1]]
ys = [[0.], [1.], [1.], [0.]]

network = \
    [[[random.random() for _ in range(2 + 1)],
           [random.random() for _ in range(2 + 1)]],
     [[random.random() for _ in range(2 + 1)]]
]

from lesson7_Gradient_Descent import gradient_step
import tqdm

learninng_rate = 1.0

for epoch in tqdm.trange(20000, desc='neural net for xor'):
    for x,y in zip(xs, ys):
        gradients = sqerror_gradients(network, x, y)

        network = [[gradient_step(neuron, grad, -learninng_rate)
                    for neuron, grad in zip(layer, layer_grad)]
                   for layer, layer_grad in zip(network, gradients)]


def fizz_buzz_encode(x: int) -> Vector:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x: int) -> Vector:
    binary: List[float] = []
    for i in range(10):
        binary.append(x % 2)
        x = x // 2

    return binary


xs = [binary_encode(n) for n in range(101, 1024)]
ys = [fizz_buzz_encode(n) for n in range(101, 1024)]
NUM_HIDDEN = 25
network = [
    [[random.random() for _ in range(10 + 1) for _ in range(NUM_HIDDEN)]],
    [[random.random() for _ in range(NUM_HIDDEN + 1) for _ in range(4)]]
]

from lesson3 import squared_distance


learninng_rate = 1.0

with tqdm.trange(500) as t:
    for epoch in t:
        epoch_loss = 0.0
    for x, y in zip(xs, ys):
        predicted = feed_forward(network, x)[-1]
        epoch_loss += squared_distance(predicted, x)
        gradients = sqerror_gradients(network, x, y)
        network = [[gradient_step(neuron, grad, -learninng_rate)
                    for neuron, grad in zip(layer, layer_grad)]
                   for layer, layer_grad in zip(network, gradients)]
        t.set_description(f"fizz buzz (loss: {epoch_loss:.2f})")


def argmax(xs: List) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


num_correct = 0
for n in range(1, 101):
    x = binary_encode(n)
    predicted = argmax(feed_forward(network, x)[-1])
    actual = argmax(fizz_buzz_encode(n))
    labels = [str(n), "fizz", "buzz", "fizzbuzz"]
    print(n, labels[predicted], labels[actual])
    if predicted == actual:
        num_correct += 1

print(num_correct, "/", 100)