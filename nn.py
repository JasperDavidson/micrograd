import random

from engine import Value


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):
    def __init__(
        self,
        nin: int,
        weights: list[Value] | None = None,
        bias: Value | None = None,
    ) -> None:
        if weights is None:
            self.weights: list[Value] = [
                Value(random.uniform(-1, 1)) for _ in range(nin)
            ]
        else:
            self.weights = weights

        if bias is None:
            self.bias: Value = Value(random.uniform(-1, 1))
        else:
            self.bias = bias

    def __call__(self, input_data: list[Value]) -> Value:
        out = sum((i * w for i, w in zip(input_data, self.weights)), self.bias)
        return out.tanh()

    def parameters(self) -> list[Value]:
        return self.weights + [self.bias]


class Layer(Module):
    def __init__(self, nin: int, layer_size: int) -> None:
        self.neurons: list[Neuron] = [Neuron(nin) for _ in range(layer_size)]

    def __call__(self, input_data: list[Value]) -> list[Value]:
        out = [n(input_data) for n in self.neurons]
        return out

    def parameters(self) -> list[Value]:
        params = []
        for n in self.neurons:
            params.extend(n.parameters())

        return params


class MLP(Module):
    def __init__(self, nin: int, layer_sizes: list[int]) -> None:
        input_sizes = [nin] + layer_sizes

        self.layers: list[Layer] = [
            Layer(nin=n, layer_size=l) for n, l in zip(input_sizes, layer_sizes)
        ]

    def __call__(self, input_data: list[Value]) -> list[Value]:
        for layer in self.layers:
            input_data = layer(input_data)

        return input_data

    def parameters(self) -> list[Value]:
        params = []
        for l in self.layers:
            params.extend(l.parameters())

        return params


def compute_loss(actual: list[list[Value]], expected: list[list[Value]]) -> Value:
    # Ensure the batches match in size
    if len(actual) != len(expected):
        raise ValueError("Batch sizes do not match.")

    total_elements = 0
    total_loss = Value(0)
    for act_vals, exp_vals in zip(actual, expected):
        for act, exp in zip(act_vals, exp_vals):
            total_loss += (exp - act) ** 2
            total_elements += 1

    if total_elements == 0:
        return Value(0)

    return total_loss * (1 / total_elements)


learning_rate = 0.01
loss_tolerance = 0.01


def grad_descent(
    input_data: list[list[Value]], expected_data: list[list[Value]], mlp: MLP
) -> None:
    loss = Value(float("inf"))

    while loss.data > loss_tolerance:
        # Zero prior grad
        mlp.zero_grad()

        # Perform forward -> backward pass
        out = [mlp(x) for x in input_data]
        loss = compute_loss(out, expected_data)

        loss.back()

        # Update params according to loss
        for p in mlp.parameters():
            p.data += -learning_rate * p.grad


def basic_test():
    input_data = [Value(1), Value(2), Value(3)]
    layer_sizes = [4, 4, 1]
    mlp = MLP(len(input_data), layer_sizes)
    mlp_out = mlp(input_data)
    print(mlp_out)


def training_test():
    xs_pure = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]

    xs = [Value.convert_to_vals(x) for x in xs_pure]
    ys = [[Value(1.0)], [Value(1.0)], [Value(1.0)], [Value(-1.0)]]

    layer_sizes = [4, 4, 1]
    mlp = MLP(len(xs[0]), layer_sizes)

    print("Before grad descent: ", compute_loss([mlp(x) for x in xs], ys))
    grad_descent(xs, ys, mlp)
    print("After grad descent: ", compute_loss([mlp(x) for x in xs], ys))


if __name__ == "__main__":
    training_test()
