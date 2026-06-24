import random

from engine import Value


class Neuron:
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


class Layer:
    def __init__(self, nin: int, layer_size: int) -> None:
        self.neurons: list[Neuron] = [Neuron(nin) for _ in range(layer_size)]

    def __call__(self, input_data: list[Value]) -> list[Value]:
        out = [n(input_data) for n in self.neurons]
        return out


class MLP:
    def __init__(self, nin: int, layer_sizes: list[int]) -> None:
        input_sizes = [nin] + layer_sizes

        self.layers: list[Layer] = [
            Layer(nin=n, layer_size=l) for n, l in zip(input_sizes, layer_sizes)
        ]

    def __call__(self, input_data: list[Value]) -> list[Value]:
        for layer in self.layers:
            input_data = layer(input_data)

        return input_data


def setup():
    input_data = [Value(1), Value(2), Value(3)]
    layer_sizes = [4, 4, 1]
    mlp = MLP(len(input_data), layer_sizes)
    mlp_out = mlp(input_data)
    print(mlp_out)


if __name__ == "__main__":
    setup()
