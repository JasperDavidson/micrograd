import random

from engine import Value


class Neuron:
    def __init__(
        self,
        input_data: tuple[Value, ...],
        weights: tuple[Value, ...] | None = None,
        bias: Value | None = None,
    ) -> None:
        self.input_data: tuple[Value, ...] = input_data

        if weights is None:
            self.weights: tuple[Value, ...] = tuple(
                Value(random.uniform(-1, 1)) for _ in range(len(input_data))
            )
        else:
            self.weights = weights

        if bias is None:
            self.bias: Value = Value(random.uniform(-1, 1))
        else:
            self.bias = bias

    def apply(self) -> Value:
        out = sum((i * w for i, w in zip(self.input_data, self.weights)), self.bias)
        return out.tanh()
