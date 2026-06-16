from engine import Value


class Neuron:
    def __init__(
        self, input_data: tuple[Value, ...], weights: tuple[Value, ...], bias: Value
    ) -> None:
        self.input_data: tuple[Value, ...] = input_data
        self.weights: tuple[Value, ...] = weights
        self.bias: Value = bias

    def apply(self) -> Value:
        out = sum((i * w for i, w in zip(self.input_data, self.weights)), self.bias)
        return out.tanh()
