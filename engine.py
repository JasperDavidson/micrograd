import math
from collections.abc import Callable
from enum import Enum, auto
from typing import Self, override


class OpType(Enum):
    INIT = auto()
    NEG = auto()
    ADD = auto()
    SUB = auto()
    MULT = auto()
    DIV = auto()
    EXP = auto()
    TANH = auto()


def tanh_calc(x: float) -> float:
    return (1 - math.exp(-2 * x)) / (1 + math.exp(-2 * x))


class Value:
    def __init__(
        self, data: float, _children: tuple[Value, ...] = (), _op: OpType = OpType.INIT
    ) -> None:
        self.data: float = data
        self.grad: float = 0.0
        self._back: Callable[[], None] = lambda: None
        self._prev: tuple[Value, ...] = _children
        self._op: OpType = _op

    @override
    def __repr__(self):
        return f"Value(data={self.data})"

    ### Unary operations

    def __neg__(self) -> Value:
        out = type(self)(-self.data, (self,), OpType.NEG)

        def back() -> None:
            self.grad += -1.0 * out.grad

        out._back = back
        return out

    def __pow__(self, num: float) -> Value:
        out = Value(math.pow(self.data, num), (self,), OpType.INIT)

        def back() -> None:
            self.grad += out.grad * num * (self.data ** (num - 1))

        out._back = back
        return out

    def tanh(self) -> Self:
        tanh_data = tanh_calc(self.data)
        out = type(self)(tanh_data, (self,), OpType.TANH)

        def back() -> None:
            self.grad += out.grad * (1 - (tanh_calc(self.data)) ** 2)

        out._back = back
        return out

    ### Binary operations

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = type(self)(self.data + other.data, (self, other), OpType.ADD)

        def back() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._back = back
        return out

    def __radd__(self, num: float) -> Value:
        return self + Value(num, (), OpType.INIT)

    def __sub__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __rsub__(self, other: float) -> Value:
        return Value(other) - self

    def __mul__(self, other: Value | float) -> Self:
        other = other if isinstance(other, Value) else Value(other)
        out = type(self)(self.data * other.data, (self, other), OpType.MULT)

        def back() -> None:
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._back = back
        return out

    def __rmul__(self, num: float) -> Value:
        return self * num

    def __truediv__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** (-1))

    def __rtruediv__(self, num: float) -> Value:
        return Value(num) / self

    def back(self) -> None:
        self.grad = 1.0
        topo_sort_nodes: list[Value] = []
        dfs_stack: list[Value] = [self]
        dfs_visited: set[Value] = set()

        while dfs_stack:
            cur_node = dfs_stack[-1]

            if not cur_node._prev or cur_node in dfs_visited:
                _ = dfs_stack.pop()
                topo_sort_nodes.append(cur_node)
                continue

            dfs_visited.add(cur_node)
            for child in cur_node._prev:
                if not child in dfs_visited:
                    dfs_stack.append(child)

        while topo_sort_nodes:
            cur_node = topo_sort_nodes.pop()
            cur_node._back()


def testing():
    a = Value(5.0)
    b = Value(3.0)
    c = a / b
    d = a * b
    e = c + d

    e.back()
    print(f"a.grad={a.grad}, b.grad={b.grad}")


if __name__ == "__main__":
    testing()
