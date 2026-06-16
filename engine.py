from collections import defaultdict, deque
from enum import Enum, auto
from typing import Self, override


class OpType(Enum):
    INIT = auto()
    ADD = auto()
    SUB = auto()
    MULT = auto()
    DIV = auto()


class Value:
    def __init__(self, data: float, _children: tuple[Self, ...], _op: OpType) -> None:
        self.data: float = data
        self.grad: float = 0.0
        self._prev: tuple[Value, ...] = _children
        self._op: OpType = _op

    @override
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other: Self) -> Self:
        return type(self)(self.data + other.data, (self, other), OpType.ADD)

    def __sub__(self, other: Self) -> Self:
        return type(self)(self.data - other.data, (self, other), OpType.SUB)

    def __mul__(self, other: Self) -> Self:
        return type(self)(self.data * other.data, (self, other), OpType.MULT)

    def __truediv__(self, other: Self) -> Self:
        return type(self)(self.data / other.data, (self, other), OpType.DIV)

    def back(self) -> None:
        self.grad = 1.0
        cur = self

        children_dict: dict[Value, tuple[Value, ...]] = defaultdict()
        node_queue: deque[Value] = deque()
        node_queue.append(cur)
        children_dict[cur] = cur._prev

        while node_queue:
            # TOOD: Implement topological sort instead of BFS
            next_node = node_queue.pop()
            next_children = children_dict[next_node]

            if not next_children:
                continue

            left, right = next_children
            if not left in children_dict:
                children_dict[left] = right._prev
                node_queue.append(left)
            if not right in children_dict:
                children_dict[right] = right._prev
                node_queue.append(right)

            if next_node._op == OpType.ADD:
                left.grad += next_node.grad
                right.grad += next_node.grad
            elif next_node._op == OpType.SUB:
                left.grad += next_node.grad
                right.grad += -next_node.grad
            elif next_node._op == OpType.MULT:
                left.grad += right.data * next_node.grad
                right.grad += left.data * next_node.grad
            elif next_node._op == OpType.DIV:
                left.grad += (1 / right.data) * next_node.grad
                right.grad += -1 * (left.data / right.data**2) * next_node.grad


def testing():
    a = Value(5.0, (), OpType.INIT)
    b = Value(3.0, (), OpType.INIT)
    c = a / b
    d = a * b
    e = c + d

    print(f"a={a}, b={b}, c={c}, d={d}, e={e}")


if __name__ == "__main__":
    testing()
