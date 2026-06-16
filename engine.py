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
            if not cur_node._prev[0] in dfs_visited:
                dfs_stack.append(cur_node._prev[0])
            if not cur_node._prev[1] in dfs_visited:
                dfs_stack.append(cur_node._prev[1])

        while topo_sort_nodes:
            cur_node = topo_sort_nodes.pop()

            if not cur_node._prev:
                continue

            left, right = cur_node._prev
            if cur_node._op == OpType.ADD:
                left.grad += cur_node.grad
                right.grad += cur_node.grad
            elif cur_node._op == OpType.SUB:
                left.grad += cur_node.grad
                right.grad += -cur_node.grad
            elif cur_node._op == OpType.MULT:
                left.grad += right.data * cur_node.grad
                right.grad += left.data * cur_node.grad
            elif cur_node._op == OpType.DIV:
                left.grad += (1 / right.data) * cur_node.grad
                right.grad += -1 * (left.data / right.data**2) * cur_node.grad


def testing():
    a = Value(5.0, (), OpType.INIT)
    b = Value(3.0, (), OpType.INIT)
    c = a / b
    d = a * b
    e = c + d

    e.back()
    print(f"a.grad={a.grad}, b.grad={b.grad}")


if __name__ == "__main__":
    testing()
