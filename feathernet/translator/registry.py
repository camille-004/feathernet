from typing import Callable


class OperationRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, Callable] = {}

    def register(self, operation_name: str, handler: Callable) -> None:
        self._registry[operation_name] = handler

    def get_handler(self, operation_name: str) -> Callable:
        return self._registry.get(operation_name)


operation_registry = OperationRegistry()


def register_op(operation_name: str) -> Callable:
    def decorator(handler: Callable) -> Callable:
        operation_registry.register(operation_name, handler)
        return handler

    return decorator
