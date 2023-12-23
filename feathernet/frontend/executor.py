from typing import Any

from feathernet.compiler.codegen import Executor, NetworkCodeGen
from feathernet.dl.optimizers import Optimizer
from feathernet.frontend import DataLoader


class FrontendExecutor:
    def __init__(
        self,
        network_codegen: NetworkCodeGen,
        optimizer: Optimizer,
        training_params: dict[str, Any],
        data_loader: DataLoader,
    ) -> None:
        self.network_codegen = network_codegen
        self.optimizer = optimizer
        self.training_params = training_params
        self.data_loader = data_loader
        self.executor: Executor = None
        self.compiled: bool = False

    def generate_and_compile(self) -> None:
        source = self.network_codegen.generate_network_code(
            self.optimizer, self.training_params
        )
        print(source)
        self.executor = Executor(source)
        self.compiled = self.executor.compile()

        if not self.compiled:
            raise RuntimeError("Failed to compile the generated model code.")

    def run_training(self) -> str:
        if not self.compiled:
            raise RuntimeError(
                "Code must be compiled before running training."
            )

        output = self.executor.exec()
        self.executor.cleanup()
        return output

    def train(self) -> str:
        self.generate_and_compile()
        training_output = self.run_training()
        return training_output
