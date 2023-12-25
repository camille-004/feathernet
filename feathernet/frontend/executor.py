from subprocess import PIPE, Popen
from typing import Any

import numpy as np

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
        self.executor = Executor(source)
        self.compiled = self.executor.compile()

        if not self.compiled:
            raise RuntimeError("Failed to compile the generated model code.")

    def batch_to_str(self, X_batch: np.ndarray, y_batch: np.ndarray) -> str:
        batch_data_str = ""
        for x, y in zip(X_batch, y_batch):
            batch_data_str += " ".join(map(str, x)) + " " + str(y) + "\n"
        return batch_data_str

    def train(self) -> str:
        if not self.compiled:
            self.generate_and_compile()

        with Popen(
            self.executor.binary_path,
            stdin=PIPE,
            text=True,
            bufsize=1,
        ) as process:
            try:
                for X_batch, y_batch in self.data_loader:
                    batch_data_str = self.batch_to_str(X_batch, y_batch)
                    # print("Sending batch data:", batch_data_str)
                    process.stdin.write(batch_data_str)
                    process.stdin.flush()
            except Exception as e:
                print(f"Error while sending data: {e}")

            stdout, stderr = process.communicate()
            if stderr:
                print("Error:", stderr)

            if process.returncode != 0:
                raise RuntimeError(
                    f"Process exited with code {process.returncode}"
                )

        return stdout

    def cleanup(self) -> None:
        if self.executor:
            self.executor.cleanup()
