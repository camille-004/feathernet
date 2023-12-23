from typing import Any

import numpy as np

from feathernet.compiler.codegen import NetworkCodeGen
from feathernet.compiler.ir import convert_model_to_ir
from feathernet.dl import Network
from feathernet.dl.optimizers import Optimizer
from feathernet.frontend import DataLoader, FrontendExecutor


class ModelTrainingInterface:
    def __init__(
        self,
        model: Network,
        dataset: dict[str, np.ndarray],
        optimizer: Optimizer,
        training_params: dict[str, Any],
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.training_params = training_params
        self.executor: FrontendExecutor = None

    def setup_training(self) -> None:
        model_ir = convert_model_to_ir(self.model, optimizations={})
        network_codegen = NetworkCodeGen(model_ir)
        data_loader = DataLoader(
            self.dataset["X"],
            self.dataset["y"],
            batch_size=self.training_params["batch_size"],
        )
        self.executor = FrontendExecutor(
            network_codegen, self.optimizer, self.training_params, data_loader
        )

    def train(self) -> str:
        if self.executor is None:
            self.setup_training()

        training_output = self.executor.train()
        return training_output
