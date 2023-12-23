from sklearn.datasets import make_moons

from feathernet.dl import Network
from feathernet.dl.layers import Dense
from feathernet.dl.initializers import he_initializer
from feathernet.dl.optimizers import SGD
from feathernet.frontend import ModelTrainingInterface

if __name__ == "__main__":
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    dataset = {"X": X, "y": y}

    network = Network()

    input_dim = 2
    hidden_dim1 = 16
    hidden_dim2 = 8
    output_dim = 1

    network.add(Dense(input_dim, hidden_dim1, initializer=he_initializer))
    network.add(Dense(hidden_dim1, hidden_dim2, initializer=he_initializer))
    network.add(Dense(hidden_dim2, output_dim, initializer=he_initializer))

    optimizer = SGD(0.01)
    training_params = {"num_epochs": 10, "batch_size": 32}

    trainer = ModelTrainingInterface(
        network, dataset, optimizer, training_params
    )
    training_output = trainer.train()

    print(training_output)
