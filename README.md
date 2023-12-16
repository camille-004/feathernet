<h1 align="center" style="display: block; font-size: 2.5em; font-weight: bold; margin-block-start: 1em; margin-block-end: 1em;">
<img align="center" src="assets/imgs/feathernet_logo.png" style="width:50%;height:50%"/>
  <br /><br /><strong>Feathernet</strong>
</h1>

---

## Introduction [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#introduction)

**Feathernet** is a lightweight deep learning and compiler suite. Aptly named for its light-as-a-feather footprint, Feathernet is specifically crafted for those keen on exploring the realms of deep learning and compiler technology. The `dl` package encompasses essential deep learning components, enabling users to effortlessly build, train, and evaluate basic neural network models. The `compiler` package offers tools for graph optimization and Intermediate Representations (IR), positioning itself as a potential resource for understanding and implementing advanced model optimization techniques.

---

## Running Tests [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#run-tests)

To ensure the highest quality and reliability of the code, Feathernet includes a comprehensive suite of unit tests.

Reflecting its dual-package structure, the tests in Feathernet are divided into two main categories.
- Tests for the `dl` package cover deep learning components.
- Tests for the `compiler` package focus on DL compiler components like graph optimization and IR.

### Running Feathernet Tests Locally

To run Feathernet on your local machine, follow these steps:

1. **Clone the Repository**:
    If you haven't already, clone the Feathernet repository:
    ```bash
    git clone https://github.com/camille-004/feathernet.git
    cd feathernet
    ```
2. **Install Poetry**:
    If Poetry is not already installed, you can install it by following the instructions [here](https://python-poetry.org/docs/#installing-with-pipx).
3. **Install Dependencies**:
    Use Poetry to install the project dependencies.
    ```bash
    poetry install
    ```
4. **Activate a Virtual Environment**:
    Activate the Poetry-created virtual environment for the project:
    ```
    poetry shell
    ```
5. **Run Tests**:
    - **Run All Tests**: To run all tests (both `dl` and `compiler`), use the command:
      ```bash
      make test
      ```
