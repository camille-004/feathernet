<h1 align="center" style="display: block; font-size: 2.5em; font-weight: bold; margin-block-start: 1em; margin-block-end: 1em;">
<img align="center" src="assets/imgs/feathernet_logo.png" style="width:50%;height:50%"/>
  <br /><br /><strong>Feathernet</strong>
</h1>

![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/t/camille-004/feathernet?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/camille-004/feathernet?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/camille-004/feathernet?style=social)
![GitHub forks](https://img.shields.io/github/forks/camille-004/feathernet?style=social)

---

## Project Status [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#project-status)

<table class="no-border">
  <tr>
    <td><img alt="Codecov" src="https://img.shields.io/codecov/c/github/camille-004/feathernet?style=for-the-badge"></td>
    <td><img alt="GitHub Workflow Status (with event)" src="https://img.shields.io/github/actions/workflow/status/camille-004/feathernet/ci.yml?style=for-the-badge"></td>
    <td><img alt="GitHub top language" src="https://img.shields.io/github/languages/top/camille-004/feathernet?style=for-the-badge"></td>
</table>

---

## Introduction [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#introduction)

**Feathernet** is a lightweight deep learning and compiler suite. Aptly named for its light-as-a-feather footprint, Feathernet is specifically crafted for those keen on exploring the realms of deep learning and compiler technology. The `dl` package encompasses essential deep learning components, enabling users to effortlessly build, train, and evaluate basic neural network models. The `compiler` package offers tools for graph optimization and Intermediate Representations (IR), positioning itself as a potential resource for understanding and implementing advanced model optimization techniques.

---

## Table of Contents [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#table-of-contents)

- [Motivation](#motivation-)
- [From Scratch in PYthon](#from-scratch-in-python-)
- [Composition](#composition-)
- [Running Tests](#running-tests-)
    - [Running Tests Locally](#running-tests-locally)
    - [Running Tests in the Docker Container](#running-tests-in-the-docker-container)

---

## Motivation [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#motivation)

Deep learning compilers translate high-level neural network models into optimized, hardware-specific code, bridging the gap between theoretical model design and practical application. Examples of prominent deep learning compilers include [TensorRT](https://developer.nvidia.com/tensorrt) (inference optimization), [XLA (*Accelerated Linear Algebra*)](https://www.tensorflow.org/xla), and [Glow](https://ai.meta.com/tools/glow/).

<div align="center"><img src="assets/imgs/dl_compiler.png" alt="Deep Learning Compiler Framework" style="width:40%;height:40%"/></div>

The venture into deep learning compilers is driven by the need to:
- *Enhance Computational Efficiency*: Optimizing neural network models for specific hardware accelerates performance and reduces resource consumption.
- *Facilitate Cross-Platform Compatibility*: Compilers enable models to run seamlessly across different types of hardware.
- *Democratize AI Development*: By simplifying the process of model optimization, compilers make AI development more accessible to a broader range of developers and researchers.

Feathernet aims to provide an intuitive understanding of these concepts through its `compiler` package, offering tools for graph optimization and IRs.

Key resources for further reading on deep learning compilers:
- [AI Compilers Demystified](https://medium.com/geekculture/ai-compilers-ae28afbc4907)
- [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)

<div align="right">[ <a href="#table-of-contents">↑ Back to top ↑</a> ]</div>

---

## From Scratch in Python [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#from-scratch-in-python)

Feathernet's journey of being developed from scratch in Python is driven by an ambition to unveil the synergy between DL libraries and compilers. This allows us to observe and understand how these two critical components of AI technology work in tandem.

Key aspects of this approach include:
- *Integrated Development*: Simultaneously crafting a deep learning library and a compiler reveals the intricate interplay between model creation and optimization. It provides a holistic view of how neural networks are translated into efficient, executable code.
- *Focus on CPU as Target*: While the deep learning field often focuses on GPU optimizations, Feathernet places its emphasis on CPU capabilities. This choice offers insides into how compilers optimize for different hardware architectures, particulraly the more universally accessible CPUs.
- *Practical Learning Experience*: This dual development approach grants practical insights into compiler development challenges, particularly in translating high-level neural network models into executable code.

> 💡 The code in the `dl` package of the Feathernet suite is intentionally less optimized. This is because the focus is more on the compiler aspect, emphasizing and understanding compiler technologies rather than optimizing the deep learning components themselves.

<div align="right">[ <a href="#table-of-contents">↑ Back to top ↑</a> ]</div>

---

## Composition [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#composition)

Feathernet is composed of two primary packages.
- [`dl`](https://github.com/camille-004/feathernet/tree/main/feathernet/dl): This package encompasses core deep learning components, including layers, optimizers, initializers, and losses. Additionally, each component in the `dl` package is equipped with serialization capabilities, enabling integration with the compiler's Intermediate Representation (IR).
- [`compiler`](https://github.com/camille-004/feathernet/tree/main/feathernet/compiler): This package offers tools and modules for graph optimization and Intermediate Representation (IR). It includes functionalities for layer fusion, pruning, and quantization.

<div align="right">[ <a href="#table-of-contents">↑ Back to top ↑</a> ]</div>

---

## Running Tests [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#run-tests)

To ensure the highest quality and reliability of the code, Feathernet includes a comprehensive suite of unit tests.

Reflecting its dual-package structure, the tests in Feathernet are divided into two main categories.
- Tests for the `dl` package cover deep learning components.
- Tests for the `compiler` package focus on DL compiler components like graph optimization and IR.

### Running Tests Locally

| **Prerequisite** | **Installation** |
| --- | ---|
| `make` | [GNU Make](https://www.gnu.org/software/make/) |
| `poetry` | [Poetry](https://python-poetry.org/docs/#installation) |

To run Feathernet on your local machine, follow these steps:

1. **Clone the Repository**:
    If you haven't already, clone the Feathernet repository:
    ```bash
    git clone https://github.com/camille-004/feathernet.git
    cd feathernet
    ```
2. **Install Dependencies**:
    Use Poetry to install the project dependencies.
    ```bash
    poetry install
    ```
3. **Activate a Virtual Environment**:
    Activate the Poetry-created virtual environment for the project:
    ```
    poetry shell
    ```
4. **Run Tests**:
    - **Run All Tests**: To run all tests (both `dl` and `compiler`), use the command:
      ```bash
      make test
      ```
    - **Run DL Tests Only**: To run only the deep learning tests, use:
      ```bash
      make test-dl
      ```
    - **Run Compiler Tests Only**: To run only compiler tests, use:
      ```bash
      make test-compiler
      ```

### Running Tests in the Docker Container

Feathernet is available as a Docker container. You can pull the latest Feathernet Docker container from GitHub Container Registry.

1.  **Pull the Docker Image**:
    ```bash
    docker pull ghcr.io/camille-004/feathernet:latest
    ```
2. **Run the Docker Container**:
    Start a container from the image. This will open an interactive shell:
    ```bash
    docker run -it --rm ghcr.io/camille-004/feathernet:latest /bin/bash
   ```
3. **Execute the Commands Inside the Container**:
    ```bash
   make test
    ```
4. **Exit the Container**:
    Type `exit` to leave the container.

<div align="right">[ <a href="#table-of-contents">↑ Back to top ↑</a> ]</div>
