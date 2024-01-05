LINT = flake8
FORMAT = black --line-length 79
SORT = isort
LINT_CPP = cpplint
FORMAT_CPP = clang-format -i
CMAKE = cmake
MAKE = make

SRC_DIR := feathernet/
EXAMPLES_DIR := examples/
TEST_DIR := tests/
COMPILER_TESTS := $(TEST_DIR)compiler/
BUILD_DIR := $(CPP_DIR)build/

.PHONY: lint format build-cpp

lint:
	@echo "Linting code..."
	$(LINT) $(SRC_DIR) $(TEST_DIR)
	@echo "Checking import order..."
	$(SORT) --check-only $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

format:
	@echo "Formatting code..."
	$(FORMAT) $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@echo "Sorting imports..."
	$(SORT) $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

.PHONY: test

test:
	@echo "Running tests with coverage..."
	coverage run -m unittest discover -s $(TEST_DIR)
