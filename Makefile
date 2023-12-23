LINT = flake8
FORMAT = black --line-length 79
SORT = isort
LINT_CPP = cpplint
FORMAT_CPP = clang-format -i

SRC_DIR := feathernet/
EXAMPLES_DIR := examples/
TEST_DIR := tests/
DL_TESTS := $(TEST_DIR)dl/
COMPILER_TESTS := $(TEST_DIR)compiler/
CPP_TEMPLATES_DIR := $(SRC_DIR)compiler/codegen/templates/

.PHONY: lint format

lint:
	@echo "Linting code..."
	$(LINT) $(SRC_DIR) $(TEST_DIR)
	@echo "Checking import order..."
	$(SORT) --check-only $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@echo "Linting C++ code..."
	$(LINT_CPP) $(CPP_TEMPLATES_DIR)*.cpp $(CPP_SRC_DIR)*.h

format:
	@echo "Formatting code..."
	$(FORMAT) $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@echo "Sorting imports..."
	$(SORT) $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@#echo "Formatting C++ code..."
	#$(FORMAT_CPP) $(CPP_TEMPLATES_DIR)*.cpp $(CPP_TEMPLATES_DIR)*.h

.PHONY: test test-dl test-compiler

test:
	@echo "Running tests with coverage..."
	coverage run -m unittest discover -s $(TEST_DIR)

test-dl:
	@echo "Running DL with coverage..."
	coverage run -m unittest discover -s $(DL_TESTS)

test-compiler:
	@echo "Running Compiler with coverage..."
	coverage run -m unittest discover -s $(COMPILER_TESTS)
