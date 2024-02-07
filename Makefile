LINT = flake8
FORMAT = black --line-length 79
SORT = isort

SRC_DIR := feathernet/
EXAMPLES_DIR := examples/
TEST_DIR := tests/

.PHONY: lint format

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
	@echo "Running tests locally..."
	python -m coverage run -m unittest discover -s $(TEST_DIR)

test-ci:
	@echo "Running tests for CI..."
	python -m coverage run -m unittest discover -s $(TEST_DIR)
	@echo "Generating coverage report..."
	python -m coverage xml -o coverage.xml
	curl -s https://codecov.io/bash | bash -s -- -y coverage.xml
