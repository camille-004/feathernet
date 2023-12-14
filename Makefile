LINT = flake8
FORMAT = black --line-length 79
SORT = isort

SRC_DIR := feathernet/
TEST_DIR := tests/

.PHONY: lint format

lint:
	@echo "Linting code..."
	$(LINT) $(SRC_DIR) $(TEST_DIR)
	@echo "Checking import order..."
	$(SORT) --check-only $(SRC_DIR) $(TEST_DIR)

format:
	@echo "Formatting code..."
	$(FORMAT) $(SRC_DIR) $(TEST_DIR)
	@echo "Sorting imports..."
	$(SORT) $(SRC_DIR) $(TEST_DIR)

.PHONY: test-coverage

test-coverage:
	coverage run -m unittest discover -s $(TEST_DIR)
	coverage report
	coverage report --fail-under=80
