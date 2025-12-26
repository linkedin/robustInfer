.PHONY: install build test clean lint format docker-build docker-test

# =============================================================================
# PYTHON TARGETS
# =============================================================================

# Install Python dependencies using UV
python-install:
	cd python_lib && uv sync --group dev

# Build Python package using UV + Hatchling
python-build:
	cd python_lib && uv build

# Run Python tests with UV + Tox
python-test:
	cd python_lib && uv run tox -e py313

# Lint Python code with Ruff
python-lint:
	cd python_lib && uv run ruff check .

# Format Python code with Ruff
python-format:
	cd python_lib && uv run ruff format .

# Clean up Python build artifacts
python-clean:
	cd python_lib && rm -rf dist build *.egg-info .tox .pytest_cache .ruff_cache
	rm -rf dist build *.egg-info

# =============================================================================
# SCALA TARGETS
# =============================================================================

# Install Scala dependencies (if needed)
scala-install:
	gradle -p scala_lib dependencies

# Build Scala package
scala-build:
	gradle -p scala_lib build

# Run Scala tests
scala-test:
	gradle -p scala_lib test

# Format Scala code with scalafmt
scala-format:
	gradle -p scala_lib format

# Check Scala formatting
scala-format-check:
	gradle -p scala_lib checkFormat

# Lint Scala code (check formatting + compiler warnings)
scala-lint:
	gradle -p scala_lib lint

# Clean up Scala build artifacts
scala-clean:
	gradle -p scala_lib clean

# =============================================================================
# COMBINED TARGETS
# =============================================================================

# Install all dependencies
install: python-install scala-install

# Build both Python and Scala
build: python-build scala-build

# Run all tests
test: scala-test python-test

# Lint both Python and Scala
lint: python-lint scala-lint

# Format both Python and Scala  
format: python-format scala-format

# Check formatting without applying changes
format-check: scala-format-check
	cd python_lib && uv run ruff format --check .

# Clean up all build artifacts
clean-eggs:
	find . -type d -name '*.egg-info' -exec rm -rf {} +

clean: python-clean scala-clean clean-eggs

clean-all: clean
	git clean -fdX

# Build and test in one step
build_and_test: build test

# =============================================================================
# DOCKER TARGETS
# =============================================================================

# Extract Spark version profile from gradle.properties
SPARK_PROFILE := $(shell grep 'versionProfile=' scala_lib/gradle.properties | cut -d'=' -f2)

# Map profiles to Spark and Hadoop versions
SPARK_VERSION_spark31 := 3.1.3
HADOOP_VERSION_spark31 := 3.2

SPARK_VERSION_spark34 := 3.4.1
HADOOP_VERSION_spark34 := 3

SPARK_VERSION_spark35 := 3.5.0
HADOOP_VERSION_spark35 := 3

# Get versions for current profile
SPARK_VERSION := $(SPARK_VERSION_$(SPARK_PROFILE))
HADOOP_VERSION := $(HADOOP_VERSION_$(SPARK_PROFILE))

docker-build:
	@echo "Building Docker image with Spark $(SPARK_VERSION) (profile: $(SPARK_PROFILE))"
	docker build \
		--build-arg SPARK_VERSION=$(SPARK_VERSION) \
		--build-arg HADOOP_VERSION=$(HADOOP_VERSION) \
		-t robustinfer-notebook .

docker-test:
	docker run robustinfer-notebook python -c "import robustinfer; print('Python import successful')"
