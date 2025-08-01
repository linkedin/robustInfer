.PHONY: venv build test clean build_and_test

# Create a virtual environment
venv:
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

# # Build the package
py-build:
	python3 setup.py sdist bdist_wheel

scala-build:
	gradle -p scala_lib build

build: py-build scala-build

# Run unit tests
py-test: venv
	./venv/bin/pip install -e . 
	./venv/bin/pytest python_lib/tests/

scala-test:
	gradle -p scala_lib test

test: scala-test py-test

# Clean up Scala build artifacts using Gradle
scala-clean:
	gradle -p scala_lib clean

# Clean up build artifacts and virtual environment
clean-eggs:
	find . -type d -name '*.egg-info' -exec rm -rf {} +

clean: clean-eggs
	rm -rf dist build *.egg-info venv
	make scala-clean

clean-all: clean
	git clean -fdX

# Build and test in one step
build_and_test: build test
