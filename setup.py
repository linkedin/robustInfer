from setuptools import setup, find_packages

setup(
    name="robustinfer",
    version="0.1.0",
    packages=find_packages(where="python_lib/src"),
    package_dir={"": "python_lib/src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "jax",
        "dataclasses"
    ],
    description="A Python library for robust inference",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # url="https://github.com/<TODO: new repo>",
    author="chawei",
    license="Apache-2.0",
)