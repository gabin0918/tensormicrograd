from setuptools import setup, find_packages

setup(
    name="tensormicrograd",
    version="0.1",
    description="A tensor-based autograd engine with broadcasting support, based on Andrej Karpathy's micrograd.",
    author="gabin0918",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    python_requires=">=3.6",
)