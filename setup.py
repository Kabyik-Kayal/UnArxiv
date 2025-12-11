"""Setup script for the UnArxiv package."""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="UnArxiv",
    version="0.1",
    author="Kabyik",
    packages=find_packages(),
    install_requires = requirements,
)