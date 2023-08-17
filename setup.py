# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="prompt_risk",
    version="0.0.1",
    description="Distribution-free guarantees for LLM prompts",
    long_description=readme,
    author="anonymous",
    author_email="anonymous",
    url="anonymous",
    license=license,
    packages=find_packages(),
)