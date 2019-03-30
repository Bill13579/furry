import platform
from setuptools import setup, Extension, find_packages

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(name="furry",
    version="1.0.0",
    description="",
    long_description=long_description,
    author="Bill Kudo",
    author_email="bluesky42624@gmail.com",
    license="MIT",
    packages=find_packages(),
    scripts=[],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3"
    ])
