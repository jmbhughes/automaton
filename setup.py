#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='automaton',
    python_requires='>=3.7',
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
    install_requires=["numpy", "scipy", "matplotlib", "scikit-image", "networkx", "pygraphviz"],
    version='0.0.2',
    author='J. Marcus Hughes',
    author_email='hughes.jmb@gmail.com',
    packages=find_packages(),
    url='',
    description='A simple implementation of an automaton',
    license='LICENSE.txt',
    long_description=open('readme.md').read(),
    long_description_content_type="text/markdown",
)
