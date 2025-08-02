#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for lammphonon
"""

from setuptools import setup, find_packages

# 读取README.md作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lammphonon",
    version="1.0.0",
    author="Shuming Liang",
    author_email="lsm315@mail.ustc.edu.cn",
    description="Phonon Analysis Toolkit for LAMMPS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liangshuming/lammphonon",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.16.0",
        "scipy>=1.2.0",
        "matplotlib>=3.0.0",
    ],
    scripts=["bin/lammphonon_menu"],
    entry_points={
        "console_scripts": [
            "lammphonon=lammphonon.menu.main_menu:main",
        ],
    },
) 