#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "numpy",
    "pydicom",
    "pydicom_seg",
    "scikit-image",
    "pandas",
    "click",
    "tqdm",
    "pyyaml",
    "SimpleITK",
    "pyradiomics",
],

setup_requirements = []

test_requirements = []

setup(
    author="Pierre Fontaine, Valentin Oreiller",
    author_email='valentin.oreiller@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description=
    "toolbox for biomedical analysis from preprocessing to model evaluation",
    entry_points={
        'console_scripts': [
            'okapy=okapy.cli:main',
            'okapyconvert=okapy.dicomconverter.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='okapy',
    name='okapy',
    packages=find_packages(exclude=['tests']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/voreille/okapy',
    version='0.1.3',
    zip_safe=False,
)
