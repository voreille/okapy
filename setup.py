#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'pydicom==1.2.2',
    'scikit_image==0.15.0',
    'scipy==1.3.0',
    'numpy==1.16.3',
    'ipdb==0.12',
    'pyradiomics==2.2.0',
    'sympy==1.4',
    'six==1.12.0',
    'Click==7.0',
    'pandas==0.24.2',
    'matplotlib==3.1.0',
    'radiomics==0.1',
    'scikit_learn==0.21.3',
    'SimpleITK==1.2.4',
    'skimage==0.0',
    'tensorflow==2.0.0',
    'termcolor==1.1.0',
],

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Pierre Fontaine, Valentin Oreiller",
    author_email='valentin.oreiller@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="toolbox for biomedical analysis from preprocessing to model evaluation",
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
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/voreille/okapy',
    version='0.1.0',
    zip_safe=False,
)
