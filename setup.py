import os
import re
from setuptools import setup, find_packages


def read(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as infile:
        text = infile.read()
    return text


setup(
    name='rads',
    version='4.0.1a1',
    author='Michael R. Shannon',
    author_email='mrshannon.aerospace@gmail.com',
    description='Python front end for the Radar Altimeter Database System.',
    long_description=read('README.rst'),
    license='MIT',
    url='https://github.com/ccarocean/pyrads',
    packages=find_packages(),
    setup_requires=['pytest-runner'],
    install_requires=[
        'numpy',
        'scipy',
        'lxml',
        'yzal',
        'cached_property'
    ],
    tests_require=[
        'pytest',
        'pytest-mock',
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Database :: Front-Ends'
        'Topic :: Scientific/Engineering'
        'Topic :: Scientific/Engineering :: GIS'
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    zip_safe=False
)
