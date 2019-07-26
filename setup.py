import os
import re
from setuptools import setup, find_packages


def read_version(filename):
    return re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
read(filename), re.MULTILINE).group(1)


def read(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as infile:
        text = infile.read()
    return text


docs_require = ['sphinx>=1.7', 'sphinx_paramlinks']
tests_require = ['flake8>=3.7.7', 'mypy', 'pytest', 'pytest-cov',
                 'pytest-mock', 'pylint', 'pycodestyle>=2.1.0',
                 'pyflakes>=2.1.0']

setup(
    name='rads',
    version=read_version('rads/__init__.py'),
    author='Michael R. Shannon',
    author_email='mrshannon.aerospace@gmail.com',
    description='Python front end for the Radar Altimeter Database System.',
    long_description=read('README.rst'),
    license='MIT',
    url='https://github.com/ccarocean/pyrads',
    packages=find_packages(),
    setup_requires=['pytest-runner'],
    install_requires=[
        'appdirs',
        'astropy',
        'cached_property',
        'cf_units>=2.1.1',
        'dataclasses;python_version=="3.6"',
        'dataclass-builder>=1.1.2',
        'fortran_format_converter>=0.1.2',
        'numpy>=1.16.0',
        'scipy',
        'wrapt',
        'yzal',
    ],
    extras_require={
        'libxml2': ['lxml'],  # use libxml2 to read configuration files
        'docs': docs_require,
        'tests': tests_require,
        'dev': docs_require + tests_require
    },
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
