import os
import re
from pathlib import Path

from setuptools import find_packages, setup

_SETUP = Path(__file__)
_PROJECT = _SETUP.parent


def read_version(filename):
    return re.search(
        r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", read(filename), re.MULTILINE
    ).group(1)


def read(filename):
    with open(_PROJECT / filename) as infile:
        text = infile.read()
    return text


if os.environ.get("READTHEDOCS") == "True":
    install_requires = ["dataclass_builder", "dataclasses;python_version=='3.6'"]
else:
    install_requires = [
        "appdirs",
        "astropy",
        "cached_property",
        "cf_units>=2.1.1",
        "dataclasses;python_version=='3.6'",
        "dataclass-builder>=1.1.3",
        "fortran-format-converter>=0.1.3",
        "numpy>=1.16.0",
        "regex",
        "scipy",
        "wrapt",
        "yzal",
    ]

docs_require = ["sphinx>=1.7", "sphinxcontrib-apidoc"]
tests_require = [
    "flake8>=3.7.7",
    "mypy",
    "pydocstyle",
    "pytest",
    "pytest-cov",
    "pytest-mock",
    "pyflakes>=2.1.0",
    "typing-extensions",
]

setup(
    name="rads",
    version=read_version("rads/__version__.py"),
    author="Michael R. Shannon",
    author_email="mrshannon.aerospace@gmail.com",
    description="Python front end for the Radar Altimeter Database System.",
    long_description=read("README.rst"),
    long_description_content_type="text/x-rst",
    license="MIT",
    url="https://github.com/ccarocean/pyrads",
    packages=find_packages(),
    package_data={
        "rads": ["py.typed"],
        "rads.config": ["py.typed"],
        "rads.xml": ["py.typed"],
    },
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require={
        "lxml": ["lxml"],  # use libxml2 to read configuration files
        "docs": docs_require,
        "tests": tests_require,
        "dev": docs_require + tests_require + ["black", "isort", "twine"],
    },
    tests_require=tests_require,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Database :: Front-Ends"
        "Topic :: Scientific/Engineering"
        "Topic :: Scientific/Engineering :: GIS"
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    zip_safe=False,
)
