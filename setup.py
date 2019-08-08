import os
import re
import sys
from pathlib import Path
from subprocess import CalledProcessError, run

from setuptools import Command, find_packages, setup

_SETUP = Path(__file__)
_PROJECT = _SETUP.parent
_DOCS = _PROJECT / "docs"
_PACKAGE = _PROJECT / "rads"
_TESTS = _PROJECT / "tests"


def read_version(filename):
    return re.search(
        r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", read(filename), re.MULTILINE
    ).group(1)


def read(filename):
    with open(_PROJECT / filename) as infile:
        text = infile.read()
    return text


class Quality(Command):

    description = "run code quality checkers"
    user_options = [("format", "f", "run isort+black before QA")]

    def initialize_options(self):
        self.format = False

    def finalize_options(self):
        self.format = bool(self.format)

    def run(self):
        try:
            if self.format:
                run(["isort", "-rc", _PROJECT], check=True)
                run(["black", _PROJECT], check=True)
            run(["mypy", _PACKAGE], check=True)
            run(["mypy", "--config-file", _TESTS / "mypy.ini", _TESTS], check=True)
            run(["flake8", _SETUP, _PACKAGE, _TESTS], check=True)
            run(["pydocstyle", _PACKAGE], check=True)
        except CalledProcessError as err:
            print("\n💀 Quality analysis failed 💀")
            sys.exit(err.returncode)
        print("\n✨ Quality analysis complete ✨")


class Test(Command):

    description = "run tests"
    user_options = [("coverage", "c", "generate test coverage report")]

    def initialize_options(self):
        self.coverage = False
        pass

    def finalize_options(self):
        self.coverage = bool(self.coverage)

    def run(self):
        try:
            if self.coverage:
                run(["pytest", "--cov", _PACKAGE, "--cov-branch"], check=True)
                run(["coverage", "html"], check=True)
            else:
                run(["pytest"], check=True)
        except CalledProcessError as err:
            print("\n💀 Testing failed 💀")
            sys.exit(err.returncode)
        print("\n✨ Testing complete ✨")


class Dist(Command):

    description = "build source and wheel distributions"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            run(["python", _SETUP, "sdist"], check=True)
            run(["python", _SETUP, "bdist_wheel"], check=True)
            run(["twine", "check", _PROJECT / "dist/*"], check=True)
        except CalledProcessError as err:
            print("\n💀 Build failed 💀")
            sys.exit(err.returncode)
        print("\n✨ Build complete ✨")


class Doc(Command):

    description = "build documentation"
    user_options = [("pdf", None, "Build PDF instead of HTML documentation.")]

    def initialize_options(self):
        self.pdf = False

    def finalize_options(self):
        self.pdf = bool(self.pdf)

    def run(self):
        try:
            if self.pdf:
                run(
                    ["sphinx-build", "-M", "latexpdf", _DOCS, _DOCS / "_build"],
                    check=True,
                )
            else:
                run(
                    ["sphinx-build", "-b", "html", _DOCS, _DOCS / "_build" / "html"],
                    check=True,
                )
        except CalledProcessError as err:
            print("\n💀 Documentation build failed 💀")
            sys.exit(err.returncode)
        print("\n✨ Documentation build complete ✨")


class Cleanup(Command):

    description = "cleanup temporary files"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    @staticmethod
    def remove(path):
        if path.is_file():
            print(f"removing {path}")
            path.unlink()
        if path.is_dir():
            for file in path.iterdir():
                Cleanup.remove(file)
            path.rmdir()
            print(f"removing {path}")

    @staticmethod  # noqa: C901
    def remove_matching(matcher, files=False, dirs=False):
        for file in _PROJECT.iterdir():
            if files and file.is_file() and matcher(file):
                Cleanup.remove(file)
            if dirs and file.is_dir() and matcher(file):
                Cleanup.remove(file)

        for dir in ["rads", "tests"]:
            for root, dirs_, files_ in os.walk(_PROJECT / dir):
                if files:
                    for file in [Path(root) / f for f in files_]:
                        if matcher(file):
                            Cleanup.remove(file)
                if dirs:
                    for dir in [Path(root) / f for f in dirs_]:
                        if matcher(dir):
                            Cleanup.remove(dir)

    def run(self):
        try:
            for file in (_DOCS / "api" / "apidoc").iterdir():
                if file.is_file() and file.suffix == ".rst":
                    self.remove(file)
            self.remove(_DOCS / "_build")
            self.remove(_PROJECT / "dist")
            self.remove(_PROJECT / "build")
            self.remove_matching(lambda f: f.suffix == ".pyc", files=True)
            self.remove_matching(lambda f: f.name == ".coverage", files=True)
            self.remove_matching(lambda f: f.name == ".pytest_cache", dirs=True)
            self.remove_matching(lambda f: f.name == ".mypy_cache", dirs=True)
            self.remove_matching(lambda f: f.name == "htmlcov", dirs=True)
            self.remove_matching(lambda f: f.suffix == ".egg-info", dirs=True)
        except Exception:
            print("\n💀 Documentation build failed 💀")
            sys.exit(1)
        print("\n✨ Cleanup complete ✨")


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
    cmdclass={
        "quality": Quality,
        "test": Test,
        "dist": Dist,
        "doc": Doc,
        "cleanup": Cleanup,
    },
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
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require={
        "lxml": ["lxml"],  # use libxml2 to read configuration files
        "docs": docs_require,
        "tests": tests_require,
        "dev": docs_require + tests_require + ["black", "isort", "twine"],
    },
    tests_require=["pytest", "pytest-mock"],
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
