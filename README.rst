PyRADS |build-status| |coverage-status|
=======================================

WORK IN PROGRESS - This will be removed upon first release.

Python library for access to the `Radar Altimeter Database System`_ (RADS).

Requirements
------------

* Python 3.5 or greater
* NumPy + SciPy

.. note::
    A Fortran compiler is not required as PyRADS is not a wrapper around the
    official Fortran library but a complete re-write into Python.


.. _Radar Altimeter Database System: https://github.com/remkos/rads
.. |build-status| image:: https://travis-ci.org/mrshannon/pyrads.svg?branch=master&style=flat
   :target: https://travis-ci.org/mrshannon/pyrads
   :alt: Build status
.. |coverage-status| image:: http://codecov.io/github/mrshannon/pyrads/coverage.svg?branch=master
   :target: http://codecov.io/github/mrshannon/pyrads?branch=master
   :alt: Test coverage