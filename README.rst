PyRADS
======

WORK IN PROGRESS - This will be removed upon first release.

Python library for access to the `Radar Altimeter Database System`_ (RADS).

|build-status|
|coverage-status|

Requirements
------------

* Python 3.6 or greater
* NumPy + SciPy
* UDUNITS

*NOTE: A Fortran compiler is not required as PyRADS is not a wrapper around the
official Fortran library but a complete re-write in Python.*


.. _Radar Altimeter Database System: https://github.com/remkos/rads
.. |build-status| image:: https://travis-ci.com/ccarocean/pyrads.svg?branch=master&style=flat
   :target: https://travis-ci.com/ccarocean/pyrads
   :alt: Build status
.. |coverage-status| image:: http://codecov.io/github/ccarocean/pyrads/coverage.svg?branch=master
   :target: http://codecov.io/github/ccarocean/pyrads?branch=master
   :alt: Test coverage
