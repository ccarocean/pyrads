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


Usage
-----

*PyRADS is currently in development, and will remain so until its first v1.0.0
release.  Until then, only functions and classes exported at the top (rads
module) level are considered somewhat stable.*


Loading RADS configuration files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, PyRADS is only capable of loading the RADS XML files.  Loading data
is planned for a future release.

The easiest way to load the RADS v4 XML files is to ensure that the
:code:`RADSDATAROOT` environment variable is set with the path to the RADS
data root as documented in the official `RADS User Manual`_.  Once this is
done, PyRADS can load all the same XML files that the official RADS
implementation does by default (it will also load PyRADS specific XML files
as well).

.. code-block:: python

    >>> import rads
    >>> rads_config = rads.load_config()

The files that will be loaded are (on Unix):

1. :code:`<dataroot>/conf/rads.xml`
2. :code:`/etc/pyrads/settings.xml`
3. :code:`~/.rads/rads.xml`
4. :code:`~/.config/pyrads/settings.xml`
5. :code:`rads.xml`
6. :code:`pyrads.xml`

where each file can override or append to the settings of the previous files.

*If not using Unix, use the rads.config_files() function to retrieve
your platform specific list.*

To add XML files to this:

.. code-block:: python

    >>> import rads
    >>> xml_files = rads.config_files() + ['path/to/custom/file.xml']
    >>> rads_config = rads.load_config(xml_files=xml_files)

The RADS data root can also be overridden:

.. code-block:: python

    >>> import rads
    >>> rads_config = rads.load_config(dataroot='/path/to/custom/dataroot')

For more information on loading of RADS v4 XML configuration files consult the
documentation.


.. _Radar Altimeter Database System: https://github.com/remkos/rads
.. _RADS User Manual: https://github.com/remkos/rads/blob/master/doc/manuals/rads4_user_manual.pdf
.. |build-status| image:: https://travis-ci.com/ccarocean/pyrads.svg?branch=master&style=flat
   :target: https://travis-ci.com/ccarocean/pyrads
   :alt: Build status
.. |coverage-status| image:: http://codecov.io/github/ccarocean/pyrads/coverage.svg?branch=master
   :target: http://codecov.io/github/ccarocean/pyrads?branch=master
   :alt: Test coverage
