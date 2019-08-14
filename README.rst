.. image:: https://raw.githubusercontent.com/ccarocean/pyrads/master/docs/_static/logo_black.png
    :alt: PyRADS
    :align: center

Python access to the Radar Altimeter Database System
====================================================

|build-status|
|doc-status|
|coverage-status|
|code-style|

WORK IN PROGRESS - This will be removed upon first release.

Python library for access to the `Radar Altimeter Database System`_ (RADS).

Documentation
-------------

Documentation for PyRADS can be found at `https://pyrads.readthedocs.io/en/latest/ <https://pyrads.readthedocs.io/en/latest/>`_ or in `PDF <https://readthedocs.org/projects/pyrads/downloads/pdf/latest/>`_ and `Epub <https://readthedocs.org/projects/pyrads/downloads/epub/latest/>`_ formats.


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


Development
-----------

invoke_
^^^^^^^

PyRADS uses invoke_ to make common development tasks easier.  For example the simplest way to get started working on PyRADS is to fork and clone the repository and then from within the main project directory:

.. code-block::

    pip install invoke && invoke develop

This will install all development requirements with :code:`pip` and thus it is recommended to do this from a :code:`virtualenv`.

If you are working on a system where libxml2_ is installed you may wish to also install lxml_ to provide faster XML parsing, but more importantly better error messages.  With lxml_, configuration parsing errors will be identified by line number.

To get the full list tasks that can be run by invoke_:

.. code-block::

    invoke -l

For example, to run the formatters (isort_ and black_), static checkers, and
all tests (with coverage report):

.. code-block::

    invoke format check test --coverage

*NOTE: This should be ran before making any commits.*

If on a non UNIX environment some of the tasks may fail.  If this happens you can use the :code:`--dry` flag to print out the commands that would be ran and then adjust accordingly.


tox_
^^^^

While the above invoke_ tasks are relatively quick and are good for development they are insufficient to ensure PyRADS is working properly across all options (lxml_ or not) and all supported Python versions.  For this a tox_ configuration is provided.  To run the full test suite simply run:

.. code-block::

    tox

Or if you have a recent version of :code:`tox` you can speed up the process with:

.. code-block::

    tox --parallel auto

The :code:`doc-pdf` environment will fail if XeTeX_, xindy_, and latexmk_.  This is usually fine.

If all tests run by tox succeed (except for :code:`doc-pdf`) the TravisCI build should succeed as well.


.. _Radar Altimeter Database System: https://github.com/remkos/rads
.. _RADS User Manual: https://github.com/remkos/rads/blob/master/doc/manuals/rads4_user_manual.pdf
.. _libxml2: http://www.xmlsoft.org/
.. _lxml: https://lxml.de/
.. _invoke: http://www.pyinvoke.org/
.. _isort: https://github.com/timothycrosley/isort
.. _black: https://black.readthedocs.io/en/stable/
.. _tox: https://tox.readthedocs.io/en/latest/
.. _XeTeX: http://xetex.sourceforge.net/
.. _xindy: http://xindy.sourceforge.net/
.. _latexmk: https://mg.readthedocs.io/latexmk.html
.. |build-status| image:: https://travis-ci.com/ccarocean/pyrads.svg?branch=master&style=flat
   :target: https://travis-ci.com/ccarocean/pyrads
   :alt: Build status
.. |doc-status| image:: https://readthedocs.org/projects/pyrads/badge/?version=latest
   :target: https://pyrads.readthedocs.io/en/latest/
   :alt: Documentation status
.. |coverage-status| image:: https://codecov.io/github/ccarocean/pyrads/coverage.svg?branch=master
   :target: https://codecov.io/github/ccarocean/pyrads?branch=master
   :alt: Test coverage
.. |code-style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style is black
