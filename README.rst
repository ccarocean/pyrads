PyRADS
======

|build-status|
|coverage-status|
|code-style|

WORK IN PROGRESS - This will be removed upon first release.

Python library for access to the `Radar Altimeter Database System`_ (RADS).


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

The simplest way to get started working on PyRADS is:

.. code-block:: bash

    git pull git@github.com:ccarocean/pyrads.git
    cd pyrads
    python3 -m venv --prompt PyRADS .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -e ".[dev]"

*Naturally, you should fork the repository first.*

If you are working on a system where libxml2_ is installed you may wish to replace the last command with:

.. code-block:: bash

    pip install -e ".[lxml,dev]"

This will provide for faster XML parsing and more importantly better error messages.

setup.py commands
^^^^^^^^^^^^^^^^^

PyRADS uses custom :code:`setup.py` commands to ease development.

To run all quality checks simply use:

.. code-block:: bash

    python setup.py quality

To run isort_ and black_ before the quality checks (recommended) use

.. code-block:: bash

    python setup.py quality --format

To run all tests:

.. code-block:: bash

    python setup.py test

or with coverage reports:

.. code-block:: bash

    python setup.py test --coverage

To build source and wheel distributions (and check them):

.. code-block::

    python setup.py dist

To build the HTML documentation:

.. code-block::

    python setup.py doc

or the PDF documentation (requires XeTeX_, xindy_, and latexmk_):

.. code-block::

    python setup.py doc --pdf

Finally, to cleanup temporary files:

.. code-block::

    python setup.py cleanup


tox
^^^

While the above :code:`setup.py` commands are relatively quick and are good for development they are insufficient to ensure PyRADS is working properly across all options (lxml or not) and all supported Python versions.  For this a tox configuration is provided.  To run the full test suite simply run:

.. code-block::

    tox

Or if you have a recent version of :code:`tox` you can speed up the process with:

.. code-block::

    tox --parallel auto

The :code:`doc-pdf` environment will fail if XeTeX_, xindy_, and latexmk_.  This is usually fine.

If all tests run by tox succeed the TravisCI build should succeed as well.


.. _Radar Altimeter Database System: https://github.com/remkos/rads
.. _RADS User Manual: https://github.com/remkos/rads/blob/master/doc/manuals/rads4_user_manual.pdf
.. _libxml2: http://www.xmlsoft.org/
.. _isort: https://github.com/timothycrosley/isort
.. _black: https://black.readthedocs.io/en/stable/
.. _XeTeX: http://xetex.sourceforge.net/
.. _xindy: http://xindy.sourceforge.net/
.. _latexmk: https://mg.readthedocs.io/latexmk.html
.. |build-status| image:: https://travis-ci.com/ccarocean/pyrads.svg?branch=master&style=flat
   :target: https://travis-ci.com/ccarocean/pyrads
   :alt: Build status
.. |coverage-status| image:: https://codecov.io/github/ccarocean/pyrads/coverage.svg?branch=master
   :target: https://codecov.io/github/ccarocean/pyrads?branch=master
   :alt: Test coverage
.. |code-style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style is black
