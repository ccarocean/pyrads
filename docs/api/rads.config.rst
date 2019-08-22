rads.config
===========

These dataclasses make up the configuration tree returned by :func:`rads.load_config`.  They are documented here to aid in modification of the returned configuration or for scratch construction of a :class:`rads.config.Config` object.

.. autoclass:: rads.config.Config
    :members:
    :noindex:

.. autoclass:: rads.config.Satellite
    :members:
    :noindex:

.. autoclass:: rads.config.Phase
    :members:
    :noindex:

.. autoclass:: rads.config.Variable
    :members:
    :noindex:


Phase Nodes
-----------

.. autoclass:: rads.config.Cycles
    :members:
    :noindex:

.. autoclass:: rads.config.Repeat
    :members:
    :noindex:

.. autoclass:: rads.config.ReferencePass
    :members:
    :noindex:

.. autoclass:: rads.config.SubCycles
    :members:
    :noindex:


Variable Nodes
--------------

.. autoclass:: rads.config.Constant
    :members:
    :noindex:

.. autoclass:: rads.rpn.CompleteExpression
    :members:
    :noindex:

.. autoclass:: rads.config.SingleBitFlag
    :members:
    :noindex:

.. autoclass:: rads.config.MultiBitFlag
    :members:
    :noindex:

.. autoclass:: rads.config.SurfaceType
    :members:
    :noindex:

.. autoclass:: rads.config.Grid
    :members:
    :noindex:

.. autoclass:: rads.config.NetCDFAttribute
    :members:
    :noindex:

.. autoclass:: rads.config.NetCDFVariable
    :members:
    :noindex:

.. autoclass:: rads.config.Compress
    :members:
    :noindex:

.. autoclass:: rads.config.Range
    :members:
    :noindex:
