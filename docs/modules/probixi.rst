=============
API reference
=============

.. currentmodule:: probixi


Pipeline
========

.. autoclass:: Probixi
   :members:
   :member-order: bysource

Streams and results
===================

.. autoclass:: probixi.indexer.IndexStream
   :members:
   :undoc-members:
   :member-order: bysource

.. autoclass:: probixi.indexer.IndexResult
   :members:

Output writers
==============

.. autoclass:: DuckDBOffloader
   :members:

.. autoclass:: DataOffloader
   :members:

.. autoclass:: PeakOffloader
   :members:

Indexing ambiguity
==================

.. autoclass:: Ambigator
   :members:

.. autoclass:: AmbigatorResult
   :members:

.. autofunction:: probixi.ambigator.ambiguity_operator

.. autofunction:: probixi.ambigator.point_group_ops

.. autofunction:: probixi.ambigator.parse_operator

Multi-GPU
=========

.. autofunction:: run_data_parallel

Utilities
=========

.. autofunction:: probixi.io.frame_id
