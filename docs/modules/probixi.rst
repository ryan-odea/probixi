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

Multi-GPU
=========

.. autofunction:: run_data_parallel

Utilities
=========

.. autofunction:: probixi.io.frame_id
