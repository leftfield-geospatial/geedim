Command line
============

Paths and URIs
^^^^^^^^^^^^^^

Command line and API file / directory parameters can be specified as local paths or remote URIs.  Geedim uses `fsspec <https://github.com/fsspec/filesystem_spec>`__ for file IO, which provides built-in support for `a number of remote file systems <https://filesystem-spec.readthedocs.io/en/stable/api.html#implementations>`__.  Support for other remote systems is available by installing `the relevant extension package <https://filesystem-spec.readthedocs.io/en/latest/api.html#other-known-implementations>`__.  See the `fsspec documentation <https://filesystem-spec.readthedocs.io/en/stable/features.html#configuration>`__ if your file system requires credentials or other configuration.
