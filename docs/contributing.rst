Contributing
============

If Geedim is useful to you, please consider `making a donation <https://github.com/sponsors/leftfield-geospatial>`__.

Bug reports and feature requests are welcome, and can be made with the `GitHub issue tracker <https://github.com/leftfield-geospatial/geedim/issues>`__.

Development
-----------

Setup
~~~~~

To create up a development setup, start by forking the `repository <https://github.com/leftfield-geospatial/geedim>`__, then clone the fork with:

.. code-block::

    git clone https://github.com/<username>/geedim
    cd geedim

Dependencies required for running tests can be installed, and Geedim linked into your environment with:

.. code-block::

    pip install -e .[tests]

Pull requests
~~~~~~~~~~~~~

Make changes in a new branch and submit a `pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`__ from your fork when the changes are ready for review.  You can use a draft pull request to get early feedback on changes before they are complete.  Geedim uses the `GitHub Flow <https://docs.github.com/en/get-started/using-github/github-flow>`__ workflow.

Please include reStructuredText style docstrings and `pytest <https://docs.pytest.org>`__ unit tests with your code.  Geedim uses `Ruff <https://docs.astral.sh/ruff>`__ for linting and formatting, with settings in |pyproject.toml|_.

.. |pyproject.toml| replace:: ``pyproject.toml``
.. _pyproject.toml: https://github.com/leftfield-geospatial/geedim/blob/main/pyproject.toml
