"""Pytest bootstrap: put the repo root on ``sys.path``.

Its mere presence lets the tests ``import embed_worker``, ``import serve``, and
``import vgi_embed`` without an installed-package path.
"""
