"""Connection reuse for concurrent provider calls; request limits live elsewhere."""

from __future__ import annotations

import os

import requests
from requests.adapters import HTTPAdapter


def create_provider_session() -> requests.Session:
    raw = os.getenv("STORYMEM_HTTP_POOL_MAXSIZE", "128")
    try:
        size = int(raw)
    except ValueError:
        raise ValueError("STORYMEM_HTTP_POOL_MAXSIZE must be a positive integer") from None
    if size < 1:
        raise ValueError("STORYMEM_HTTP_POOL_MAXSIZE must be a positive integer")
    session = requests.Session()
    for scheme in ("http://", "https://"):
        session.mount(scheme, HTTPAdapter(pool_connections=16, pool_maxsize=size,
                                         pool_block=False, max_retries=0))
    return session
