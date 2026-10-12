from __future__ import annotations

import contextlib
import math
import threading
import time
from urllib.parse import quote, urlsplit, urlunsplit


TEXT_MODELS = ('Gemini-3.5-Flash-Third', 'Gemini 3.5 Flash', 'ge-3.5-fl-zl')
GEMINI_IMAGE_MODEL = 'ge-3.1-fl-im-zl'
SEEDREAM_LITE_MODEL = 'Doubao-Seedream-5.0-lite'
ROUTING_VERSION = 'reel.capacity-aware-provider-pool.v2'


def gateway_url(configured: str, path: str) -> str:
    base = urlsplit(configured)
    if base.scheme not in {'http', 'https'} or not base.netloc or base.username or base.password:
        raise ValueError('provider URL must have an HTTP(S) origin without embedded credentials')
    return urlunsplit((base.scheme, base.netloc, path, '', ''))


def native_text_url(configured: str, model: str) -> str:
    if model not in TEXT_MODELS:
        raise ValueError(f'unapproved text model: {model}')
    return gateway_url(configured, f'/v1/models/{quote(model, safe="")}:generateContent') if model == 'ge-3.5-fl-zl' else configured


def text_capacities(capacity: int, native_capacity: int) -> dict:
    if type(native_capacity) is not int or native_capacity < 1:
        raise ValueError('native text capacity must be a positive integer')
    native = min(native_capacity, max(1, capacity // len(TEXT_MODELS)))
    legacy = math.ceil(max(1, capacity - native) / (len(TEXT_MODELS) - 1))
    return {model: native if model == 'ge-3.5-fl-zl' else legacy for model in TEXT_MODELS}


class RoutedRateLimiter:
    def __init__(self, global_limiter, model_limiter):
        self.global_limiter = global_limiter
        self.model_limiter = model_limiter

    @property
    def cooldown_seconds(self) -> float:
        return self.model_limiter.cooldown_seconds

    def acquire(self) -> float:
        return self.model_limiter.acquire() + self.global_limiter.acquire()

    def extend_cooldown(self, seconds: float | None = None) -> float:
        return self.model_limiter.extend_cooldown(seconds)


class CapacityPool:
    def __init__(self, names, capacity: int, *, route_capacities: dict | None = None):
        self.names = tuple(names)
        if not self.names or len(set(self.names)) != len(self.names) or any(not name for name in self.names):
            raise ValueError('pool routes must be nonempty and unique')
        if type(capacity) is not int or capacity < 1:
            raise ValueError('pool capacity must be a positive integer')
        self.capacity = capacity
        self.route_capacity = math.ceil(capacity / len(self.names))
        self.limits = dict(route_capacities) if route_capacities is not None else dict.fromkeys(self.names, self.route_capacity)
        if set(self.limits) != set(self.names) or any(type(limit) is not int or limit < 1 for limit in self.limits.values()):
            raise ValueError('each route must have one positive integer capacity')
        if sum(self.limits.values()) < capacity:
            raise ValueError('route limits cannot leave the global pool permanently underfilled')
        self.condition = threading.Condition()
        self.cursor = 0
        self.active = 0
        self.peak_active = 0
        self.waiting = 0
        self.routes = {name: {'active': 0, 'peak_active': 0, 'completed': 0, 'failed': 0,
                              'successful_seconds': 0.0, 'successful_max_seconds': 0.0}
                       for name in self.names}

    @contextlib.contextmanager
    def slot(self, preferred: str | None = None, *, excluded=()):
        if preferred is not None and preferred not in self.routes:
            raise ValueError(f'unknown pool route: {preferred}')
        excluded = frozenset(excluded)
        if not excluded.issubset(self.routes) or not set(self.names) - excluded or preferred in excluded:
            raise ValueError('excluded pool routes must leave at least one eligible route')
        with self.condition:
            self.waiting += 1
            try:
                while True:
                    candidates = [(self.cursor + offset) % len(self.names) for offset in range(len(self.names))]
                    available = next((index for index in candidates
                                      if (preferred is None or self.names[index] == preferred)
                                      and self.names[index] not in excluded
                                      and self.routes[self.names[index]]['active'] < self.limits[self.names[index]]), None)
                    if self.active < self.capacity and available is not None:
                        name = self.names[available]
                        self.cursor = (available + 1) % len(self.names)
                        self.active += 1
                        self.peak_active = max(self.peak_active, self.active)
                        route = self.routes[name]
                        route['active'] += 1
                        route['peak_active'] = max(route['peak_active'], route['active'])
                        break
                    self.condition.wait()
            finally:
                self.waiting -= 1
        started = time.monotonic()
        succeeded = False
        try:
            yield name
            succeeded = True
        finally:
            duration = time.monotonic() - started
            with self.condition:
                route['active'] -= 1
                self.active -= 1
                route['completed' if succeeded else 'failed'] += 1
                if succeeded:
                    route['successful_seconds'] += duration
                    route['successful_max_seconds'] = max(route['successful_max_seconds'], duration)
                self.condition.notify_all()

    def snapshot(self) -> dict:
        with self.condition:
            routes = {name: {**values, 'capacity': self.limits[name],
                              'successful_mean_seconds': round(values['successful_seconds'] / values['completed'], 3)
                              if values['completed'] else None}
                      for name, values in self.routes.items()}
            return {'capacity': self.capacity, 'active': self.active, 'peak_active': self.peak_active,
                    'waiting': self.waiting, 'routes': routes}
