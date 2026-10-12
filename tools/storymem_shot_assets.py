from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import os
import random
import re
import signal
import sqlite3
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from storymem_teacher.providers import GenericImageProvider, ImageGenerationRequest, ProviderHTTPError, ProviderResult, RemoteTeacherClient, SharedRequestRateLimiter, endpoint_sha256
from storymem_provider_pool import CapacityPool, GEMINI_IMAGE_MODEL, ROUTING_VERSION, RoutedRateLimiter, SEEDREAM_LITE_MODEL, TEXT_MODELS, gateway_url, native_text_url, text_capacities
from storymem_teacher.media import extract_video_frames
from storymem_teacher.util import atomic_write_json, sha256_file, sha256_json


VERSION = 'storymem.shot-driven-assets.v1'
ALIASES = TEXT_MODELS
PE_MAX_OUTPUT_TOKENS = 32768
STOP = threading.Event()
INVENTORY_SYSTEM = '''Analyze a natural multi-shot story for a memory-management dataset. Return strict JSON only.
Do not rewrite the supplied shot prompts. Do not impose a fixed narrative arc or a six-asset kit.
Return {"entities":[{"id":"entity_1","name":"...","description":"stable visual facts only",
"asset_type":"character_reference|scene_plate|object_closeup|style_reference",
"category":"person|animal|robot|creature|object|scene|style","first_shot":0}],
"shots":[{"shot_index":0,"entity_ids":["entity_1"],"analysis":"specific current visual needs, mutable state, relations and reference purposes"}]}.
Include every shot exactly once, zero-indexed. Entities may be people, animals, objects, locations or style.
Use only facts grounded in the supplied story. Separate stable identity from changing clothes, weather,
pose, ownership and object state. Stable descriptions must not disclose later events or later states.
Do not turn internal memory operations into things that appear in the image. No fixed vocabulary of people or scenes.
At most 40 entities. first_shot is the first index where that entity appears in shots[].entity_ids.
Every shot analysis must be specific to that shot's prompt. Do not categorize animals or robots as human people.'''
SHOT_SYSTEM = '''You design simulated user-uploaded images for exactly one video shot, NOT the video itself.
Return strict JSON {"analysis":"shot-specific needs and deliberate gaps", "assets":[{
"purpose":"identity|scene|object|state|interaction|opening|style",
"entity_ids":["entity_1"], "reference_ids":["observed_ref_id"],
"user_description":"short truthful user description, no evaluation labels or instructions",
"prompt":"complete unique natural visual description for Seedream, normally 100-220 English words",
"aspect_ratio":"portrait|landscape|square", "rationale":"why this image suits this shot"}]}.
Produce exactly requested_images images, or fewer only if they would be artificial or incompatible; explain the gap.
Analyze the actual current raw prompt. Write a DIFFERENT content-rich prompt for each image; do not fill generic
character/scene/object templates. Images provide selected evidence, not an oracle solution to the entire shot.
No images when requested_images is zero. Preserve explicit user intent. Do not add costumes or state changes for variety.
Use actual attached reference images for recurring identity, object design or scene identity whenever available.
Copy only provided entity and reference IDs. If an entity already appeared in generated history but has no reliable
reference, return an empty assets array and explain missing_identity_evidence; never independently invent its new appearance.
Reference IDs are metadata; in the actual Seedream prompt call them reference image 1, 2, etc. in listed order.
Single static image, never a storyboard or action montage. Distinguish identity reference from exact opening pose.
Vary useful framing, visible details, lighting and composition, but honor the story's actual style and state.
weak_or_redundant means still usable limited or redundant information, NOT severe defects. partial means deliberately
incomplete coverage. historical_state means preserve identity but depict an explicitly established prior state and
describe it truthfully as an older reference; never label another person's face with a trusted protagonist ID.
Do not expose private analysis, correct tool choices, asset quality labels or future events in user_description.
If prior images conflict, prefer the established canonical identity and explain uncertainty rather than inventing facts.'''
CHECK_SYSTEM = '''Independently inspect the final attached candidate image against the supplied current request,
image purpose, truthful user description, and preceding reference images. Return strict JSON:
{"pass":true|false,"identity_preserved":true|false,"requirement_preserved":true|false,
"usable_for_purpose":true|false,"observations":"visible evidence, no assumptions",
"repair_prompt":"complete corrected Seedream prompt if failed, else null"}.
The final image is the candidate, earlier images are references in order. Do not trust the generation prompt's
claim that an identity or object was preserved: inspect the pixels. For partial/weak references, assess their
stated limited purpose, not whether they solve every video requirement. An older-state reference can be valid
only when its description explicitly says so. Reject unexpected identity drift, wrong task, major defects,
unintended montage, and misleading user descriptions. Do not penalize harmless stylistic details.'''


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def stable_number(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:12], 16)


def write_once(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'immutable artifact conflict: {path}')
        return
    atomic_write_json(path, value)


class AssetQualityRejected(ValueError):
    pass


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = self.root / 'progress.sqlite3'

    @contextlib.contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS stories (
                    story_id TEXT PRIMARY KEY, source_path TEXT NOT NULL, split TEXT NOT NULL,
                    num_shots INTEGER NOT NULL, asset_status TEXT NOT NULL DEFAULT 'queued',
                    trajectory_status TEXT NOT NULL DEFAULT 'queued', updated REAL NOT NULL,
                    error TEXT, failures INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS shots (
                    story_id TEXT NOT NULL, shot_index INTEGER NOT NULL, raw_prompt TEXT NOT NULL,
                    requested_images INTEGER NOT NULL, mode TEXT NOT NULL,
                    asset_status TEXT NOT NULL DEFAULT 'queued', image_count INTEGER NOT NULL DEFAULT 0,
                    trajectory_status TEXT NOT NULL DEFAULT 'queued', updated REAL NOT NULL,
                    error TEXT, failures INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(story_id, shot_index));
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY, story_id TEXT NOT NULL, shot_index INTEGER NOT NULL,
                    status TEXT NOT NULL, path TEXT, prompt TEXT, purpose TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL, error TEXT);
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,
                    kind TEXT NOT NULL, story_id TEXT, shot_index INTEGER, detail TEXT);
                CREATE INDEX IF NOT EXISTS shots_status ON shots(asset_status, trajectory_status);
                CREATE INDEX IF NOT EXISTS assets_updated ON assets(updated);
                CREATE INDEX IF NOT EXISTS stories_status ON stories(asset_status, trajectory_status, updated);
            ''')

    def execute(self, sql: str, parameters=()) -> None:
        with self.connection() as connection:
            connection.execute(sql, parameters)

    def query(self, sql: str, parameters=()) -> list[dict]:
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(sql, parameters)]

    def event(self, kind: str, story_id: str = '', shot_index=None, detail: str = '') -> None:
        self.execute('INSERT INTO events(created,kind,story_id,shot_index,detail) VALUES(?,?,?,?,?)',
                     (time.time(), kind, story_id, shot_index, detail[:2000]))
        print(json.dumps({'time': time.time(), 'event': kind, 'story': story_id,
                          'shot': shot_index, 'detail': detail[:250]}, ensure_ascii=False), flush=True)

    def shot(self, story_id: str, shot_index: int, status: str, *, images=0, error=None) -> None:
        self.execute('UPDATE shots SET asset_status=?,image_count=?,error=?,updated=? WHERE story_id=? AND shot_index=?',
                     (status, images, error, time.time(), story_id, shot_index))


def select_sources(manifest_path: Path, count: int, seed: int) -> list[dict]:
    manifest = read_json(manifest_path)
    rows = sorted(manifest['records'], key=lambda row: stable_number(f'{seed}:{row["story_id"]}'))
    quotas = {'train': count - 2 * (count // 20), 'validation': count // 20, 'test': count // 20}
    selected = []
    family_counts = Counter()
    split_counts = Counter()
    for row in rows:
        metadata = row.get('metadata') or {}
        family = str(metadata.get('source_file') or row['story_id'])
        bucket = stable_number(f'split:{seed}:{family}') % 100
        split = 'train' if bucket < 90 else 'validation' if bucket < 95 else 'test'
        if family_counts[family] >= 2 or split_counts[split] >= quotas[split]:
            continue
        source = Path(row['path'])
        if not source.is_absolute():
            source = manifest_path.parents[2] / source
        if not source.is_file():
            continue
        value = read_json(source)
        prompts = value.get('prompts') or []
        if len(prompts) < 2 or any(not isinstance(prompt, str) or len(prompt.strip()) < 100 for prompt in prompts):
            continue
        selected.append({'story_id': row['story_id'], 'path': str(source.resolve()),
                         'sha256': sha256_file(source), 'split': split, 'family': family,
                         'num_shots': len(prompts)})
        family_counts[family] += 1
        split_counts[split] += 1
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} eligible stories for split quotas {quotas}')
    return selected


def initialize(root: Path, source_manifest: Path, count: int, seed: int) -> None:
    store = Store(root)
    store.initialize()
    manifest_path = root / 'source_manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['requested_stories'] != count or manifest['seed'] != seed:
            raise ValueError('run already initialized with different source selection')
    else:
        records = select_sources(source_manifest.resolve(), count, seed)
        manifest = {'schema_version': VERSION, 'run_id': uuid.uuid4().hex, 'requested_stories': count, 'seed': seed,
                    'source_manifest': str(source_manifest.resolve()), 'source_manifest_sha256': sha256_file(source_manifest),
                    'expected_shots': sum(row['num_shots'] for row in records), 'records': records}
        write_once(manifest_path, manifest)
    story_rows = []
    shot_rows = []
    for row in manifest['records']:
        story_id = row['story_id']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', story_id) or story_id in {'.', '..'}:
            raise ValueError('source story_id is not safe for local paths')
        original = read_json(Path(row['path']))
        seed_path = root / 'stories' / story_id / 'seed.json'
        write_once(seed_path, original)
        story_rows.append((story_id, str(seed_path), row['split'], len(original['prompts']), time.time()))
        pure = stable_number(f'no_upload:{seed}:{story_id}') % 100 < 10
        for shot_index, prompt in enumerate(original['prompts']):
            rng = random.Random(f'{seed}:{story_id}:{shot_index}')
            count_images = 0 if pure else rng.choices([0, 1, 2, 3], weights=[40, 30, 15, 5])[0]
            mode = rng.choices(['useful', 'partial', 'weak_or_redundant', 'historical_state'], [60, 25, 10, 5])[0]
            if shot_index == 0 and mode == 'historical_state':
                mode = 'partial'
            shot_rows.append((story_id, shot_index, prompt, count_images, mode, time.time()))
    with store.connection() as connection:
        connection.executemany('INSERT OR IGNORE INTO stories(story_id,source_path,split,num_shots,updated) VALUES(?,?,?,?,?)', story_rows)
        connection.executemany('INSERT OR IGNORE INTO shots(story_id,shot_index,raw_prompt,requested_images,mode,updated) VALUES(?,?,?,?,?,?)', shot_rows)
    store.event('initialized', detail=f'{count} stories; {manifest["expected_shots"]} shots; no old images imported')


class Services:
    def __init__(self, root: Path, config_path: Path, *, concurrency=8, qps=2.5,
                 image_concurrency=16, offline_image_concurrency=12, native_text_concurrency=4, native_text_qps=1.0,
                 image_providers=('seedream', 'seedream5'), api_key_script: Path | None = None):
        from storymem_policy_collector import build_parser, _resolve_provider_configuration
        arguments = build_parser().parse_args(['--input-manifest', str(root / 'source_manifest.json'),
                                               '--output-root', str(root), '--api-config', str(config_path)])
        self.config = _resolve_provider_configuration(arguments)
        if api_key_script is not None:
            match = re.search(r'''(?m)^\s*(?:export\s+)?RECAPTION_API_KEY\s*=\s*(["'])(.*?)\1\s*$''', api_key_script.read_text())
            if not match or not match.group(2).strip() or '$' in match.group(2):
                raise ValueError('credential script must contain a literal nonempty RECAPTION_API_KEY; it is never executed')
            key = match.group(2).strip()
            self.config['teacher_key'] = os.getenv('STORYMEM_RL_GEMINI_API_KEY') or os.getenv(arguments.teacher_api_key_env) or os.getenv('API_KEY') or key
            self.config['pe_key'] = os.getenv(arguments.pe_api_key_env) or self.config['teacher_key']
            self.config['image_key'] = os.getenv('STORYMEM_RL_IMAGE_API_KEY') or os.getenv(arguments.image_api_key_env) or os.getenv('API_KEY') or key
        self.root = root
        if not 0 < offline_image_concurrency < image_concurrency or concurrency < 1:
            raise ValueError('reserve positive online image capacity and positive text capacity')
        self.text_pool = CapacityPool(ALIASES, concurrency, route_capacities=text_capacities(concurrency, native_text_concurrency))
        providers = tuple(image_providers)
        if not providers or len(set(providers)) != len(providers) or not set(providers).issubset({'seedream', 'seedream5', 'gemini'}):
            raise ValueError('select unique supported image providers: seedream, seedream5, gemini')
        available = {
            'seedream': {'model': self.config['image_model'], 'adapter': self.config['image_adapter'], 'url': self.config['image_url']},
            'seedream5': {'model': SEEDREAM_LITE_MODEL, 'adapter': 'seedream', 'url': self.config['image_url']},
            'gemini': {'model': GEMINI_IMAGE_MODEL, 'adapter': 'gemini',
                       'url': gateway_url(self.config['pe_url'], '/v1/images/gemini_flash/generations')},
        }
        routes = [available[name] for name in providers]
        if len({route['model'] for route in routes}) != len(routes):
            raise ValueError('image provider aliases must resolve to distinct models')
        self.image_routes = {route['model']: route for route in routes}
        self.image_pool = CapacityPool(self.image_routes, image_concurrency)
        self.offline_image_capacity = threading.BoundedSemaphore(offline_image_concurrency)
        self.online_image_capacity = threading.BoundedSemaphore(image_concurrency - offline_image_concurrency)
        self.limiter = SharedRequestRateLimiter(qps)
        self.model_limiters = {model: RoutedRateLimiter(self.limiter, SharedRequestRateLimiter(native_text_qps if model == 'ge-3.5-fl-zl' else qps))
                               for model in ALIASES}
        self.store = Store(root)
        self.dispatch_locks = [threading.Lock() for _ in range(128)]

    @contextlib.contextmanager
    def text_slot(self, *, excluded=()):
        with self.text_pool.slot(excluded=excluded) as model:
            yield model

    @staticmethod
    def image_profile(route: dict) -> dict:
        return {'adapter': route['adapter'], 'model': route['model'], 'endpoint_sha256': endpoint_sha256(route['url'])}

    def provider_profile(self) -> dict:
        return {**self.config['public'], 'routing_version': ROUTING_VERSION,
                'text_failover': {'retryable_transport_only': True, 'max_distinct_routes': len(ALIASES),
                                  'http_attempts_per_route': 1, 'http_timeout_seconds': 120},
                'text_pool': [{'model': model, 'teacher_endpoint_sha256': endpoint_sha256(native_text_url(self.config['teacher_url'], model)),
                               'pe_endpoint_sha256': endpoint_sha256(native_text_url(self.config['pe_url'], model)),
                               'api_style': 'gemini' if model == 'ge-3.5-fl-zl' else self.config['teacher_style'],
                               'cooldown_scope': 'model_only'} for model in ALIASES],
                'image_pool': [self.image_profile(route) for route in self.image_routes.values()]}

    def snapshot(self) -> dict:
        return {'text': self.text_pool.snapshot(), 'image': self.image_pool.snapshot()}

    def client(self, model: str) -> RemoteTeacherClient:
        return RemoteTeacherClient(native_text_url(self.config['teacher_url'], model), api_key=self.config['teacher_key'], model=model,
                                   api_style='gemini' if model == 'ge-3.5-fl-zl' else self.config['teacher_style'], timeout=120, max_attempts=1,
                                   max_output_tokens=12000, rate_limiter=self.model_limiters[model])

    def text_request(self, operation: str, invoke):
        attempted = set()
        failures = []
        audit_path = None
        for _ in ALIASES:
            model = None
            try:
                with self.text_slot(excluded=attempted) as model:
                    attempted.add(model)
                    result = invoke(model)
            except Exception as error:
                if model is None:
                    raise
                provenance = getattr(error, 'provenance', {})
                attempts = provenance.get('attempts') or []
                retryable = isinstance(error, ProviderHTTPError) and bool(attempts) and attempts[-1].get('retryable') is True
                retry = retryable and len(attempted) < len(ALIASES)
                failures.append({'model': model, 'error': self.safe_error(error),
                                 'request_payload_sha256': provenance.get('request_payload_sha256'),
                                 'attempts': [{key: attempt.get(key) for key in ('trace_id', 'http_status', 'retryable', 'latency_ms',
                                              'response_sha256', 'shared_cooldown_seconds')} for attempt in attempts]})
                if audit_path is None:
                    audit_path = self.root / 'text_failovers' / f'{uuid.uuid4().hex}.json'
                atomic_write_json(audit_path, {'operation': operation, 'status': 'retrying' if retry else 'failed',
                                              'failed_routes': failures})
                if not retry:
                    raise
            else:
                if audit_path is not None:
                    provenance = getattr(result, 'provenance', {})
                    atomic_write_json(audit_path, {'operation': operation, 'status': 'recovered', 'failed_routes': failures,
                                                  'selected_model': model, 'request_payload_sha256': provenance.get('request_payload_sha256'),
                                                  'response_sha256': provenance.get('response_sha256')})
                return result
        raise RuntimeError('text route attempts exhausted without a result')

    def call(self, **kwargs):
        return self.text_request('teacher', lambda model: self.client(model).call(**kwargs))

    def generate_text(self, **kwargs):
        from storymem_adapter.gemini import IsolatedGeminiTextClient
        def invoke(model):
            client = IsolatedGeminiTextClient(native_text_url(self.config['pe_url'], model), api_key=self.config['pe_key'],
                                             model=model, timeout=120, max_attempts=1,
                                             max_output_tokens=PE_MAX_OUTPUT_TOKENS, rate_limiter=self.model_limiters[model])
            return client.generate_text(**kwargs)
        return self.text_request('pe', invoke)

    def generate(self, request: ImageGenerationRequest):
        return self.image(request, offline=False)

    def safe_error(self, error: Exception) -> str:
        text = f'{type(error).__name__}: {error}'
        for key, value in self.config.items():
            if key.endswith('_key') and isinstance(value, str) and value:
                text = text.replace(value, '[redacted]')
        return text[:2000]

    def json(self, system: str, content: Any, audit_path: Path, *, temperature=0.4, max_tokens=10000) -> dict:
        if audit_path.exists():
            audit = read_json(audit_path)
            if audit.get('input_sha256') != sha256_json({'system': system, 'content': self.fingerprint(content)}):
                raise ValueError(f'cached text input changed: {audit_path}')
            return audit['value']
        result = self.text_request('json', lambda model: self.client(model).call_json(
            system, content, temperature=temperature, max_output_tokens=max_tokens))
        atomic_write_json(audit_path, {'input_sha256': sha256_json({'system': system, 'content': self.fingerprint(content)}),
                                       'value': result.value, 'provenance': result.provenance})
        return result.value

    @staticmethod
    def fingerprint(content):
        if isinstance(content, Path):
            return {'media_sha256': sha256_file(content)}
        if isinstance(content, list):
            return [Services.fingerprint(item) for item in content]
        if isinstance(content, dict):
            return {key: Services.fingerprint(value) for key, value in content.items()}
        return content

    def image(self, request: ImageGenerationRequest, *, offline=True):
        lane = self.offline_image_capacity if offline else self.online_image_capacity
        with self.dispatch_locks[stable_number(request.request_id) % len(self.dispatch_locks)], lane:
            directory = Path(request.output_dir)
            dispatched = directory / f'{request.request_id}.dispatch.json'
            result_path = directory / f'{request.request_id}.provider_result.json'
            identity = {'request_id': request.request_id, 'prompt': request.prompt,
                        'reference_sha256': [sha256_file(Path(path)) for path in request.reference_paths],
                        'width': request.width, 'height': request.height, 'seed': request.seed}
            if result_path.exists():
                cached = read_json(result_path)
                if cached['identity_sha256'] != sha256_json(identity):
                    raise ValueError('image request identity changed on resume')
                result = ProviderResult(**cached['result'])
                if sha256_file(Path(result.output_path)) != result.provenance['output_sha256']:
                    raise ValueError('cached generated image bytes changed')
                profile = cached.get('provider_profile')
                if profile is not None and (not dispatched.exists() or read_json(dispatched).get('provider_profile') != profile
                        or any(result.provenance.get(key) != value for key, value in profile.items())):
                    raise ValueError('cached image provider differs from its pinned dispatch')
                return result
            if dispatched.exists():
                raise RuntimeError(f'image dispatch outcome uncertain; refusing blind repeated POST: {request.request_id}')
            with self.image_pool.slot() as model:
                route = self.image_routes[model]
                profile = self.image_profile(route)
                write_once(dispatched, {'identity_sha256': sha256_json(identity), 'created_at': time.time(),
                                        'automatic_post_retries': 0, 'provider_profile': profile,
                                        'lane': 'offline' if offline else 'online', 'reference_conditioned': bool(request.reference_paths)})
                provider = GenericImageProvider(api_url=route['url'], api_key=self.config['image_key'],
                                                model=model, adapter=route['adapter'], timeout=360, max_attempts=1)
                try:
                    result = provider.generate(request)
                    if any(result.provenance.get(key) != value for key, value in profile.items()):
                        raise ValueError('image provider result differs from its pinned dispatch')
                except Exception as error:
                    write_once(directory / f'{request.request_id}.dispatch_error.json',
                               {'error': self.safe_error(error), 'provenance': getattr(error, 'provenance', {}),
                                'provider_profile': profile, 'outcome': 'unconfirmed_do_not_replay'})
                    raise
                write_once(result_path, {'identity_sha256': sha256_json(identity), 'provider_profile': profile, 'result': result.to_dict()})
                return result


def normalize_inventory_ids(value: dict) -> dict:
    entities = value.get('entities')
    shots = value.get('shots')
    if not isinstance(entities, list) or not isinstance(shots, list):
        raise ValueError('inventory must contain entity and shot arrays')
    identifiers = [entity.get('id') for entity in entities]
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError('inventory entity IDs must be nonempty and unique')
    mapping = {identifier: f'entity_{index:04d}' for index, identifier in enumerate(identifiers, start=1)}
    normalized_shots = []
    for shot in shots:
        if not isinstance(shot.get('entity_ids'), list) or any(identifier not in mapping for identifier in shot['entity_ids']):
            raise ValueError('inventory shot refers to an unknown entity ID')
        normalized_shots.append({**shot, 'entity_ids': [mapping[identifier] for identifier in shot['entity_ids']]})
    return {**value, 'entities': [{**entity, 'id': mapping[entity['id']]} for entity in entities],
            'shots': normalized_shots, 'planner_entity_id_map': mapping}


def validate_inventory(value: dict, shot_count: int) -> None:
    entities = value.get('entities')
    shots = value.get('shots')
    if not isinstance(entities, list) or not 1 <= len(entities) <= 40 or not isinstance(shots, list):
        raise ValueError('inventory must contain entities and shot analyses')
    identifiers = set()
    for entity in entities:
        identifier = entity.get('id')
        if not isinstance(identifier, str) or not re.fullmatch(r'entity_[a-z0-9_]+', identifier) or identifier in identifiers:
            raise ValueError('invalid/duplicate inventory entity ID')
        if not all(isinstance(entity.get(key), str) and entity[key].strip() for key in ('name', 'description', 'asset_type')):
            raise ValueError('missing entity description')
        if entity['asset_type'] not in {'character_reference', 'scene_plate', 'object_closeup', 'style_reference'}:
            raise ValueError('unsupported inventory asset type')
        if entity.get('category') not in {'person', 'animal', 'robot', 'creature', 'object', 'scene', 'style'}:
            raise ValueError('inventory needs explicit grounded subject taxonomy')
        if type(entity.get('first_shot')) is not int or not 0 <= entity['first_shot'] < shot_count:
            raise ValueError('invalid first_shot')
        identifiers.add(identifier)
    if len(shots) != shot_count or [shot.get('shot_index') for shot in shots] != list(range(shot_count)):
        raise ValueError('inventory must analyze every shot in order')
    for shot in shots:
        if not isinstance(shot.get('analysis'), str) or len(shot['analysis']) < 30:
            raise ValueError('shot analysis is missing')
        if not isinstance(shot.get('entity_ids'), list) or not set(shot['entity_ids']).issubset(identifiers):
            raise ValueError('unknown shot entity')
    for entity in entities:
        occurrences = [shot['shot_index'] for shot in shots if entity['id'] in shot['entity_ids']]
        if not occurrences or entity['first_shot'] != min(occurrences):
            raise ValueError('first_shot must match actual entity occurrences')


def ensure_inventory(store: Store, services: Services, story: dict) -> dict:
    directory = store.root / 'stories' / story['story_id']
    seed = read_json(Path(story['source_path']))
    path = directory / 'inventory.json'
    already_exists = path.exists()
    if path.exists():
        value = read_json(path)
        validate_inventory(value, len(seed['prompts']))
    else:
        store.event('inventory_started', story['story_id'])
        content = json.dumps({'story_id': story['story_id'], 'story_bible': seed.get('story_bible'), 'prompts': seed['prompts']}, ensure_ascii=False)
        for attempt in range(2):
            raw_value = services.json(INVENTORY_SYSTEM, content, directory / f'inventory_call_{attempt}.json', max_tokens=14000)
            try:
                value = normalize_inventory_ids(raw_value)
                validate_inventory(value, len(seed['prompts']))
                break
            except ValueError as error:
                if attempt:
                    raise
                content += '\nYour preceding output: ' + json.dumps(raw_value, ensure_ascii=False) + '\nValidation error: ' + str(error) + '\nReturn the complete corrected object.'
        write_once(path, value)
    library = {}
    for entity in value['entities']:
        library[entity['id']] = {'material_id': entity['id'], 'tag': entity['id'], 'tags': [entity['id']],
                                 'name': entity['name'], 'description': entity['description'], 'asset_type': entity['asset_type'],
                                 'category': entity['category'], 'first_shot': entity['first_shot']}
    source = {'story_id': story['story_id'], 'title': seed.get('title'), 'prompts': seed['prompts'],
              'material_library': library, 'uploaded_assets_by_shot': {},
              'streaming_asset_root': str(directory), 'streaming_asset_schema': VERSION,
              'source_seed_sha256': sha256_file(Path(story['source_path'])), 'split': story['split']}
    write_once(directory / 'source.json', source)
    if not already_exists:
        store.event('inventory_ready', story['story_id'], detail=f'{len(value["entities"])} entities; {len(value["shots"])} shot analyses')
    return value


def prior_references(store: Store, story_id: str, shot_index: int) -> list[dict]:
    references = []
    directory = store.root / 'stories' / story_id
    for previous_index in range(shot_index):
        pack_path = directory / 'shots' / f'{previous_index:04d}' / 'upload_pack.json'
        if not pack_path.exists():
            continue
        for asset in read_json(pack_path)['assets']:
            references.append({'id': asset['asset_id'], 'entity_ids': asset['entity_ids'], 'path': asset['image_path'],
                               'sha256': asset['sha256'], 'description': asset['user_description'], 'source_shot': previous_index})
    checkpoint_path = store.root / 'trajectories' / 'stories' / story_id / 'story_checkpoint.json'
    if checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        if checkpoint['next_shot_index'] <= shot_index:
            for evidence_id, evidence in {**checkpoint.get('memory_bank', {}), **checkpoint.get('previous_evidence', {})}.items():
                for representation in evidence.get('representations', {}).values():
                    if representation.get('modality') != 'image':
                        continue
                    references.append({'id': evidence_id, 'entity_ids': [subject.removeprefix('subject_') for subject in evidence.get('subject_ids', [])],
                                       'path': representation['path'], 'sha256': representation['sha256'],
                                       'description': evidence.get('description'), 'source_shot': checkpoint['next_shot_index'] - 1})
    return references


def observe_prefix_references(store: Store, services: Services, story: dict, inventory: dict,
                              shot_index: int, missing: set[str]) -> list[dict] | None:
    checkpoint_path = store.root / 'trajectories' / 'stories' / story['story_id'] / 'story_checkpoint.json'
    if not checkpoint_path.exists():
        return None
    checkpoint = read_json(checkpoint_path)
    if checkpoint['next_shot_index'] < shot_index:
        return None
    directory = store.root / 'stories' / story['story_id'] / 'shots' / f'{shot_index:04d}' / 'prefix_refs'
    context_path = directory / 'context.json'
    if context_path.exists():
        context = read_json(context_path)
    else:
        candidates = []
        covered = Counter()
        for previous_index in reversed(range(shot_index)):
            relevant = missing.intersection(inventory['shots'][previous_index]['entity_ids'])
            if not relevant or all(covered[identifier] >= 2 for identifier in relevant):
                continue
            row = checkpoint['shots'][str(previous_index)]
            video = Path(row['output_path'])
            if sha256_file(video) != row['output_sha256']:
                raise ValueError('accepted prefix video bytes changed')
            frames = extract_video_frames(video, directory / f'source_{previous_index:04d}', count=3)
            for ordinal, frame in enumerate(frames):
                candidates.append({'id': f'prefix_s{previous_index:04d}_f{ordinal}', 'path': str(frame),
                                   'sha256': sha256_file(frame), 'source_shot': previous_index,
                                   'source_video_sha256': row['output_sha256'],
                                   'source_prompt': inventory['shots'][previous_index]['analysis']})
            covered.update(relevant)
            if len(candidates) >= 12:
                break
        context = {'entities': [entity for entity in inventory['entities'] if entity['id'] in missing],
                   'candidates': candidates}
        write_once(context_path, context)
    if not context['candidates']:
        raise ValueError('no accepted prefix frames for a previously seen entity')
    system = ('Inspect only actual supplied frames from already accepted story history. Return strict JSON '
              '{"references":[{"id":"exact supplied frame id","entity_ids":["exact known entity id"],'
              '"description":"what is actually visible; separate stable identity from transient state"}]}. '
              'Associate only entities clearly visible and identifiable from their grounded story descriptions. '
              'Do not infer visibility from text. Omit unusable/occluded/uncertain frames and unknown entities. '
              'These are real historical frames, not new generated identity examples.')
    safe_context = {**context, 'candidates': [{key: value for key, value in row.items() if key != 'path'} for row in context['candidates']]}
    result = services.json(system, [json.dumps(safe_context, ensure_ascii=False),
                                   *[Path(row['path']) for row in context['candidates']]],
                           directory / 'observation.json', temperature=0, max_tokens=6000)
    by_id = {row['id']: row for row in context['candidates']}
    references = []
    for row in result.get('references', []):
        if row.get('id') not in by_id or not isinstance(row.get('entity_ids'), list) or not set(row['entity_ids']).issubset(missing):
            raise ValueError('prefix observer invented a frame/entity')
        if row['entity_ids'] and isinstance(row.get('description'), str):
            references.append({**by_id[row['id']], 'entity_ids': row['entity_ids'], 'description': row['description']})
    write_once(directory / 'references.json', {'references': references})
    store.event('prefix_identity_observed', story['story_id'], shot_index, f'{len(references)} grounded historical frames')
    return references


def validate_asset_plan(value: dict, inventory: dict, references: list[dict], requested: int) -> None:
    assets = value.get('assets')
    if not isinstance(value.get('analysis'), str) or not isinstance(assets, list) or len(assets) > requested:
        raise ValueError('invalid per-shot asset plan')
    entity_ids = {entity['id'] for entity in inventory['entities']}
    reference_ids = {reference['id'] for reference in references}
    for asset in assets:
        if asset.get('purpose') not in {'identity', 'scene', 'object', 'state', 'interaction', 'opening', 'style'}:
            raise ValueError('invalid asset purpose')
        if not isinstance(asset.get('entity_ids'), list) or not asset['entity_ids'] or not set(asset['entity_ids']).issubset(entity_ids):
            raise ValueError('asset invents entity IDs')
        if not isinstance(asset.get('reference_ids'), list) or not set(asset['reference_ids']).issubset(reference_ids):
            raise ValueError('asset invents reference IDs')
        if len(asset['reference_ids']) > 4 or len(set(asset['reference_ids'])) != len(asset['reference_ids']):
            raise ValueError('use at most four distinct actual image references per asset')
        for key in ('prompt', 'user_description', 'rationale'):
            if not isinstance(asset.get(key), str) or len(asset[key].strip()) < (120 if key == 'prompt' else 5):
                raise ValueError(f'missing asset {key}')
        if asset.get('aspect_ratio') not in {'portrait', 'landscape', 'square'}:
            raise ValueError('invalid image aspect ratio')
        for identifier in asset['entity_ids']:
            available = {reference['id'] for reference in references if identifier in reference['entity_ids']}
            if available and not available.intersection(asset['reference_ids']):
                raise ValueError(f'asset ignores available identity reference for {identifier}')


def create_asset(store: Store, services: Services, story: dict, shot: dict, ordinal: int, plan: dict, references: list[dict]) -> dict:
    story_id = story['story_id']
    shot_index = shot['shot_index']
    asset_id = f'{story_id}_s{shot_index:04d}_a{ordinal:02d}'
    directory = store.root / 'stories' / story_id / 'shots' / f'{shot_index:04d}' / 'assets' / asset_id
    receipt_path = directory / 'receipt.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if sha256_file(Path(receipt['image_path'])) != receipt['sha256'] or receipt['plan_sha256'] != sha256_json(plan):
            raise ValueError('asset receipt or image changed')
        return receipt
    by_id = {reference['id']: reference for reference in references}
    selected = [by_id[identifier] for identifier in plan['reference_ids']]
    for reference in selected:
        if sha256_file(Path(reference['path'])) != reference['sha256']:
            raise ValueError('parent reference bytes changed')
    paths = [Path(reference['path']) for reference in selected]
    width, height = {'portrait': (864, 1152), 'landscape': (1280, 736), 'square': (1024, 1024)}[plan['aspect_ratio']]
    store.execute('INSERT OR IGNORE INTO assets(asset_id,story_id,shot_index,status,prompt,purpose,updated) VALUES(?,?,?,?,?,?,?)',
                  (asset_id, story_id, shot_index, 'generating', plan['prompt'], plan['purpose'], time.time()))
    prompt = plan['prompt']
    run_id = read_json(store.root / 'source_manifest.json')['run_id']
    for attempt in range(2):
        if STOP.is_set():
            raise InterruptedError('stopping')
        attempt_directory = directory / f'attempt_{attempt + 1}'
        request = ImageGenerationRequest(request_id=f'upload_{run_id[:12]}_{asset_id}_try{attempt+1}', prompt=prompt, reference_paths=paths,
                                         seed=stable_number(asset_id) % 2147483647, width=width, height=height,
                                         output_dir=attempt_directory)
        write_once(attempt_directory / 'request.json', {'asset_id': asset_id, 'prompt': prompt, 'reference_inputs': selected,
                   'seed': request.seed, 'width': width, 'height': height, 'attempt': attempt + 1})
        result_path = attempt_directory / 'result.json'
        if result_path.exists():
            result = read_json(result_path)
        else:
            store.event('image_started', story_id, shot_index, f'{asset_id} attempt {attempt + 1}')
            generated = services.image(request)
            result = {'output_path': generated.output_path, 'provenance': generated.provenance}
            write_once(result_path, result)
        store.execute('UPDATE assets SET status=?,path=?,attempts=?,updated=? WHERE asset_id=?',
                      ('validating', result['output_path'], attempt + 1, time.time(), asset_id))
        check_input = [json.dumps({'raw_prompt': shot['raw_prompt'], 'purpose': plan['purpose'], 'mode': shot['mode'],
                       'user_description': plan['user_description'], 'generation_prompt': prompt}, ensure_ascii=False),
                       *paths, Path(result['output_path'])]
        check = services.json(CHECK_SYSTEM, check_input, attempt_directory / 'assessment.json', temperature=0, max_tokens=4000)
        if any(type(check.get(key)) is not bool for key in ('pass', 'identity_preserved', 'requirement_preserved', 'usable_for_purpose')):
            raise ValueError('image review must contain four explicit boolean verdicts')
        passed = all(check.get(key) is True for key in ('pass', 'identity_preserved', 'requirement_preserved', 'usable_for_purpose'))
        if passed:
            receipt = {**plan, 'asset_id': asset_id, 'image_path': result['output_path'], 'sha256': sha256_file(Path(result['output_path'])),
                       'plan_sha256': sha256_json(plan), 'references': selected, 'attempts': attempt + 1,
                       'assessment': check, 'source_kind': 'new_gemini_simulated_upload' if result['provenance'].get('adapter') == 'gemini' else 'new_seedream_simulated_upload',
                       'image_provider': {key: result['provenance'].get(key) for key in ('model', 'adapter', 'endpoint_sha256')}, 'created_at': time.time()}
            write_once(receipt_path, receipt)
            store.execute('UPDATE assets SET status=?,error=NULL,updated=? WHERE asset_id=?', ('ready', time.time(), asset_id))
            store.event('image_ready', story_id, shot_index, asset_id)
            return receipt
        prompt = check.get('repair_prompt')
        if not isinstance(prompt, str) or len(prompt) < 120:
            break
    store.execute('UPDATE assets SET status=?,error=?,updated=? WHERE asset_id=?',
                  ('quality_rejected', json.dumps(check, ensure_ascii=False)[:2000], time.time(), asset_id))
    raise AssetQualityRejected(f'image failed independent review: {asset_id}')


def produce_story_step(store: Store, services: Services, story: dict) -> str:
    story_id = story['story_id']
    inventory = ensure_inventory(store, services, story)
    rows = store.query('SELECT * FROM shots WHERE story_id=? AND asset_status!=? ORDER BY shot_index LIMIT 1', (story_id, 'ready'))
    if not rows:
        store.execute('UPDATE stories SET asset_status=?,updated=? WHERE story_id=?', ('complete', time.time(), story_id))
        return 'complete'
    shot = rows[0]
    if shot['asset_status'] == 'failed':
        return 'failed'
    shot_index = shot['shot_index']
    directory = store.root / 'stories' / story_id / 'shots' / f'{shot_index:04d}'
    pack_path = directory / 'upload_pack.json'
    if pack_path.exists():
        pack = read_json(pack_path)
        if (pack['raw_prompt_sha256'] != sha256_json(shot['raw_prompt'])
                or pack.get('pack_sha256') != sha256_json({key: value for key, value in pack.items() if key != 'pack_sha256'})):
            raise ValueError('published pack does not match shot')
        for asset in pack['assets']:
            if sha256_file(Path(asset['image_path'])) != asset['sha256']:
                raise ValueError('published asset bytes changed')
        store.shot(story_id, shot_index, 'ready', images=len(pack['assets']))
        return 'ready'
    snapshot_path = directory / 'planning_input.json'
    snapshot = read_json(snapshot_path) if snapshot_path.exists() else None
    references = snapshot['references'] if snapshot else prior_references(store, story_id, shot_index)
    relevant = set(inventory['shots'][shot_index]['entity_ids'])
    references = [reference for reference in references if relevant.intersection(reference['entity_ids'])]
    canonical = {}
    for reference in references:
        for identifier in reference['entity_ids']:
            canonical.setdefault(identifier, reference)
    recent = list(reversed(references))[:3]
    references = list({reference['id']: reference for reference in [*canonical.values(), *recent]}.values())[:8]
    prior_entity_ids = {entity['id'] for entity in inventory['entities'] if entity['first_shot'] < shot_index}
    missing = relevant.intersection(prior_entity_ids) - {identifier for reference in references for identifier in reference['entity_ids']}
    if not snapshot and shot['requested_images'] and missing:
        observed = observe_prefix_references(store, services, story, inventory, shot_index, missing)
        if observed is None:
            store.shot(story_id, shot_index, 'waiting_prefix', error='Need real prior identity evidence: ' + ','.join(sorted(missing)))
            return 'waiting_prefix'
        references = [*references, *observed]
        missing -= {identifier for reference in observed for identifier in reference['entity_ids']}
    store.shot(story_id, shot_index, 'planning')
    context = {'story_id': story_id, 'shot_index': shot_index, 'raw_prompt': shot['raw_prompt'],
               'analysis': inventory['shots'][shot_index]['analysis'],
               'entities': [entity for entity in inventory['entities'] if entity['id'] in relevant],
               'requested_images': shot['requested_images'], 'coverage_mode': shot['mode'],
               'unobservable_previous_entity_ids': sorted(missing),
               'references': [{key: value for key, value in reference.items() if key != 'path'} for reference in references]}
    if snapshot:
        context = snapshot['context']
        references = snapshot['references']
        missing = set(context.get('unobservable_previous_entity_ids') or [])
    else:
        write_once(snapshot_path, {'context': context, 'references': references})
    for reference in references:
        if sha256_file(Path(reference['path'])) != reference['sha256']:
            raise ValueError('frozen planning reference bytes changed')
    content = [json.dumps(context, ensure_ascii=False), *[Path(reference['path']) for reference in references]]
    for attempt in range(2):
        plan = services.json(SHOT_SYSTEM, content, directory / f'plan_call_{attempt}.json', temperature=0.65, max_tokens=8000)
        try:
            validate_asset_plan(plan, inventory, references, shot['requested_images'])
            if any(not set(asset['entity_ids']).issubset(relevant) or missing.intersection(asset['entity_ids']) for asset in plan['assets']):
                raise ValueError('asset depicts a future, irrelevant or unobservable previous entity')
            break
        except ValueError as error:
            if attempt:
                raise
            content = [*content, 'Your previous output: ' + json.dumps(plan, ensure_ascii=False) + '\nValidation error: ' + str(error) + '. Return the full corrected object.']
    if shot['requested_images'] and not plan['assets'] and 'missing_identity_evidence' in plan['analysis']:
        raise ValueError('image planner lacks usable identity evidence')
    write_once(directory / 'plan.json', {**context, 'decision': plan})
    store.shot(story_id, shot_index, 'generating')
    assets = [create_asset(store, services, story, shot, ordinal, asset, references)
              for ordinal, asset in enumerate(plan['assets'], start=1)]
    pack = {'schema_version': VERSION, 'story_id': story_id, 'shot_index': shot_index,
            'raw_prompt_sha256': sha256_json(shot['raw_prompt']), 'plan_sha256': sha256_json(plan),
            'requested_images': shot['requested_images'], 'assets': assets, 'created_at': time.time()}
    pack['pack_sha256'] = sha256_json(pack)
    write_once(pack_path, pack)
    store.shot(story_id, shot_index, 'ready', images=len(assets))
    store.event('shot_assets_ready', story_id, shot_index, f'{len(assets)} new images; requested {shot["requested_images"]}')
    return 'ready'


def run_producer(root: Path, services: Services, *, workers: int, active_stories: int, max_failures: int) -> None:
    store = Store(root)
    active = set()
    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while not STOP.is_set():
            completed, _ = concurrent.futures.wait(futures, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED) if futures else ([], [])
            for future in completed:
                story = futures.pop(future)
                active.remove(story['story_id'])
                try:
                    result = future.result()
                    store.execute('UPDATE stories SET asset_status=?,error=NULL,updated=? WHERE story_id=?',
                                  ('complete' if result == 'complete' else 'active', time.time(), story['story_id']))
                except InterruptedError:
                    STOP.set()
                except Exception as error:
                    detail = f'{type(error).__name__}: {error}'[:2000]
                    store.execute('UPDATE stories SET asset_status=?,error=?,failures=failures+1,updated=? WHERE story_id=?',
                                  ('failed', detail, time.time(), story['story_id']))
                    store.execute("UPDATE shots SET asset_status='failed',error=?,failures=failures+1,updated=? WHERE story_id=? AND asset_status IN ('planning','generating')",
                                  (detail, time.time(), story['story_id']))
                    store.event('asset_story_failed', story['story_id'], detail=detail)
            failed = store.query("SELECT count(*) AS count FROM stories WHERE asset_status='failed'")[0]['count']
            if failed >= max_failures:
                store.event('production_fuse', detail=f'{failed} failed stories; inspect before resuming')
                STOP.set()
                break
            remaining = store.query("SELECT * FROM stories WHERE asset_status NOT IN ('complete','failed') ORDER BY CASE asset_status WHEN 'active' THEN 0 ELSE 1 END,updated,story_id LIMIT ?", (active_stories,))
            if not remaining and not futures:
                store.event('asset_production_complete')
                break
            submitted = 0
            for story in remaining:
                if len(futures) >= workers or story['story_id'] in active:
                    continue
                pending = store.query("SELECT asset_status,updated FROM shots WHERE story_id=? AND asset_status!='ready' ORDER BY shot_index LIMIT 1", (story['story_id'],))
                if pending and pending[0]['asset_status'] == 'waiting_prefix' and time.time() - pending[0]['updated'] < 30:
                    continue
                active.add(story['story_id'])
                futures[executor.submit(produce_story_step, store, services, story)] = story
                submitted += 1
            if not futures and not submitted:
                STOP.wait(5)


def main() -> None:
    parser = argparse.ArgumentParser(description='Per-shot, reference-grounded Seedream asset production; no old image reuse.')
    parser.add_argument('command', choices=('init', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'data/source/manifest.json')
    parser.add_argument('--stories', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--api-config', type=Path, default=Path(__file__).resolve().parents[1] / 'Director_Agent/.config.local.json')
    parser.add_argument('--api-key-script', type=Path, default=None)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--active-stories', type=int, default=64)
    parser.add_argument('--max-failures', type=int, default=10)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.producer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'init':
            initialize(root, args.source_manifest, args.stories, args.seed)
            return
        if not (root / 'source_manifest.json').is_file():
            raise ValueError('initialize the immutable source manifest first')
        signal.signal(signal.SIGTERM, lambda signum, frame: STOP.set())
        signal.signal(signal.SIGINT, lambda signum, frame: STOP.set())
        atomic_write_json(root / 'producer_process.json', {'pid': os.getpid(), 'started_at': time.time(),
                          'version': VERSION, 'workers': args.workers, 'active_stories': args.active_stories})
        services = Services(root, args.api_config, api_key_script=args.api_key_script)
        run_producer(root, services, workers=args.workers, active_stories=args.active_stories, max_failures=args.max_failures)


if __name__ == '__main__':
    main()
