from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests

from .common import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward one REEL shot request to a shared-filesystem model service.")
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    endpoint = os.environ.get("REEL_MODEL_ENDPOINT")
    if not endpoint:
        raise ValueError("set REEL_MODEL_ENDPOINT to your model's synchronous shot endpoint")
    request = read_json(args.request)
    headers = {"Idempotency-Key": request["request_id"]}
    token = os.environ.get("REEL_MODEL_API_KEY")
    if token:
        headers["Authorization"] = "Bearer " + token
    response = requests.post(endpoint, json=request, headers=headers, timeout=float(os.environ.get("REEL_MODEL_TIMEOUT", "1700")))
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict) or value.get("request_id") != request["request_id"]:
        raise ValueError("model service must echo request_id and return the adapter response schema")
    write_json(args.response, value)


if __name__ == "__main__":
    main()
