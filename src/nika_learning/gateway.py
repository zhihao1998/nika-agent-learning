"""Read gateway credentials locally and verify a pinned deployment."""

import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx


def token(path):
    path = Path(path).expanduser()
    stat = path.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise ValueError("Gateway token must belong to this user and require mode 600")
    return path.read_text().strip()


def inference_headers(config):
    gateway = config["gateway"]
    url = urlsplit(config["base_url"])
    if url.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Gateway access requires a loopback endpoint")
    if not config["model"].startswith(gateway["cluster"] + "/"):
        raise ValueError("Gateway alias does not match the selected cluster")
    headers = {
        "Authorization": "Bearer " + token(gateway["token_file"]),
        "X-Gateway-Session": gateway["session"],
    }
    if gateway["cluster"] == "orix":
        if gateway.get("external_data_approved") is not True:
            raise ValueError("ORIX requires explicit external data approval")
        headers.update(
            {"X-Compute-Project": "nika", "X-External-Data-Approved": "true"}
        )
    return headers


def serving_identity(config):
    gateway = config["gateway"]
    inference_headers(config)
    if config.get("serving_exclusive"):
        raise ValueError(
            "Gateway deployment state does not establish exclusive serving"
        )
    with httpx.Client(timeout=20, trust_env=False) as client:
        response = client.get(
            config["base_url"].removesuffix("/v1")
            + "/api/v1/deployments/"
            + gateway["deployment_id"],
            headers={
                "Authorization": "Bearer " + token(gateway["operator_token_file"])
            },
        )
        response.raise_for_status()
        deployment = response.json()
    spec = deployment["spec"]
    if (
        deployment["id"] != gateway["deployment_id"]
        or deployment["state"] != "READY"
        or spec["cluster"] != gateway["cluster"]
        or spec["revision"] != gateway["revision"]
        or deployment["profile"] != gateway["profile"]
        or str(deployment["job_id"])
        not in {str(job) for job in config["serving_job_ids"]}
    ):
        raise RuntimeError(
            "Pinned gateway deployment is not ready or its identity changed"
        )
    return {
        key: deployment[key] for key in ("id", "state", "job_id", "profile", "spec")
    }
