"""Local ComfyUI queue client. The server is expected on 127.0.0.1 only."""

from __future__ import annotations

import time
import uuid
from io import BytesIO
from urllib.parse import urlparse

import httpx
from PIL import Image


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 1800.0):
        parsed = urlparse(base_url)
        host = parsed.hostname or ""
        if host not in {"127.0.0.1", "localhost"}:
            raise ComfyError(f"refusing non-local ComfyUI URL {base_url}")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(60.0, read=self.timeout))

    def healthy(self) -> bool:
        try:
            with self._client() as client:
                response = client.get("/system_stats")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def system_stats(self) -> dict:
        with self._client() as client:
            response = client.get("/system_stats")
            response.raise_for_status()
            return response.json()

    def object_info(self) -> dict:
        with self._client() as client:
            response = client.get("/object_info")
            response.raise_for_status()
            return response.json()

    def missing_nodes(self, names: list[str]) -> list[str]:
        info = self.object_info()
        return [name for name in names if name not in info]

    def upload_image(self, image: Image.Image, name: str) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        buffer.seek(0)
        with self._client() as client:
            response = client.post(
                "/upload/image",
                files={"image": (name, buffer.getvalue(), "image/png")},
                data={"overwrite": "true"},
            )
            response.raise_for_status()
            payload = response.json()
        uploaded = payload.get("name")
        if not uploaded:
            raise ComfyError(f"upload did not return a name: {payload}")
        return uploaded

    def submit(self, workflow: dict) -> str:
        body = {"prompt": workflow, "client_id": self.client_id}
        with self._client() as client:
            response = client.post("/prompt", json=body)
            if response.status_code >= 400:
                raise ComfyError(f"prompt rejected: {response.status_code} {response.text[:500]}")
            payload = response.json()
        if payload.get("node_errors"):
            raise ComfyError(f"node errors: {payload['node_errors']}")
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"no prompt_id in {payload}")
        return prompt_id

    def wait(self, prompt_id: str, poll: float = 1.0) -> dict:
        deadline = time.monotonic() + self.timeout
        with self._client() as client:
            while time.monotonic() < deadline:
                response = client.get(f"/history/{prompt_id}")
                response.raise_for_status()
                history = response.json()
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        raise ComfyError(f"generation failed: {status.get('messages')}")
                    return entry
                time.sleep(poll)
        raise ComfyError(f"timed out waiting for {prompt_id}")

    def download(self, image_info: dict) -> Image.Image:
        params = {
            "filename": image_info["filename"],
            "subfolder": image_info.get("subfolder") or "",
            "type": image_info.get("type") or "output",
        }
        with self._client() as client:
            response = client.get("/view", params=params)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image.load()
            return image.copy()

    def result_images(self, entry: dict) -> list[dict]:
        images = []
        for node in (entry.get("outputs") or {}).values():
            images.extend(node.get("images") or [])
        if not images:
            raise ComfyError("prompt finished without images")
        return images

    def free(self) -> None:
        with self._client() as client:
            response = client.post("/free", json={"unload_models": True, "free_memory": True})
            if response.status_code >= 400:
                raise ComfyError(f"free failed: {response.status_code}")

    def interrupt(self) -> None:
        with self._client() as client:
            client.post("/interrupt")
