import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from prk_remaster.comfy_client import ComfyClient, ComfyError
from prk_remaster.models import ModelError, license_accepted, require_license, accept_license
from prk_remaster.qwen import build_workflow
import pytest


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/system_stats"):
            self._json(200, {"devices": []})
            return
        if self.path.startswith("/history/"):
            self._json(200, {"pid": {"status": {"status_str": "success"}, "outputs": {"13": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}}}})
            return
        if self.path.startswith("/view"):
            buffer = io.BytesIO()
            Image.new("RGB", (8, 8), (1, 2, 3)).save(buffer, format="PNG")
            data = buffer.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json(404, {})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path.startswith("/upload"):
            self._json(200, {"name": "uploaded.png"})
            return
        if self.path.startswith("/prompt"):
            self.server.last_prompt = raw  # type: ignore[attr-defined]
            self._json(200, {"prompt_id": "pid"})
            return
        if self.path.startswith("/free"):
            self._json(200, {})
            return
        self._json(404, {})


def test_client_refuses_public_url():
    with pytest.raises(ComfyError):
        ComfyClient("http://example.com:8188")


def test_mock_queue_round_trip():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        client = ComfyClient(f"http://127.0.0.1:{port}", timeout=5)
        assert client.healthy()
        image = Image.new("RGB", (8, 8), (4, 5, 6))
        name = client.upload_image(image, "src.png")
        workflow = build_workflow(
            image_name=name, prompt="stone", seed=7,
            diffusion="qwen_image_2.1_turbo_int8_convrot.safetensors",
            text_encoder="qwen3vl_8b_w4a8.safetensors",
            vae="qwen_image_2.1_vae_bf16.safetensors",
            sigmas="1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0",
        )
        assert workflow["10"]["inputs"]["cfg"] == 1.0
        assert workflow["8"]["inputs"]["sampler_name"] == "euler"
        assert "0.414568" in workflow["7"]["inputs"]["sigmas"]
        assert workflow["5"]["inputs"]["negative_prompt"] == ""
        prompt_id = client.submit(workflow)
        entry = client.wait(prompt_id, poll=0.01)
        result = client.download(client.result_images(entry)[0])
        assert result.size == (8, 8)
        client.free(settle=False)
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_license_gate(tmp_path):
    path = tmp_path / "license-acceptance.json"
    with pytest.raises(ModelError):
        require_license(path)
    accept_license(path)
    assert license_accepted(path)
    require_license(path)
