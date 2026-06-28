import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm_quant_bench.client import ModelConfig, OpenAIChatClient, _is_loopback_url


class OpenAIMockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        stream_options = body.get("stream_options") or {}
        if "stream_options" in body and stream_options.get("include_usage") is not True:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"unsupported stream_options")
            return

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for token in ["hello", " ", "world"]:
                event = {"choices": [{"delta": {"content": token}}]}
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
            if stream_options.get("include_usage"):
                event = {
                    "choices": [],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            return

        payload = {
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, format, *args):
        return


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), OpenAIMockHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address
        cls.base_url = f"http://{host}:{port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)
        cls.server.server_close()

    def test_non_streaming_generation(self):
        client = OpenAIChatClient(
            ModelConfig(name="mock", base_url=self.base_url, model="mock-model")
        )
        result = client.generate("hi", stream=False)
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.output_tokens, 2)
        self.assertIsNone(result.ttft_s)

    def test_streaming_generation_without_default_stream_options(self):
        client = OpenAIChatClient(
            ModelConfig(name="mock", base_url=self.base_url, model="mock-model")
        )
        result = client.generate("hi", stream=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "hello world")
        self.assertGreater(result.output_tokens, 0)
        self.assertIsNotNone(result.ttft_s)
        self.assertIsNone(result.prompt_tokens)

    def test_streaming_generation_can_capture_usage(self):
        client = OpenAIChatClient(
            ModelConfig(
                name="mock",
                base_url=self.base_url,
                model="mock-model",
                stream_options={"include_usage": True},
            )
        )
        result = client.generate("hi", stream=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.prompt_tokens, 3)
        self.assertEqual(result.output_tokens, 2)

    def test_loopback_detection_for_proxy_bypass(self):
        self.assertTrue(_is_loopback_url("http://127.0.0.1:8000/v1"))
        self.assertTrue(_is_loopback_url("http://localhost:8000/v1"))
        self.assertTrue(_is_loopback_url("http://[::1]:8000/v1"))
        self.assertFalse(_is_loopback_url("https://api.openai.example/v1"))


if __name__ == "__main__":
    unittest.main()
