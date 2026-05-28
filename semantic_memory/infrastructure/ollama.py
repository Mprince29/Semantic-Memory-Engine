from typing import Any


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "The 'requests' package is required for live Ollama inference."
            ) from exc

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
