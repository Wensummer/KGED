from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import requests


@dataclass
class BaseLLMClient:
    provider: str
    model: str
    timeout: int

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: Optional[int] = None) -> Dict:
        raise NotImplementedError

    def _new_session(self) -> requests.Session:
        session = requests.Session()
        session.trust_env = False
        return session


@dataclass
class OllamaClient(BaseLLMClient):
    url: str

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: Optional[int] = None) -> Dict:
        options = {"temperature": temperature}
        if max_tokens is not None and max_tokens > 0:
            options["num_predict"] = int(max_tokens)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        session = self._new_session()
        resp = session.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        raw = resp.json()
        return {
            "text": str(raw.get("response", "")),
            "raw": raw,
        }


@dataclass
class OpenAICompatibleClient(BaseLLMClient):
    api_base: str
    api_key: str = ""

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: Optional[int] = None) -> Dict:
        endpoint = self.api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = int(max_tokens)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        session = self._new_session()
        resp = session.post(endpoint, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        raw = resp.json()
        text = ""
        choices = raw.get("choices") or []
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message") or {}
                if isinstance(message, dict):
                    text = str(message.get("content", "") or "")
                if not text:
                    text = str(first.get("text", "") or "")
        return {
            "text": text,
            "raw": raw,
        }


def build_llm_client(
    provider: str,
    model: str,
    ollama_url: str,
    llm_api_base: str,
    llm_api_key: str,
    llm_timeout: int,
) -> BaseLLMClient:
    provider = (provider or "ollama").strip().lower()
    if provider == "ollama":
        return OllamaClient(
            provider="ollama",
            model=model,
            timeout=llm_timeout,
            url=ollama_url,
        )
    if provider == "openai_compatible":
        api_base = llm_api_base.strip() if llm_api_base else "http://localhost:8000/v1"
        return OpenAICompatibleClient(
            provider="openai_compatible",
            model=model,
            timeout=llm_timeout,
            api_base=api_base,
            api_key=llm_api_key or "",
        )
    raise ValueError(f"Unsupported llm provider: {provider}")
