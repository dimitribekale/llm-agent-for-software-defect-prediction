import time
import requests
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from ..config import LLMConfig


class LLMException(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMTimeoutException(LLMException):
    """Raised when LLM request times out."""
    pass


class LLMConnectionException(LLMException):
    """Raised when connection to LLM fails."""
    pass


class LLMResponse:
    """Structured response from LLM."""

    def __init__(
        self,
        content: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.model = model
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return f"LLMResponse(model={self.model}, length={len(self.content)})"


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        pass


class OllamaClient(BaseLLMClient):
    """Ollama LLM client with retry logic and error handling."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.session = requests.Session()
        self._last_request_time = 0
        self._min_request_interval = 0.1  # Rate limiting

    def generate(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response with exponential backoff retry logic.

        Args:
            prompt: Input prompt for the LLM
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (doubles each retry)
            **kwargs: Override config parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse object

        Raises:
            LLMTimeoutException: If request times out
            LLMConnectionException: If connection fails after retries
            LLMException: For other LLM-related errors
        """
        # Rate limiting
        time_since_last = time.time() - self._last_request_time
        if time_since_last < self._min_request_interval:
            time.sleep(self._min_request_interval - time_since_last)

        # Build payload
        payload = {
            "model": kwargs.get("model", self.config.model_name),
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": kwargs.get("stream", self.config.stream)
        }

        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    self.config.api_url,
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                self._last_request_time = time.time()

                data = response.json()
                return LLMResponse(
                    content=data.get("response", ""),
                    model=payload["model"],
                    metadata={
                        "total_duration": data.get("total_duration"),
                        "load_duration": data.get("load_duration"),
                        "prompt_eval_count": data.get("prompt_eval_count"),
                        "eval_count": data.get("eval_count"),
                    }
                )

            except requests.exceptions.Timeout as e:
                last_exception = LLMTimeoutException(
                    f"Request timed out after {self.config.timeout}s"
                )

            except requests.exceptions.ConnectionError as e:
                last_exception = LLMConnectionException(
                    f"Failed to connect to {self.config.api_url}: {str(e)}"
                )

            except requests.exceptions.HTTPError as e:
                last_exception = LLMException(
                    f"HTTP error {e.response.status_code}: {str(e)}"
                )

            except Exception as e:
                last_exception = LLMException(f"Unexpected error: {str(e)}")

            # Exponential backoff
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)

        # All retries exhausted
        raise last_exception

    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            # Try to get model list or version
            response = self.session.get(
                self.config.api_url.replace("/api/generate", "/api/tags"),
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = self.session.get(
                self.config.api_url.replace("/api/generate", "/api/tags"),
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            raise LLMException(f"Failed to list models: {str(e)}")


class LLMClientFactory:
    """Factory for creating LLM clients."""

    _clients = {
        "ollama": OllamaClient,
    }

    @classmethod
    def create(cls, client_type: str, config: LLMConfig) -> BaseLLMClient:
        """
        Create LLM client of specified type.

        Args:
            client_type: Type of client (e.g., "ollama")
            config: LLM configuration

        Returns:
            LLM client instance
        """
        client_class = cls._clients.get(client_type.lower())
        if not client_class:
            raise ValueError(
                f"Unknown client type: {client_type}. "
                f"Available: {list(cls._clients.keys())}"
            )
        return client_class(config)

    @classmethod
    def register_client(cls, name: str, client_class: type):
        """Register a new client type."""
        cls._clients[name.lower()] = client_class
