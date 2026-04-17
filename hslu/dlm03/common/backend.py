"""This file provides abstractions for interacting with various LLM backends."""

from typing import Any, Self

import openai

from hslu.dlm03.common import chat as chat_lib
from hslu.dlm03.common import config as config_lib
from hslu.dlm03.common import types
from hslu.dlm03.util import ratelimit


class LLM:
    """A class that provides an interface to an LLM backend.

    It uses an `openai.Client` to interact with the LLM and enforces a
    rate limit on the number of calls per minute.
    """
    _client: openai.Client
    _model: str
    _ratelimiter: ratelimit.RateLimiter

    def __init__(self, *, client: openai.Client, model: str, ratelimiter: ratelimit.RateLimiter) -> None:
        """Initializes an `LLM` instance.

        Args:
            client: The OpenAI client.
            model: The model name.
            ratelimiter: The rate limiter.
        """
        self._client = client
        self._model = model
        self._ratelimiter = ratelimiter

    def __call__(self, *, response_format: type | None = None, **kwargs: Any) -> types.ModelResponse:
        """Calls the LLM backend and returns its output.

        Args:
            response_format: The type of the expected response from the model.
            **kwargs: Additional keyword arguments to pass to the API call.

        Returns:
            A `types.ModelResponse` object representing the model's response.
        """
        with self._ratelimiter:
            if response_format is not None:
                kwargs["response_format"] = response_format
                fn = self._client.chat.completions.parse
            else:
                fn = self._client.chat.completions.create
            return fn(model=self._model, **kwargs)

    def generate(self, chat: chat_lib.Chat, /, **kwargs: Any) -> types.ModelResponse:
        """Generates a response from the LLM based on the provided chat history.

        Args:
            chat: A `chat_lib.Chat` object containing the conversation history.
            **kwargs: Additional keyword arguments to pass to the `__call__` method.

        Returns:
            A `types.ModelResponse` object representing the model's response.

        """
        return self(messages=chat.messages, **kwargs)

    @classmethod
    def from_config(cls, config: config_lib.Config) -> Self:
        """Instantiates an `LLM` instance from a configuration object.

        Args:
            config: The configuration object to use.

        Returns:
            An `LLM` instance.
        """
        client, _ = config.get_client()
        ratelimiter = ratelimit.RateLimiter(config.rpm)
        return cls(client=client, model=config.model_name, ratelimiter=ratelimiter)


class AsyncLLM:
    """A class that provides an asynchronous interface to an LLM backend.

    It uses an `openai.AsyncClient` to interact with the LLM and enforces a
    rate limit on the number of calls per minute.
    """
    _client: openai.AsyncClient
    """The underlying OpenAI async client."""
    _model: str
    """The model name."""
    _ratelimiter: ratelimit.RateLimiter
    """The rate limiter."""

    def __init__(self, *, client: openai.AsyncClient, model: str, ratelimiter: ratelimit.RateLimiter) -> None:
        """Initializes an `AsyncLLM` instance.

        Args:
            client: The OpenAI async client.
            model: The model name.
            ratelimiter: The rate limiter.
        """
        self._client = client
        self._model = model
        self._ratelimiter = ratelimiter

    async def __call__(self, *, response_format: type | None = None, **kwargs: Any) -> types.ModelResponse:
        """Calls the LLM backend and returns its output.

        Args:
            response_format: `The type of the expected response from the model.`
            **kwargs: Additional keyword arguments to pass to the API call.

        Returns:
            A `types.ModelResponse` object representing the model's response.
        """
        with self._ratelimiter:
            if response_format is not None:
                kwargs["response_format"] = response_format
                fn = self._client.chat.completions.parse
            else:
                fn = self._client.chat.completions.create
            return await fn(model=self._model, **kwargs)

    async def generate(self, chat: chat_lib.Chat, /, **kwargs: Any) -> types.ModelResponse:
        """Generates a response from the LLM asynchronously based on the chat history.

        Args:
            chat: A `chat_lib.Chat` object containing the conversation history.
            **kwargs: Additional keyword arguments to pass to the `__call__` method.

        Returns:
            A `types.ModelResponse` object representing the model's response.

        """
        return await self(messages=chat.messages, **kwargs)

    @classmethod
    def from_config(cls, config: config_lib.Config) -> Self:
        """Instantiates an `AsyncLLM` instance from a configuration object.

        Args:
            config: The configuration object to use.

        Returns:
            An `AsyncLLM` instance.
        """
        client, _ = config.get_async_client()
        ratelimiter = ratelimit.RateLimiter(config.rpm)
        return cls(client=client, model=config.model_name, ratelimiter=ratelimiter)
