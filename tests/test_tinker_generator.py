import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.llm.generator import TinkerGenerator, build_generator


class _FakeFuture:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeTokenizer:
    eos_token_id = 99

    def __init__(self):
        self.last_messages = None
        self.last_tokens = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        self.last_messages = messages
        return [101, 102, 103]

    def decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        self.last_tokens = list(tokens)
        return "decoded output"


class _FakeSamplingClient:
    def __init__(self):
        self.sample_kwargs = None
        self.tokenizer = _FakeTokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    def sample(self, **kwargs):
        self.sample_kwargs = kwargs
        result = SimpleNamespace(
            sequences=[SimpleNamespace(tokens=[201, 202, 203])]
        )
        return _FakeFuture(result)


class _FakeServiceClient:
    latest_init_kwargs = None
    latest_kwargs = None
    latest_client = None

    def __init__(self, **kwargs):
        type(self).latest_init_kwargs = kwargs

    def create_sampling_client(self, **kwargs):
        type(self).latest_kwargs = kwargs
        type(self).latest_client = _FakeSamplingClient()
        return type(self).latest_client


class _FakeModelInput:
    @classmethod
    def from_ints(cls, ints):
        return {"ints": list(ints)}


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fake_tinker(monkeypatch):
    fake_tinker = ModuleType("tinker")
    fake_tinker.ServiceClient = _FakeServiceClient
    fake_tinker.AuthenticationError = type("AuthenticationError", (Exception,), {})
    fake_tinker.types = SimpleNamespace(
        ModelInput=_FakeModelInput,
        SamplingParams=_FakeSamplingParams,
    )
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)


def test_build_generator_uses_tinker_backend(monkeypatch):
    _install_fake_tinker(monkeypatch)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    config = {
        "model": {
            "generator": {
                "backend": "tinker",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "api_key_env_var": "TINKER_API_KEY",
            }
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 0.95,
            "max_new_tokens": 64,
        },
    }

    generator = build_generator(config)

    assert isinstance(generator, TinkerGenerator)

    text = generator.generate(
        [{"role": "user", "content": "Help me invest"}],
        max_new_tokens=17,
    )

    assert text == "decoded output"
    assert _FakeServiceClient.latest_init_kwargs == {"api_key": "test-key"}
    assert _FakeServiceClient.latest_kwargs == {
        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507"
    }

    sample_kwargs = _FakeServiceClient.latest_client.sample_kwargs
    assert sample_kwargs["prompt"] == {"ints": [101, 102, 103]}
    assert sample_kwargs["num_samples"] == 1
    assert sample_kwargs["sampling_params"].kwargs["max_tokens"] == 17
    assert sample_kwargs["sampling_params"].kwargs["stop"] == [99]


def test_tinker_generator_requires_api_key(monkeypatch, tmp_path):
    _install_fake_tinker(monkeypatch)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    config = {
        "model": {
            "generator": {
                "backend": "tinker",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "api_key_env_var": "TINKER_API_KEY",
            }
        }
    }

    with pytest.raises(EnvironmentError):
        TinkerGenerator(config)


def test_tinker_generator_reads_api_key_from_dotenv(monkeypatch, tmp_path):
    _install_fake_tinker(monkeypatch)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TINKER_API_KEY=dotenv-key\n", encoding="utf-8")

    config = {
        "model": {
            "generator": {
                "backend": "tinker",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "api_key_env_var": "TINKER_API_KEY",
            }
        }
    }

    generator = TinkerGenerator(config)
    assert isinstance(generator, TinkerGenerator)


def test_tinker_generator_normalizes_batch_encoding_shape(monkeypatch):
    _install_fake_tinker(monkeypatch)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    class _BatchEncodingTokenizer(_FakeTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is True
            self.last_messages = messages
            return {
                "input_ids": [[301, 302, 303]],
                "attention_mask": [[1, 1, 1]],
            }

    class _BatchEncodingSamplingClient(_FakeSamplingClient):
        def __init__(self):
            super().__init__()
            self.tokenizer = _BatchEncodingTokenizer()

    class _BatchEncodingServiceClient(_FakeServiceClient):
        def create_sampling_client(self, **kwargs):
            type(self).latest_kwargs = kwargs
            type(self).latest_client = _BatchEncodingSamplingClient()
            return type(self).latest_client

    fake_tinker = ModuleType("tinker")
    fake_tinker.ServiceClient = _BatchEncodingServiceClient
    fake_tinker.types = SimpleNamespace(
        ModelInput=_FakeModelInput,
        SamplingParams=_FakeSamplingParams,
    )
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)

    config = {
        "model": {
            "generator": {
                "backend": "tinker",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "api_key_env_var": "TINKER_API_KEY",
            }
        }
    }

    generator = TinkerGenerator(config)
    prompt = generator._prompt_from_messages([{"role": "user", "content": "hello"}])
    assert prompt == {"ints": [301, 302, 303]}


def test_tinker_generator_refreshes_once_on_auth_failure(monkeypatch):
    auth_error_cls = type("AuthenticationError", (Exception,), {})

    class _RetrySamplingClient(_FakeSamplingClient):
        def __init__(self, fail=False):
            super().__init__()
            self.fail = fail

        def sample(self, **kwargs):
            self.sample_kwargs = kwargs
            if self.fail:
                return _FakeFuture(exc=auth_error_cls("Invalid JWT"))
            return super().sample(**kwargs)

    class _RetryServiceClient(_FakeServiceClient):
        create_calls = 0

        def create_sampling_client(self, **kwargs):
            type(self).latest_kwargs = kwargs
            fail = type(self).create_calls == 0
            type(self).create_calls += 1
            type(self).latest_client = _RetrySamplingClient(fail=fail)
            return type(self).latest_client

    fake_tinker = ModuleType("tinker")
    fake_tinker.ServiceClient = _RetryServiceClient
    fake_tinker.AuthenticationError = auth_error_cls
    fake_tinker.types = SimpleNamespace(
        ModelInput=_FakeModelInput,
        SamplingParams=_FakeSamplingParams,
    )
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    config = {
        "model": {
            "generator": {
                "backend": "tinker",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "api_key_env_var": "TINKER_API_KEY",
            }
        }
    }

    generator = TinkerGenerator(config)
    text = generator.generate([{"role": "user", "content": "hello"}])

    assert text == "decoded output"
    assert _RetryServiceClient.create_calls == 2


def test_tinker_generator_retries_multiple_auth_failures(monkeypatch):
    auth_error_cls = type("AuthenticationError", (Exception,), {})

    class _RetrySamplingClient(_FakeSamplingClient):
        def __init__(self, fail=True):
            super().__init__()
            self.fail = fail

        def sample(self, **kwargs):
            self.sample_kwargs = kwargs
            if self.fail:
                return _FakeFuture(exc=auth_error_cls("Invalid JWT"))
            return super().sample(**kwargs)

    class _RetryServiceClient(_FakeServiceClient):
        create_calls = 0

        def create_sampling_client(self, **kwargs):
            type(self).latest_kwargs = kwargs
            fail = type(self).create_calls < 2
            type(self).create_calls += 1
            type(self).latest_client = _RetrySamplingClient(fail=fail)
            return type(self).latest_client

    fake_tinker = ModuleType("tinker")
    fake_tinker.ServiceClient = _RetryServiceClient
    fake_tinker.AuthenticationError = auth_error_cls
    fake_tinker.types = SimpleNamespace(
        ModelInput=_FakeModelInput,
        SamplingParams=_FakeSamplingParams,
    )
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    config = {
        "model": {
            "generator": {
                "backend": "tinker",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "api_key_env_var": "TINKER_API_KEY",
            }
        },
        "runtime": {
            "resilience": {
                "tinker_request_max_attempts": 4,
                "tinker_auth_retry_attempts": 2,
                "tinker_retry_base_delay_seconds": 0,
                "tinker_retry_max_delay_seconds": 0,
            }
        },
    }

    generator = TinkerGenerator(config)
    text = generator.generate([{"role": "user", "content": "hello"}])

    assert text == "decoded output"
    assert _RetryServiceClient.create_calls == 3
