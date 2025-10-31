# torch_filter/tests/conftest.py
# pytest fixtures for the torch_filter FastAPI app
import sys
from pathlib import Path
import types
import torch
import pytest

@pytest.fixture(scope="session")
def app_module(monkeypatch):
    """
    Import torch_filter.service with heavy deps mocked/short-circuited:
    - Monkeypatch transformers.AutoTokenizer.from_pretrained to a tiny dummy.
    - Inject fake TorchScript models for text and image.
    """
    # Ensure the torch_filter directory is on sys.path
    root = Path(__file__).resolve().parents[1]  # .../torch_filter
    monkeypatch.syspath_prepend(str(root))

    # Prepare a dummy tokenizer that returns fixed-size tensors
    class DummyTokenizer:
        def __call__(self, text, padding="max_length", truncation=True, max_length=16, return_tensors="pt"):
            # Create 1 x max_length tensors of ones
            input_ids = torch.ones(1, max_length, dtype=torch.long)
            attention_mask = torch.ones(1, max_length, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    # Monkeypatch transformers.AutoTokenizer.from_pretrained BEFORE importing service
    import importlib
    transformers = importlib.import_module("transformers")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: DummyTokenizer())

    # Reduce MAXLEN to keep tests snappy
    monkeypatch.setenv("MAXLEN", "16")

    # Now import the FastAPI service module
    service = importlib.import_module("service")

    # Fake TorchScript-like callables
    class FakeTextTS:
        def __call__(self, input_ids, attention_mask):
            # Return a deterministic low "unsafe" logit
            return torch.tensor([-10.0])

    class FakeImageTS:
        def __call__(self, x):
            # Return a deterministic high "unsafe" logit
            return torch.tensor([10.0])

    # Attach fakes to the imported module
    service.text_model = FakeTextTS()
    service.image_model = FakeImageTS()

    return service


@pytest.fixture()
def client(app_module):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)
