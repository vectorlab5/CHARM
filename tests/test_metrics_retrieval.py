import torch

from charm.metrics import foot_skate_rate
from charm.retrieval import RetrievalMemory


def test_retrieval_is_category_restricted() -> None:
    styles = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    categories = torch.tensor([0, 0, 1])
    memory = RetrievalMemory(styles, categories)
    retrieved, scores = memory.query(torch.tensor([[1.0, 0.0]]), torch.tensor([0]), k=2)
    assert retrieved.shape == (1, 2, 2)
    assert scores.shape == (1, 2)
    assert torch.all(retrieved[0, :, 0] > retrieved[0, :, 1])


def test_foot_skate_detects_contact_displacement() -> None:
    motion = torch.zeros(1, 3, 2, 3)
    motion[:, 2, 1, 0] = 0.1
    contact = torch.ones(1, 3, 1, dtype=torch.bool)
    rate = foot_skate_rate(motion, contact, foot_indices=(1,), threshold=0.02)
    assert torch.allclose(rate, torch.tensor([0.5]))
