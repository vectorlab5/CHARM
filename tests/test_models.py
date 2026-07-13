import torch

from charm.losses import AdversarialClassifier, representation_loss
from charm.models.diffusion import MotionDenoiser, MotionDiffusion
from charm.models.representation import CHARMRepresentation


def test_representation_shapes_and_loss_backward() -> None:
    torch.manual_seed(1)
    model = CHARMRepresentation(
        num_joints=3,
        sequence_length=5,
        hidden_dim=16,
        style_dim=6,
        execution_dim=6,
        transformer_layers=1,
        attention_heads=2,
        dropout=0.0,
    )
    motion = torch.randn(4, 5, 3, 3)
    categories = torch.tensor([0, 0, 1, 1])
    performers = torch.tensor([0, 1, 0, 1])
    performer_adversary = AdversarialClassifier(6, 2, 8)
    category_adversary = AdversarialClassifier(6, 2, 8)
    reconstructed, style, execution = model(motion)
    assert reconstructed.shape == motion.shape
    assert style.shape == execution.shape == (4, 6)
    output = representation_loss(
        model,
        motion,
        categories,
        performers,
        performer_adversary,
        category_adversary,
    )
    output.total.backward()
    assert torch.isfinite(output.total)


def test_diffusion_training_and_short_sampling() -> None:
    torch.manual_seed(3)
    denoiser = MotionDenoiser(
        num_joints=3,
        style_dim=6,
        hidden_dim=16,
        layers=1,
        attention_heads=2,
        dropout=0.0,
    )
    diffusion = MotionDiffusion(denoiser, steps=3, condition_dropout=0.0)
    motion = torch.randn(2, 4, 3, 3)
    style = torch.randn(2, 6)
    retrieved = torch.randn(2, 2, 6)
    loss = diffusion.training_loss(motion, style, retrieved)
    assert torch.isfinite(loss)
    sampled = diffusion.sample(motion.shape, style, retrieved, guidance_weight=1.0)
    assert sampled.shape == motion.shape
    assert torch.isfinite(sampled).all()
