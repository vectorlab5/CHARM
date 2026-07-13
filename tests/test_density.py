import torch

from charm.density import EmpiricalCDF, RealNVP


def test_realnvp_round_trip_and_finite_log_probability() -> None:
    torch.manual_seed(2)
    flow = RealNVP(dimension=6, layers=4, hidden_dim=16)
    inputs = torch.randn(5, 6)
    latent, _ = flow.transform(inputs)
    reconstructed = flow.inverse(latent)
    assert torch.allclose(inputs, reconstructed, atol=1e-5)
    assert torch.isfinite(flow.log_prob(inputs)).all()


def test_empirical_cdf_uses_add_one_rank() -> None:
    empirical = EmpiricalCDF.fit(torch.tensor([1.0, 2.0, 3.0]))
    scores = empirical(torch.tensor([0.0, 2.0, 4.0]))
    assert torch.allclose(scores, torch.tensor([0.25, 0.75, 1.0]))
