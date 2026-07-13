from charm.config import load_config


def test_extended_config_matches_project_defaults() -> None:
    config = load_config("configs/dunhuang.yaml")
    assert config["model"]["style_dim"] == 128
    assert config["model"]["execution_dim"] == 128
    assert config["reference"]["flow_layers"] == 8
    assert config["reference"]["retrieval_k"] == 8
    assert config["diffusion"]["steps"] == 1000
    assert config["diffusion"]["guidance_weight"] == 3.0
    assert config["num_categories"] == 8
