import torch

from pathlib import Path


def save_model(model: torch.nn.Module, target_dir: str, path: Path) -> None:
    """
    Saves the model to the specified path

    Takes the model and path and saves the model to the specified path
    """
    torch.save(model.state_dict(), path)

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert (
        model.name.endswith(".pth")
        or model.name.endswith(".pt")
        or model.name.endswith(".safetensors")
    )
    model_save_path = target_dir_path / model.name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
