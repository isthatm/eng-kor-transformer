import torch

checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu",
    weights_only=True,
    mmap=True,
)

torch.save(
    {
        "model_state_dict": checkpoint["model_state_dict"],
        "config": checkpoint.get("config"),
    },
    "best_model_inference.pth",
)