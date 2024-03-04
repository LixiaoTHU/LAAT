import torch


def filter_state_dict(state_dict):
    from collections import OrderedDict

    if "state_dict" in state_dict.keys():
        for k, v in state_dict.items():
            if k != "state_dict":
                print("{}: {}".format(k, v))
        state_dict = state_dict["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "sub_block" in k:
            continue
        if "module" in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_model(model_path, model, optimizer, device="cpu"):
    checkpoint = torch.load(model_path + ".pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def save_model(model_path, model, optimizer, epoch, save_best=False):
    state = {
        "epoch": epoch,
        "model_state_dict": filter_state_dict(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with open(model_path + ".dat", "wb") as f:
        torch.save(state, f)
    if save_best:
        with open(model_path + "_best" + ".dat", "wb") as f:
            torch.save(state, f)
