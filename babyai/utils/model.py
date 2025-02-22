import os
import torch

from .. import utils


def get_model_dir(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name)


def get_model_path(model_name, epoch=None):
    if epoch:
        filename = "model_epoch" + str(epoch) +"_chkpt.pt"
        return os.path.join(get_model_dir(model_name), filename)
    else:
        return os.path.join(get_model_dir(model_name), "model.pt")


def load_model(model_name, raise_not_found=True):
    path = get_model_path(model_name)
    try:
        if torch.cuda.is_available():
            model = torch.load(path)
        else:
            model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()
        return model
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No model found at {}".format(path))


def save_model(model, model_name, optimizer_dict=None, epoch=None):
    if optimizer_dict is None:
        path = get_model_path(model_name, epoch=epoch)
        utils.create_folders_if_necessary(path)
        torch.save(model, path)
    else:
        path = get_model_path(model_name, epoch=epoch)
        utils.create_folders_if_necessary(path)
        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer_state_dict': optimizer_dict
            }, path)
