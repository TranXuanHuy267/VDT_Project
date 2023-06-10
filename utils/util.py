import json, os, math
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from models.vit import VisionTransformer
from models.block import ParallelScalingBlock
from models.norm import RmsNorm


def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    
    # Save the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    
    # Save the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)
    
    # Save the model
    save_checkpoint(experiment_name, model, base_dir=base_dir)


def save_checkpoint(experiment_name, model, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'bestmodel.pt')
    torch.save(model.state_dict(), cpfile)

def load_experiment(model, experiment_name, checkpoint_name="bestmodel.pt", base_dir="experiments", multigpu=False):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model
    """
    model_args = dict(
        patch_size=config["patch_size"], 
        embed_dim=config["embed_dim"], 
        depth=config["depth"], 
        num_heads=config["num_heads"], 
        pre_norm=config["pre_norm"], 
        no_embed_class=config["no_embed_class"],
        norm_layer=RmsNorm, 
        block_fn=ParallelScalingBlock, 
        qkv_bias=config["qkv_bias"], 
        qk_norm=config["qk_norm"],
    )
    model = nn.Sequential(
        VisionTransformer(**model_args),
        nn.Linear(1000, config['number_id'])
    )
    """
    cpfile = os.path.join(outdir, checkpoint_name)
    if multigpu:
        model.module.load_state_dict(torch.load(cpfile))
        return config, model.module, train_losses, test_losses, accuracies
    
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies

def load_experiment2(experiment_name, checkpoint_name="bestmodel.pt", base_dir="experiments", multigpu=False):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model

    model_args = dict(
        patch_size=config["patch_size"], 
        embed_dim=config["embed_dim"], 
        depth=config["depth"], 
        num_heads=config["num_heads"], 
        pre_norm=config["pre_norm"], 
        no_embed_class=config["no_embed_class"],
        norm_layer=RmsNorm, 
        block_fn=ParallelScalingBlock, 
        qkv_bias=config["qkv_bias"], 
        qk_norm=config["qk_norm"],
    )
    model = nn.Sequential(
        VisionTransformer(**model_args),
        nn.Linear(1000, config['number_id'])
    )
    cpfile = os.path.join(outdir, checkpoint_name)
    if multigpu:
        model.module.load_state_dict(torch.load(cpfile))
        return config, model.module, train_losses, test_losses, accuracies
    
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies