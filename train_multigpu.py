import torch
from torch import nn, optim
import os
from utils.util import save_experiment, save_checkpoint, load_experiment
from utils.datastream import prepare_data
from models.vit import VisionTransformer
from models.block import ParallelScalingBlock
from models.norm import RmsNorm
from tqdm import tqdm
import json

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import warnings
warnings.filterwarnings("ignore")


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes 
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "2607"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, gpu_id, number_id, config):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        if gpu_id == 0:
            self.best_epoch = 0
        self.config = config
        self.config["number_id"] = number_id
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def train(self, trainloader, valloader, epochs):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        if self.gpu_id==0:
            train_losses, test_losses, accuracies = [], [], []
            best_accuracy = 0
            best_model = None
        # Train the model
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            if self.gpu_id==0:
                accuracy, test_loss = self.evaluate(valloader)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                accuracies.append(accuracy)
                print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Val loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
                if accuracy > best_accuracy:
                    print('Save checkpoint at epoch', i+1)
                    best_accuracy = accuracy
                    best_model = self.model
                    self.best_epoch = i+1
                    # Save the experiment
                    save_experiment(self.exp_name, self.config, self.model.module, train_losses, test_losses, accuracies)
                print("-"*30)
    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in tqdm(trainloader):
            # Move the batch to the device
            batch = [t.to(self.gpu_id) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            loss = self.loss_fn(self.model(images), labels)
            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    def test(self, testloader):
        _, self.model, _, _, _ = load_experiment(self.exp_name, "bestmodel.pt", multigpu=True)
        self.model = self.model.to(self.gpu_id)
        accuracy, test_loss = self.evaluate(testloader)
        print(f"Test: Best epoch: {self.best_epoch}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        predictionss = []
        labelss = []
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.gpu_id) for t in batch]
                images, labels = batch
                
                # Get predictions
                logits = self.model.module(images)

                # Calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                predictionss.append(predictions)
                labelss.append(labels)
                correct += torch.sum(predictions == labels).item()
        predictionss = torch.cat(predictionss, dim=0)
        labelss = torch.cat(labelss, dim=0)
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--config", type=str, default="own_config.json")
    parser.add_argument("--number-id", type=int, required=True)
    args = parser.parse_args()
    return args


def main(rank, world_size, args, config):
    ddp_setup(rank, world_size)    
    model_own = dict(
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
    number_id = args.number_id
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    split_ratio = {"Train": 0.7, "Valid": 0.2, "Test": 0.1}
    trainloader, valloader, testloader= prepare_data(number_id=number_id, batch_size=batch_size, split_ratio=split_ratio, multigpu=True)
    vit = VisionTransformer(**model_own)
    mlp = nn.Linear(1000, number_id)
    model = nn.Sequential(vit, mlp)
    param_groups = [
        {
            'params': [p for p in vit.parameters() if p.requires_grad],
            'lr': args.lr, 
            'weight_decay':args.weight_decay
        },
        {
            'params': [p for p in mlp.parameters() if p.requires_grad],
            'lr': args.lr, 
            'weight_decay': args.weight_decay
        }
    ]
    optimizer = optim.AdamW(params=param_groups)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, rank, number_id, config)
    trainer.train(trainloader, valloader, epochs)
    if rank == 0:
        trainer.test(testloader)
    destroy_process_group()


if __name__ == "__main__":
    torch.manual_seed(267)
    world_size=torch.cuda.device_count()
    print("Number of GPU:", world_size)
    args = parse_args()
    configfile = "configs/" + args.config
    with open(configfile, 'r') as f:
        config = json.load(f)
    mp.spawn(main, args=(world_size, args, config,), nprocs=world_size)