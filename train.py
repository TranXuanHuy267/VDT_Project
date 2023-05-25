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

import warnings
warnings.filterwarnings("ignore")


class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device, number_id, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.best_epoch = 0
        self.config = config
        self.config["number_id"] = number_id

    def train(self, trainloader, valloader, epochs):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        best_accuracy = 0
        best_model = None
        # Train the model
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
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
            print("-"*30)
        # Save the experiment
        save_experiment(self.exp_name, self.config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in tqdm(trainloader):
            # Move the batch to the device
            batch = [t.to(self.device) for t in batch]
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
        _, self.model, _, _, _ = load_experiment(self.exp_name, "bestmodel.pt")
        self.model = self.model.to(self.device)
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
                batch = [t.to(self.device) for t in batch]
                images, labels = batch
                
                # Get predictions
                logits = self.model(images)

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
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--config", type=str, default="own_config.json")
    parser.add_argument("--number-id", type=int, required=True)
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main():
    torch.manual_seed(267)
    args = parse_args()
    configfile = "configs/" + args.config
    with open(configfile, 'r') as f:
        config = json.load(f)
    
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
    device = args.device
    split_ratio = {"Train": 0.7, "Valid": 0.2, "Test": 0.1}
    trainloader, valloader, testloader= prepare_data(number_id=number_id, batch_size=batch_size, split_ratio=split_ratio)
    vit = VisionTransformer(**model_own)
    for layer in vit.children():
        for para in layer.parameters():
            para.requires_grad=True
    head = nn.Linear(1000, number_id)
    model = nn.Sequential(vit, head).to(device)
    param_groups = [
        {
            'params': [p for p in vit.parameters() if p.requires_grad],
            'lr': args.lr, 
            'weight_decay': 1e-4
        },
        {
            'params': [p for p in head.parameters() if p.requires_grad],
            'lr': args.lr, 
            'weight_decay': 1e-4
        }
    ]
    optimizer = optim.AdamW(params=param_groups)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device, number_id, config)
    trainer.train(trainloader, valloader, epochs)
    trainer.test(testloader)

if __name__ == "__main__":
    main()