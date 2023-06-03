import torch
from torch import nn, optim
import os
from utils.util import save_experiment, save_checkpoint, load_experiment
from utils.datastream import prepare_data
# from models.vit import VisionTransformer
from models.block import ParallelScalingBlock
from models.norm import RmsNorm
from models.vit_inlay import Model
from tqdm import tqdm
import json

import warnings
warnings.filterwarnings("ignore")

# Extend pretrained
# from transformers import ViTForImageClassification
from collections import OrderedDict
from transformers import ViTModel, ViTMAEModel

# mmpretrain
from mmpretrain import get_model
from mmpretrain.models.heads.margin_head import ArcFaceClsHead


class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device, number_id, config, args):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.best_epoch = 0
        self.config = config
        self.config["number_id"] = number_id
        self.args = args

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
            
            """
            loss = self.loss_fn(self.model(images), labels)
            """
            if self.args.use_inlay==0:
                # feature = self.model.backborn.extract_feat(images)[0]
                # output = self.model.head(feature)
                feature = self.model.image_encoder(images).last_hidden_state[:,0,]
                print(feature.size())
                output = self.model.head(feature)
                print(output.size())
            else:
                output = self.model(images)
            loss = self.loss_fn(output, labels)
            
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
                # output = self.model(images)
                
                if self.args.use_inlay==0:
                    # feature = self.model.backborn.extract_feat(images)[0]
                    # output = self.model.head(feature)
                    feature = self.model.image_encoder(images).last_hidden_state[:, 0,]
                    output = self.model.head(feature)
                else:
                    output = self.model(images)
                    
                # Calculate the loss
                loss = self.loss_fn(output, labels)
                
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(output, dim=1)
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--config", type=str, default="own_config.json")
    parser.add_argument("--number-id", type=int, required=True)
    
    # update inlay
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--use-inlay", type=int, default=0)
    parser.add_argument("--train-value", type=int, default=1)
    parser.add_argument("--std", type=int, default=1)
    parser.add_argument('--norm_type', type=str, default='nonorm', help="{'nonorm', 'contextnorm', 'tasksegmented_contextnorm'}")
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--ignore_diag', type=int, default=1)
    parser.add_argument('--project_higher', type=int, default=1)
    
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
        # patch_size=config["patch_size"], 
        patch_size=8,
        embed_dim=config["embed_dim"], 
        depth=config["depth"], 
        num_heads=config["num_heads"], 
        pre_norm=config["pre_norm"], 
        no_embed_class=config["no_embed_class"],
        norm_layer=RmsNorm, 
        block_fn=ParallelScalingBlock, 
        qkv_bias=config["qkv_bias"], 
        qk_norm=config["qk_norm"],

        drop_rate= 0.1,
        pos_drop_rate = 0.1,
        patch_drop_rate = 0.1,
        proj_drop_rate = 0.1,
        attn_drop_rate = 0.1,
        drop_path_rate = 0.1,
    )
    
    # update inlay:
    if args.use_inlay:
        args.train_value=1
        args.std=1
    number_id = args.number_id
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    split_ratio = {"Train": 0.7, "Valid": 0.2, "Test": 0.1}
    trainloader, valloader, testloader= prepare_data(number_id=number_id, batch_size=batch_size, split_ratio=split_ratio)
    
    """
    # 
    vit = VisionTransformer(**model_own)
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
    """
    
    if args.use_inlay:
        model = Model(args, model_own).to(device)
    else:
        head = ArcFaceClsHead(
            num_classes=number_id,
            in_channels=2048,
        )
        # backborn = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # backborn = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        backborn = get_model('resnet50-arcface_8xb32_inshop', pretrained=True)
        # backborn2 = get_model("vit-base-p16_32xb128-mae_in1k", pretrained=True)
        backborn2 = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        backborn.image_encoder = backborn2
        backborn.head = head
        model = backborn
        """model = nn.Sequential(OrderedDict([
            ("backborn", backborn),
            ("head", head)
        ])).to(device)"""

    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Number of parameter:", pytorch_total_params)
    param_groups = [
        {
            'params': [p for p in model.parameters() if p.requires_grad],
            'lr': args.lr, 
            'weight_decay': 1e-2
        },
    ]
    for name, params in model.named_parameters():
        pass
        print(name, params.size(), params.requires_grad)
    
    optimizer = optim.AdamW(params=param_groups)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device, number_id, config, args)
    trainer.train(trainloader, valloader, epochs)
    trainer.test(testloader)

if __name__ == "__main__":
    main()