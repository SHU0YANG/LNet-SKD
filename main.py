import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from utils import BalancedEntropyLoss, AverageMeter, CICIDS2017Data
from model import LNet

warnings.filterwarnings("ignore")

# Initialize TensorBoard writer
writer = SummaryWriter('')

def plot_lr(learning_rates: list, epochs: int, path: str) -> None:
    """Plots learning rates and saves the figure."""
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, color='r')
    plt.text(0, learning_rates[0], str(learning_rates[0]))
    plt.text(epochs, learning_rates[-1], str(learning_rates[-1]))
    plt.savefig(path)
    plt.close()  # Close the plot to free memory

def train_one_epoch(args, model, optimizer, dataset, pre_data: torch.Tensor, pre_target: torch.Tensor, pre_out: torch.Tensor) -> tuple:
    """Trains the model for one epoch and returns average loss and accuracy."""
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()
    model.train()

    for data, target in tqdm(dataset.train_dataloader):
        # Forward pass
        out = model(data)

        if pre_data is not None:
            pre_images = pre_data
            pre_label = pre_target
            out_pre = model(pre_images)

            # Compute losses
            cb_loss = BalancedEntropyLoss(dataset.class_weights)(
                torch.cat((out_pre, out), dim=0),
                torch.cat((pre_label, target), dim=0)
            )
            skd_loss = F.kl_div(
                F.log_softmax(out_pre / args.tau, dim=1),
                F.softmax(pre_out.detach() / args.tau, dim=1),
                reduction="batchmean"
            ) * (args.tau ** 2)

            loss = cb_loss + (args.lamda * skd_loss)
        else:
            loss = BalancedEntropyLoss(dataset.class_weights)(out, target)

        # Calculate accuracy
        predictions = out.argmax(dim=1).cpu().numpy()
        true_labels = target.cpu().numpy()
        acc = accuracy_score(true_labels, predictions)

        # Update records
        acc_recorder.update(acc, n=data.size(0))
        loss_recorder.update(loss.item(), n=data.size(0))

        # Prepare for the next iteration
        pre_data, pre_target, pre_out = data, target, out

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_recorder.avg, acc_recorder.avg, pre_data, pre_target, pre_out

def train(args, model, dataset) -> nn.Module:
    """Trains the model and plots learning rates."""
    learning_rates = []
    pre_data, pre_target, pre_out = None, None, None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-9)

    for epoch in range(args.epochs):
        train_losses, train_acces, pre_data, pre_target, pre_out = train_one_epoch(args, model, optimizer, dataset, pre_data, pre_target, pre_out)
        learning_rates.append(optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()
        writer.add_scalar('train_loss', train_losses, global_step=epoch)
        writer.add_scalar('train_acc', train_acces, global_step=epoch)

    plot_lr(learning_rates, args.epochs, os.path.join(args.save_path, "lr.jpg"))

    return model

def evaluation(model: nn.Module, dataset) -> None:
    """Evaluates the model performance on the test dataset."""
    model.eval()
    predict_labels = []
    true_labels = []

    with torch.no_grad():
        for data, target in dataset.test_dataloader:
            out = model(data)
            predictions = out.argmax(dim=1).cpu().numpy()
            predict_labels.extend(predictions)
            true_labels.extend(target.cpu().numpy())

    # Calculate metrics
    test_acc = accuracy_score(true_labels, predict_labels)
    precision = precision_score(true_labels, predict_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predict_labels, average='weighted')
    f1 = f1_score(true_labels, predict_labels, average='weighted')

    report = classification_report(true_labels, predict_labels)
    
    print(report)
    print(f"Accuracy: {test_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

def argument_parser() -> argparse.Namespace:
    """Parses command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='LNet-SKD')
    parser.add_argument('--input_size', type=int, help='Input size')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--lamda', type=float, default=1.0, help='balance parameter for loss')
    parser.add_argument('--tau', type=float, default=1.0, help='Temperature parameter')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the figure')
    return parser.parse_args()

if __name__ == '__main__':
    args = argument_parser()

    model = LNet(args.input_size, args.num_classes)
    dataset = CICIDS2017Data(args.batch_size)
    
    train(args, model, dataset)

    evaluation(model, dataset)