
import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from model.challenge_target import Challenge_Target
from train_common import *
from utils import config
import utils
import copy

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def freeze_layers(model, num_layers=0):
    """Stop tracking gradients on selected layers."""
    #TODO: modify model with the given layers frozen
    #      e.g. if num_layers=2, freeze CONV1 and CONV2
    #      Hint: https://pytorch.org/docs/master/notes/autograd.html
    layer_count = 0
    for name, param in model.named_parameters():
        if 'conv' in name and layer_count < num_layers:
            param.requires_grad = False
            print(f"Froze layer: {name}")
            if 'bias' in name:
                layer_count += 1
        elif layer_count >= num_layers:
            break   # Stop once the required number of conv layers are frozen



def train(tr_loader, va_loader, te_loader, model, model_name, num_layers=0):
    """Train transfer learning model."""
    #TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config("challenge.learning_rate"))

    #

    print("Loading target model with", num_layers, "layers frozen")
    model, start_epoch, stats = restore_checkpoint(model, model_name)

    axes = utils.make_training_plot("Target Training")

    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    #TODO: patience for early stopping
    patience = 5
    curr_patience = 0
    #

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            include_test=True,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, model_name, stats)

        curr_patience, prev_val_loss = early_stopping(
            stats, curr_patience, prev_val_loss
        )
        epoch += 1

    print("Finished Challenge Target Training")

    # Keep plot open
    utils.save_tl_training_plot(num_layers)
    utils.hold_training_plot()


def main():
    """Train transfer learning model and display training plots.

    Train four different models with {0, 1, 2, 3} layers frozen.
    """
    # data loaders
    if check_for_augmented_data("./data"):
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target",
            batch_size=config("challenge.batch_size"), augment = True
        )
        use_augment = True
    else: 
        print("HAHAHAHA!")

    freeze_none = Challenge_Target()
    print("Loading source...")
    freeze_none, _, _ = restore_checkpoint(
        freeze_none, config("source.checkpoint"), force=True, pretrain=True
    )

    freeze_one = copy.deepcopy(freeze_none)
    freeze_two = copy.deepcopy(freeze_none)
    freeze_three = copy.deepcopy(freeze_none)

    freeze_layers(freeze_one, 1)
    freeze_layers(freeze_two, 2)
    freeze_layers(freeze_three, 3)

    train(tr_loader, va_loader, te_loader, freeze_none, "./checkpoints/challenge0/", 0)
    train(tr_loader, va_loader, te_loader, freeze_one, "./checkpoints/challenge1/", 1)
    train(tr_loader, va_loader, te_loader, freeze_two, "./checkpoints/challenge2/", 2)
    train(tr_loader, va_loader, te_loader, freeze_three, "./checkpoints/challenge3/", 3)


if __name__ == "__main__":
    main()
