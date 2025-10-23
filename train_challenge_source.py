import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from dataset_challenge import get_combined_train_loader
from model.challenge_source import ChallengeSource
from train_common import *
# from train_challenge_common import *
from utils import config
import utils

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():
    """Train source model on multiclass data."""
    # Data loaders
    if check_for_augmented_data("./data"):
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="source",
            batch_size=config("challenge.batch_size"), augment = True
        )
        # tr_loader, _ = get_combined_train_loader(
        #     task="source", batch_size=config("challenge.batch_size")
        # )
        use_augment = True
    else:
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="source",
            batch_size=config("challenge.batch_size"),
        )
        use_augment = False

        
    # Model
    model = ChallengeSource()

    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config("challenge.learning_rate"), weight_decay=0.1)
    #

    print("Number of float-valued parameters:", count_parameters(model))

    # Attempts to restore the latest checkpoint if exists
    print("Loading source for challenge...")
    model, start_epoch, stats = restore_checkpoint(model, config("source.checkpoint"))

    axes = utils.make_training_plot("Challenge Source Training")

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        multiclass=True,
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    # TODO: patience for early stopping
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
            multiclass=True,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, config("source.checkpoint"), stats)

        curr_patience, prev_val_loss = early_stopping(
            stats, curr_patience, prev_val_loss
        )
        epoch += 1

    # Save figure and keep plot open
    print("Finished Challenge source Training")
    utils.save_source_training_plot(patience)
    utils.hold_training_plot()


if __name__ == "__main__":
    main()
