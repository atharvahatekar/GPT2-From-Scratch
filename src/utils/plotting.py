"""Plotting helpers for visualizing training progress."""

from __future__ import annotations


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path=None):
    """Plot training/validation loss against epochs (and tokens seen).

    A second x-axis shows the number of tokens processed. If ``save_path`` is
    given the figure is written to disk before being shown.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Second x-axis (tokens seen) sharing the same y-axis.
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # invisible, just aligns ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
