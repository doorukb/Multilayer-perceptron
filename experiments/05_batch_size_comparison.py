from __future__ import annotations
import _path_setup
import matplotlib.pyplot as plt
import numpy as np
from mlp.data import create_train_and_test
from mlp.init import init_mlp
from mlp.optimizer import grad_descent

# compare SGD, mini-batch, and full-batch gradient descent on the same problem
# same init, lr, and epochs; only batch size changes

ARCH = [2, 10, 10, 1]
EPOCHS = 2000
LEARNING_RATE = 0.05
INIT_SEED = 42
SHUFFLE_SEED = 0
OUTPUT_PATH = "05_batch_size_comparison.png"

BATCH_CONFIGS: list[tuple[str, int | None]] = [
    ("batch=1", 1),
    ("batch=8", 8),
    ("batch=32", 32),
    ("full", None),
]

def main() -> None:
    np.random.seed(0)
    train_data, _ = create_train_and_test()
    n = train_data.shape[0]

    results: list[tuple[str, list[float]]] = []

    print(f"Architecture {ARCH}  |  epochs={EPOCHS}  lr={LEARNING_RATE}  n={n}")
    print(f"{'batch_size':<10} {'initial':>10} {'final':>10}")
    print("-" * 32)

    for label, batch_size in BATCH_CONFIGS:
        np.random.seed(INIT_SEED)
        model = init_mlp(ARCH)
        losses, _ = grad_descent(train_data, model, EPOCHS, LEARNING_RATE, batch_size=batch_size, seed=SHUFFLE_SEED)
        results.append((label, losses))
        display_bs = "full" if batch_size is None else str(batch_size)
        print(f"{display_bs:<10} {losses[0]:>10.4f} {losses[-1]:>10.4f}")

    plt.figure(figsize=(9, 5))
    for label, losses in results:
        plt.plot(losses, label=label)
    plt.xlabel("epoch")
    plt.ylabel("train MSE")
    plt.title("Batch size comparison on [2, 10, 10, 1] — SGD is noisier; full-batch is smooth but fewer updates per epoch")
    plt.axhline(0.25, color="gray", linestyle=":", linewidth=1, label="noise floor ~0.25")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"\nSaved plot to {OUTPUT_PATH}")
    plt.show()

if __name__ == "__main__":
    main()