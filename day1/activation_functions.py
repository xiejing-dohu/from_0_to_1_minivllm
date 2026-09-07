"""Plot commonly used neural-network activation functions with Matplotlib."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# Keep typography consistent and preserve editable text in vector exports.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    y = np.empty_like(x)
    positive = x >= 0
    y[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    y[~positive] = exp_x / (1.0 + exp_x)
    return y


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def leaky_relu(x: np.ndarray, negative_slope: float = 0.1) -> np.ndarray:
    return np.where(x >= 0, x, negative_slope * x)


def gelu(x: np.ndarray) -> np.ndarray:
    """Fast tanh approximation used by many neural-network libraries."""
    coefficient = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(coefficient * (x + 0.044715 * x**3)))


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def plot_activation_functions(output_dir: Path) -> list[Path]:
    """Create a six-panel comparison and save PNG, SVG, and PDF versions."""
    x = np.linspace(-6.0, 6.0, 1201)
    functions = [
        ("Sigmoid", sigmoid(x), (-0.08, 1.08)),
        ("Tanh", np.tanh(x), (-1.15, 1.15)),
        ("ReLU", relu(x), (-0.5, 6.4)),
        ("Leaky ReLU (α = 0.1)", leaky_relu(x), (-0.9, 6.4)),
        ("GELU", gelu(x), (-0.7, 6.4)),
        ("SiLU / Swish", silu(x), (-0.7, 6.4)),
    ]
    colors = ["#0F4D92", "#3775BA", "#B64342", "#9A4D8E", "#42949E", "#4D4D4D"]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), sharex=True)
    for index, (ax, (name, y, ylim), color) in enumerate(
        zip(axes.flat, functions, colors)
    ):
        ax.axhline(0, color="#B8B8B8", linewidth=0.8, zorder=0)
        ax.axvline(0, color="#B8B8B8", linewidth=0.8, zorder=0)
        ax.plot(x, y, color=color, linewidth=2.2)
        ax.set_title(name, fontsize=9, fontweight="semibold", pad=5)
        ax.set_xlim(-6, 6)
        ax.set_ylim(*ylim)
        ax.set_xticks([-6, -3, 0, 3, 6])
        ax.tick_params(direction="out", length=3, width=0.8)
        ax.text(
            -0.12,
            1.04,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Input, x")
    for ax in axes[:, 0]:
        ax.set_ylabel("Output, f(x)")

    fig.suptitle("Common neural-network activation functions", fontsize=11, y=1.01)
    fig.tight_layout(pad=1.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "activation_functions"
    outputs = []
    for suffix in ("png", "svg", "pdf"):
        output_path = output_base.with_suffix(f".{suffix}")
        save_options = {"bbox_inches": "tight"}
        if suffix == "png":
            save_options["dpi"] = 300
        fig.savefig(output_path, **save_options)
        outputs.append(output_path)

    plt.close(fig)
    return outputs


if __name__ == "__main__":
    figure_dir = Path(__file__).resolve().parent / "figures"
    for saved_path in plot_activation_functions(figure_dir):
        print(f"Saved: {saved_path}")
