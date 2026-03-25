import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.scale
import matplotlib.ticker
import matplotlib.transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")

# Enable LaTeX rendering.
# https://stackoverflow.com/a/23856968
# plt.rc('text', usetex=True)
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Computer Modern"
# Can add more commands to this list
plt.rcParams["text.latex.preamble"] = "\n".join([r"\usepackage{amsmath}"])
# Increase font size.
plt.rcParams["font.size"] = 23
plt.rcParams["grid.alpha"] = 0.5
plt.rcParams["axes.grid.which"] = "both"


def format_g_legend_in_scientific_notation(g, num_decimal_digits: int = 1):
    # Round legend labels (hue values) to 3 significant figures.
    leg = getattr(g, "_legend", g.legend)
    if not hasattr(leg, "texts"):
        leg = g.legend_
    for txt in leg.texts:  # only the item labels, not the title
        try:
            txt.set_text(f"{float(txt.get_text()):.{num_decimal_digits}e}")
        except ValueError:
            pass  # skip any non-numeric labels


def format_g_legend_to_millions_and_billions(g):
    # Get the legend object
    legend = g.get_legend()

    # Get the list of text objects in the legend
    legend_texts = legend.get_texts()

    # Iterate and update the text for each label
    for text_obj in legend_texts:
        # Get the current label (e.g., "34061856.0")
        old_label_str = text_obj.get_text()

        try:
            # Convert to a number
            num = float(old_label_str)

            if 1e6 <= num < 1e9:
                # Create the new label (e.g., "34M")
                # We use int() to truncate (so 62.8M becomes "62M")
                new_label = f"{int(num / 1e6)}M"
            if 1e9 <= num:
                new_label = f"{int(num / 1e9)}B"

            # Set the new texts
            text_obj.set_text(new_label)
        except ValueError:
            # Failsafe in case a label isn't a number
            pass


def save_plot_with_multiple_extensions(
    plot_dir: str, plot_filename: str, use_tight_layout: bool = True
):
    if use_tight_layout:
        # Ensure that axis labels don't overlap.
        plt.gcf().tight_layout()

    extensions = [
        "pdf",
        "png",
    ]
    for extension in extensions:
        plot_path = os.path.join(plot_dir, plot_filename + f".{extension}")
        print(f"Plotted {plot_path}")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
