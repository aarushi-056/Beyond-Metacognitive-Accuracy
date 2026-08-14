import numpy as np
import hddm
import matplotlib.pyplot as plt
from pathlib import Path

model = hddm.load("model4.hddm")


pairs = [
    ("a(HC_conf_high)",  "v(HC_conf_high)",  "HC – High confidence"),
    ("a(HC_conf_low)",   "v(HC_conf_low)",   "HC – Low confidence"),
    ("a(SSD_conf_high)", "v(SSD_conf_high)", "SSD – High confidence"),
    ("a(SSD_conf_low)",  "v(SSD_conf_low)",  "SSD – Low confidence")
]


# Output location
output_path = Path("posterior_a_v_correlations.png")


fig, axes = plt.subplots(
    2, 2,
    figsize=(11, 9)
)

axes = axes.flatten()

print("Posterior correlations between a and v:\n")

for ax, (a_node, v_node, label) in zip(axes, pairs):

    # Extract posterior samples
    a_samples = np.asarray(
        model.nodes_db.loc[a_node, "node"].trace()
    )

    v_samples = np.asarray(
        model.nodes_db.loc[v_node, "node"].trace()
    )

    # Calculate posterior correlation
    r = np.corrcoef(a_samples, v_samples)[0, 1]

    print(f"{label:25s}: r = {r:.3f}")

    # Plot posterior samples
    ax.scatter(
        a_samples,
        v_samples,
        s=5,
        alpha=0.15
    )

    # Add regression line
    slope, intercept = np.polyfit(
        a_samples,
        v_samples,
        1
    )

    x = np.linspace(
        a_samples.min(),
        a_samples.max(),
        100
    )

    ax.plot(
        x,
        intercept + slope * x,
        linewidth=2
    )

    # Labels
    ax.set_title(
        f"{label}\n"
        f"Posterior correlation: r = {r:.3f}"
    )

    ax.set_xlabel("Boundary separation (a)")
    ax.set_ylabel("Drift rate (v)")

    ax.grid(alpha=0.2)


fig.suptitle(
    "Posterior correlations between boundary separation and drift rate",
    fontsize=14
)

plt.tight_layout()

# Save high-resolution image
fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"\nFigure saved to: {output_path.resolve()}")