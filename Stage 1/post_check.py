import hddm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

model = hddm.load("model4.hddm")
traces = model.get_traces()

v_HC_H = traces["v(HC_conf_high)"].values
v_HC_L = traces["v(HC_conf_low)"].values
v_SSD_H = traces["v(SSD_conf_high)"].values
v_SSD_L = traces["v(SSD_conf_low)"].values


interaction = ((v_HC_H - v_HC_L) - (v_SSD_H - v_SSD_L))

interaction = np.asarray(interaction).flatten()


posterior_mean = interaction.mean()

lower, upper = np.percentile(interaction,[2.5, 97.5])

p_positive = np.mean(interaction > 0)
p_negative = np.mean(interaction < 0)

print("Posterior mean:", posterior_mean)
print(f"95% Credible Interval: [{lower:.6f}, {upper:.6f}]")
print(f"P(interaction > 0): {p_positive:.4f}")
print(f"P(interaction < 0): {p_negative:.4f}")


kde = gaussian_kde(interaction)

x = np.linspace(interaction.min(),interaction.max(),500)

density = kde(x)

fig, ax = plt.subplots(figsize=(8, 5))

# Posterior density
ax.plot(x, density, linewidth=2, label="Posterior density")

# 95% credible interval
mask = (x >= lower) & (x <= upper)

ax.fill_between(x[mask],density[mask],alpha=0.25,label="95% credible interval")

# Zero reference line and posterior mean line
ax.axvline(0, linestyle="--", linewidth=1.5, label="Zero")

ax.axvline( posterior_mean, linestyle="-", linewidth=1.5, label=f"Posterior mean = {posterior_mean:.3f}")


ax.set_xlabel("Group × Previous Confidence interaction in drift rate (v)")

ax.set_ylabel("Posterior density")

ax.set_title( "Posterior Distribution of the Drift-Rate Interaction")
ax.legend(fontsize=9)
plt.tight_layout()

plt.savefig("posterior_interaction_v.png", dpi=300, bbox_inches="tight")

plt.show()
plt.close()