import pandas as pd
import hddm
import matplotlib.pyplot as plt

best_model = hddm.load("model4.hddm")
stats = best_model.gen_stats()
best_model.nodes_db.to_csv("nodes_test.csv")

print("Posterior Parameter estimates")
print(stats)
stats.to_csv("model4_parameter_estimates.csv")
print("bazinga")

best_model.plot_posteriors()
plt.tight_layout()
plt.show()
nodes = [
    "a(HC_conf_high)",
    "a(HC_conf_low)",
    "a(SSD_conf_high)",
    "a(SSD_conf_low)"
]

fig, axes = plt.subplots(len(nodes), 1, figsize=(6, 8))

for ax, n in zip(axes, nodes):
    trace = best_model.nodes_db.node[n].trace()
    ax.hist(trace, bins=50)
    ax.set_title(n, fontsize=9)
    ax.set_xlabel("a")
    ax.set_ylabel("frequency")

plt.tight_layout(rect=[0.08, 0.05, 1, 1])
plt.savefig("model4_posteriors.png", dpi=150)

print("saved")

