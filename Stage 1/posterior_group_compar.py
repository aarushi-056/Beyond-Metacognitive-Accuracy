import hddm
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde #kernel density estimator
model_file = "model4.hddm"

colors = { "HC" : "blue", 
          "SSD" : "red"}
linestyle = {"conf_high":"-", 
             "conf_low" : "--"}

model = hddm.load(model_file) #import group from posterior_group comparison , do per participant mean 
def trace(name): 
    return np.asarray(model.nodes_db.node[name].trace())

v= { #.mean 
    "HC_conf_high": trace("v(HC_conf_high)"),
    "HC_conf_low": trace("v(HC_conf_low)"),
    "SSD_conf_high": trace("v(SSD_conf_high)"),
    "SSD_conf_low": trace("v(SSD_conf_low)")
} #computes drift rate dictionary where each entry contains entire posterior distribution
 # 8000 x 5 chains ~ 40000 samples 
a = {
    "HC_conf_high": trace("a(HC_conf_high)"),
    "HC_conf_low": trace("a(HC_conf_low)"),
    "SSD_conf_high": trace("a(SSD_conf_high)"),
    "SSD_conf_low": trace("a(SSD_conf_low)")
} 
def plot_density(ax, samples, label, color, linestyle="-"):
    x = np.linspace(samples.min(), samples.max(), 500)
    kde = gaussian_kde(samples)
    ax.plot(
        x,
        kde(x),
        lw=2,
        color=color,
        linestyle=linestyle,
        label=label
    )
fig, axes = plt.subplots(1, 2, figsize=(12,5))

for key, samples in v.items(): # loops over four subplots
    group = key.split("_") [0]
    conf = "_".join(key.split("_")[1:])
    plot_density( axes[0], samples, key, colors[group], linestyle[conf])
axes[0].set_title("Posterior distributions:Drift rate(v)") 
axes[0].set_xlabel("v")
axes[0].set_ylabel("probability density")
axes[0].legend(fontsize=8) 
for key, samples in a.items(): 
    group = key.split("_")[0]
    conf = "_".join(key.split("_")[1:])

    plot_density(
        axes[1],
        samples,
        key,
        colors[group],
        linestyle[conf]
    )

axes[1].set_title("Posterior distributions: Boundary separation (a)")
axes[1].set_xlabel("a")
axes[1].set_ylabel("probability density")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("posterior_distributions.png", dpi=300)
plt.close()

def posterior_summary(name, samples):
     ci = np.percentile(samples,[2.5,97.5])
     return {"Contrast": name, "Mean": samples.mean(),"SD": samples.std(ddof=1),"CI_lower": ci[0],"CI_upper": ci[1],"P(>0)": np.mean(samples>0) }
results = []

     
results.append ( posterior_summary (
                    "v:HC(high) - SSD(high)", 
                    v["HC_conf_high"] - v["SSD_conf_high"]
))
results.append ( posterior_summary (
                    "v:HC(low) - SSD(low)", 
                    v["HC_conf_low"] - v["SSD_conf_low"]                   
))
results.append ( posterior_summary (
                    "a:HC(high) - SSD(high)", 
                    a["HC_conf_high"] - a["SSD_conf_high"]
))
results.append ( posterior_summary (
                    "a:HC(low) - SSD(low)", 
                    a["HC_conf_low"] - a["SSD_conf_low"]
))
results.append(posterior_summary(
                    "v: HC(high) - HC(low)",
                    v["HC_conf_high"] - v["HC_conf_low"]   
))
results.append(posterior_summary(
                "a:SSD(high) - SSD(low)", 
                a["SSD_conf_high"] - a["SSD_conf_low"]
))
results = pd.DataFrame(results)
results = results [ [ "Contrast", "Mean", "SD", "CI_lower", "CI_upper","P(>0)"]]

results.to_csv("posterior_contrasts_1.csv", index = False) 
print(results) 
interaction_v = (
    (v["HC_conf_high"] - v["HC_conf_low"])
    - (v["SSD_conf_high"] - v["SSD_conf_low"]))

interaction_a = (
    (a["HC_conf_high"] - a["HC_conf_low"])
    - (a["SSD_conf_high"] - a["SSD_conf_low"]))

print(posterior_summary(
    "Interaction: v",
    interaction_v
))

print(posterior_summary(
    "Interaction: a",
    interaction_a
))

print ("bazinga")



conditions = [
    "HC_conf_high",
    "HC_conf_low",
    "SSD_conf_high",
    "SSD_conf_low"
]

print("\nPosterior correlations between a and v:\n")

for cond in conditions:
    a_trace = trace(f"a({cond})")
    v_trace = trace(f"v({cond})")

    r = np.corrcoef(a_trace, v_trace)[0, 1]

    print(f"{cond:15s} : r = {r:.3f}")