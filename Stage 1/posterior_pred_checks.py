import hddm
import matplotlib.pyplot as plt


# Load fitted model
model = hddm.load("model4.hddm")


# Save observed data first
observed = model.data.copy()


print("Generating posterior predictive samples...")

model.gen_ppc()


# Save simulated data
predicted = model.data.copy()


print("Observed:")
print(observed.head())

print("\nPredicted:")
print(predicted.head())


# -------------------------
# RT PPC
# -------------------------

plt.figure(figsize=(8,5))

plt.hist(
    observed["rt"],
    bins=50,
    density=True,
    alpha=0.5,
    label="Observed"
)

plt.hist(
    predicted["rt"],
    bins=50,
    density=True,
    alpha=0.5,
    label="Posterior predictive"
)


plt.xlabel("Reaction time (s)")
plt.ylabel("Density")
plt.legend()

plt.title("Posterior Predictive Check - RT")


plt.savefig(
    "posterior_predictive_RT.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -------------------------
# Accuracy PPC
# -------------------------

print("\nObserved accuracy:",
      observed["response"].mean())

print("Predicted accuracy:",
      predicted["response"].mean())


print("\nPPC complete")