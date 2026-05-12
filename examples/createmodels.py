"""
Defining a custom model and fitting it to data.

This example demonstrates:
  - Subclassing satlas2.Model to define an arbitrary function
  - Declaring parameters with bounds and vary flags
  - Generating noisy synthetic data
  - Fitting with satlas2.Fitter and inspecting the result
  - Plotting initial guess vs. fitted model

The custom model is a damped sine wave:  A * exp(-λ·x) * sin(ω·x)
"""

import matplotlib.pyplot as plt
import numpy as np

import satlas2


# ---------------------------------------------------------------------------
# Define a custom model by subclassing satlas2.Model
# ---------------------------------------------------------------------------
class DampedSine(satlas2.Model):
    """Damped sinusoid: A * exp(-lambda * x) * sin(omega * x)."""

    def __init__(self, amplitude, lam, omega, name="DampedSine", prefunc=None):
        super().__init__(name, prefunc=prefunc)
        self.params = {
            "amplitude": satlas2.Parameter(value=amplitude, min=0, max=np.inf, vary=True),
            "lambda":    satlas2.Parameter(value=lam,       min=0, max=np.inf, vary=True),
            "omega":     satlas2.Parameter(value=omega,     min=0, max=np.inf, vary=True),
        }

    def f(self, x):
        x = self.transform(x)
        A   = self.params["amplitude"].value
        lam = self.params["lambda"].value
        w   = self.params["omega"].value
        return A * np.exp(-lam * x) * np.sin(w * x)


# ---------------------------------------------------------------------------
# Construct the model with "true" parameter values and visualise it
# ---------------------------------------------------------------------------
true_amplitude = 7.0
true_lambda    = 1.5
true_omega     = 4.0

model = DampedSine(true_amplitude, true_lambda, true_omega)

x_dense = np.linspace(0, 4, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(x_dense, model.f(x_dense))
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("True model")
print("Initial parameters:")
print(model.params)

# ---------------------------------------------------------------------------
# Create noisy synthetic data by adding Gaussian noise to the true model
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
x_data = np.linspace(0, 4, 30)
sigma  = 0.5
y_data = model.f(x_data) + rng.normal(0, sigma, size=x_data.shape)
yerr   = np.full_like(y_data, sigma)

# ---------------------------------------------------------------------------
# Perturb the model parameters to simulate a realistic starting guess
# ---------------------------------------------------------------------------
model.params["amplitude"].value = 5.0
model.params["lambda"].value    = 1.0
model.params["omega"].value     = 3.5

# ---------------------------------------------------------------------------
# Wire up a Source + Fitter and run the fit
# ---------------------------------------------------------------------------
source = satlas2.Source(x_data, y_data, yerr=yerr, name="NoisyData")
source.addModel(model)

fitter = satlas2.Fitter()
fitter.addSource(source)
fitter.fit()

print("\nFit report:")
print(fitter.reportFit())

# ---------------------------------------------------------------------------
# Compare initial guess (perturbed) and fitted result
# ---------------------------------------------------------------------------
# Reset to perturbed values to reconstruct the initial-guess curve
model.params["amplitude"].value = 5.0
model.params["lambda"].value    = 1.0
model.params["omega"].value     = 3.5
y_initial = model.f(x_dense)

# Restore fitted values
fitter.fit()
y_fitted = model.f(x_dense)

axes[1].errorbar(x_data, y_data, yerr=yerr, fmt="o", label="Data", zorder=3)
axes[1].plot(x_dense, y_initial, "--", label="Initial guess")
axes[1].plot(x_dense, y_fitted,  "-",  label="Fitted model")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("Fit result")
axes[1].legend(loc=0)

plt.tight_layout()
plt.show()
