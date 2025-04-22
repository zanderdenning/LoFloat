import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV generated from C++
df = pd.read_csv("oscillator_8bit.csv")

# Estimate v(t) numerically using finite difference (centered diff)
h = df["t"][1] - df["t"][0]

def estimate_v(u):
    return np.gradient(u, h)

# Compute v for each curve
df["v_exact"] = -np.sin(df["t"])             # analytic derivative
df["v_rne"]   = estimate_v(df["u_rne"])
df["v_sr"]    = estimate_v(df["u_sr"])

# Phase plot: u(t) vs v(t)
plt.figure(figsize=(8, 6))
plt.plot(df["u_exact"], df["v_exact"], label="Exact", linewidth=2)
plt.plot(df["u_rne"],   df["v_rne"],   label="8-bit RNE", linestyle='--')
plt.plot(df["u_sr"],    df["v_sr"],    label="8-bit StochRnd", linestyle='--')
plt.xlabel("u(t)")
plt.ylabel("v(t)")
plt.title("Phase Portrait: u(t) vs v(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
