import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load CSV ===
df = pd.read_csv("oscillator_8bit.csv", header=None)
df.columns = ["t", "u_rne", "v_rne", "u_sr", "v_sr", "u_exact", "v_exact"]

# === Force all values to float ===
df = df.apply(pd.to_numeric, errors='coerce')

# === Clean: remove NaNs or infinities ===
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# === Extract columns as arrays ===
t       = df["t"].to_numpy()
u_rne   = df["u_rne"].to_numpy()
u_sr    = df["u_sr"].to_numpy()
u_exact = df["u_exact"].to_numpy()

v_rne   = df["v_rne"].to_numpy()
v_sr    = df["v_sr"].to_numpy()
v_exact = df["v_exact"].to_numpy()

# === Save results in df if needed later ===
df["v_exact"] = v_exact
df["v_rne"] = v_rne
df["v_sr"] = v_sr

# === Sanity check: flag any garbage ===
threshold = 10
bad = df[(df["v_rne"].abs() > threshold) | (df["v_sr"].abs() > threshold)]
if not bad.empty:
    print("⚠️ High derivative values detected:")
    print(bad)

# === Plot ===
plt.figure(figsize=(8, 6))
plt.plot(u_exact, v_exact, label="Exact", linewidth=2)
plt.plot(u_rne, v_rne, label="8-bit RNE", linestyle='--')
plt.plot(u_sr, v_sr, label="8-bit StochRnd", linestyle='--')
plt.xlabel("u(t)")
plt.ylabel("v(t)")
plt.title("Phase Portrait: u(t) vs v(t)")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
