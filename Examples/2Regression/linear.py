from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Adatok betöltése
df = pd.read_csv("data.csv")
X = df[["hőmérséklet"]]
y = df["termelés"]

# Adatok felosztása
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modell képzése
model = LinearRegression()
model.fit(X_train, y_train)

# Predikció és értékelés
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# Értékelési metrikák
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("MODELL TELJESÍTMÉNY ÉRTÉKELÉS")
print("=" * 50)
print(f"Közepes négyzetes hiba (MSE): {mse:.2f}")
print(f"Gyökös közepes négyzetes hiba (RMSE): {rmse:.2f}")
print(f"Átlagos abszolút hiba (MAE): {mae:.2f}")
print(f"R² (determinációs együttható): {r2:.4f}")
print(f"\nModell paraméterek:")
print(f"Meredekség (slope): {model.coef_[0]:.4f}")
print(f"Tengelymetszet (intercept): {model.intercept_:.4f}")
print("=" * 50)

# Vizualizáció - 2x2 rács
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Lineáris Regressziós Modell Elemzése", fontsize=16, fontweight="bold")

# 1. Adatok és regressziós egyenes
ax1 = axes[0, 0]
ax1.scatter(X_train, y_train, alpha=0.6, color="blue", label="Tanító adatok", s=50)
ax1.scatter(X_test, y_test, alpha=0.6, color="green", label="Teszt adatok", s=50)

# Regressziós egyenes
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_pred = model.predict(X_range)
ax1.plot(X_range, y_range_pred, "r-", linewidth=2, label="Regressziós egyenes")

ax1.set_xlabel("Hőmérséklet (°C)", fontsize=11)
ax1.set_ylabel("Termelés", fontsize=11)
ax1.set_title("Adatok és Regressziós Egyenes", fontsize=12, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Predikciók vs Valós értékek
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred, alpha=0.6, color="purple", s=50)

# Ideális egyenes (y=x)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax2.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
    linewidth=2,
    label="Ideális előrejelzés",
)

ax2.set_xlabel("Valós értékek", fontsize=11)
ax2.set_ylabel("Előrejelzett értékek", fontsize=11)
ax2.set_title(f"Predikció vs Valós (R² = {r2:.4f})", fontsize=12, fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Hibák (reziduálisok) eloszlása
ax3 = axes[1, 0]
residuals = y_test - y_pred
ax3.scatter(y_pred, residuals, alpha=0.6, color="orange", s=50)
ax3.axhline(y=0, color="r", linestyle="--", linewidth=2)
ax3.set_xlabel("Előrejelzett értékek", fontsize=11)
ax3.set_ylabel("Hibák (Reziduálisok)", fontsize=11)
ax3.set_title("Hibák Eloszlása", fontsize=12, fontweight="bold")
ax3.grid(True, alpha=0.3)

# 4. Modell teljesítmény metrikák
ax4 = axes[1, 1]
ax4.axis("off")

# Teljesítmény értékelés szöveges formában
metrics_text = f"""
MODELL TELJESÍTMÉNY ÖSSZEFOGLALÓ

Értékelési Metrikák:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• MSE: {mse:.2f}
• RMSE: {rmse:.2f}
• MAE: {mae:.2f}
• R² Score: {r2:.4f}

Modell Paraméterek:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Meredekség: {model.coef_[0]:.4f}
• Tengelymetszet: {model.intercept_:.4f}

Adatok:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Összes adat: {len(df)}
• Tanító adatok: {len(X_train)} ({(len(X_train) / len(df) * 100):.0f}%)
• Teszt adatok: {len(X_test)} ({(len(X_test) / len(df) * 100):.0f}%)

Modell Minősítés:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# R² alapú minősítés
if r2 >= 0.9:
    quality = "Kiváló ✓✓✓"
    color = "green"
elif r2 >= 0.7:
    quality = "Jó ✓✓"
    color = "lightgreen"
elif r2 >= 0.5:
    quality = "Közepes ✓"
    color = "orange"
else:
    quality = "Gyenge ✗"
    color = "red"

metrics_text += f"• {quality}"

ax4.text(
    0.1,
    0.5,
    metrics_text,
    fontsize=11,
    verticalalignment="center",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

# Minősítés színes háttérrel
ax4.text(
    0.55,
    0.12,
    quality,
    fontsize=14,
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor=color, alpha=0.5),
)

plt.tight_layout()
plt.savefig("linear_regression_analysis.png", bbox_inches="tight", dpi=300)
print("\nVizualizáció mentve: linear_regression_analysis.png")
plt.close()
