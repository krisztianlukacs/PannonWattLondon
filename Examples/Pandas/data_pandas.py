import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Adatok betöltése CSV fájlból
df = pd.read_csv("energia_fogyasztas.csv")

# Dátum konverzió
df["időbélyeg"] = pd.to_datetime(df["időbélyeg"])
df = df.dropna()

# Alapstatisztikák
print("=" * 50)
print("ALAPSTATISZTIKÁK")
print("=" * 50)
print("\nFogyasztás statisztikái:")
print(df["fogyasztás"].describe())
print("\nHőmérséklet statisztikái:")
print(df["hőmérséklet"].describe())

# Korreláció számítás
correlation = df["fogyasztás"].corr(df["hőmérséklet"])
print(f"\nKorreláció a fogyasztás és hőmérséklet között: {correlation:.4f}")

# Havi aggregálás
df["hónap"] = df["időbélyeg"].dt.to_period("M")
monthly_stats = df.groupby("hónap").agg(
    {"fogyasztás": ["mean", "sum", "min", "max"], "hőmérséklet": "mean"}
)
print("\n" + "=" * 50)
print("HAVI ÖSSZEFOGLALÓ")
print("=" * 50)
print(monthly_stats)

# Grafikonok készítése
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    "Energia Fogyasztás és Hőmérséklet Elemzés", fontsize=16, fontweight="bold"
)

# 1. Fogyasztás időbeli alakulása
axes[0, 0].plot(df["időbélyeg"], df["fogyasztás"], linewidth=1.5, color="blue")
axes[0, 0].set_title("Fogyasztás időbeli alakulása")
axes[0, 0].set_xlabel("Dátum")
axes[0, 0].set_ylabel("Fogyasztás")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis="x", rotation=45)

# 2. Hőmérséklet időbeli alakulása
axes[0, 1].plot(df["időbélyeg"], df["hőmérséklet"], linewidth=1.5, color="red")
axes[0, 1].set_title("Hőmérséklet időbeli alakulása")
axes[0, 1].set_xlabel("Dátum")
axes[0, 1].set_ylabel("Hőmérséklet (°C)")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis="x", rotation=45)

# 3. Korreláció scatter plot
axes[1, 0].scatter(df["hőmérséklet"], df["fogyasztás"], alpha=0.6, color="green")
axes[1, 0].set_title(f"Fogyasztás vs Hőmérséklet (r={correlation:.3f})")
axes[1, 0].set_xlabel("Hőmérséklet (°C)")
axes[1, 0].set_ylabel("Fogyasztás")
axes[1, 0].grid(True, alpha=0.3)
# Trendvonal
z = np.polyfit(df["hőmérséklet"], df["fogyasztás"], 1)
p = np.poly1d(z)
axes[1, 0].plot(df["hőmérséklet"], p(df["hőmérséklet"]), "r--", alpha=0.8, linewidth=2)

# 4. Havi átlagos fogyasztás
monthly_consumption = df.groupby("hónap")["fogyasztás"].mean()
axes[1, 1].bar(
    range(len(monthly_consumption)),
    monthly_consumption.values,
    color="purple",
    alpha=0.7,
)
axes[1, 1].set_title("Átlagos havi fogyasztás")
axes[1, 1].set_xlabel("Hónap")
axes[1, 1].set_ylabel("Átlagos fogyasztás")
axes[1, 1].set_xticks(range(len(monthly_consumption)))
axes[1, 1].set_xticklabels([str(m) for m in monthly_consumption.index], rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("energia_elemzes.png", bbox_inches="tight", dpi=300)
print("\n" + "=" * 50)
print("Grafikon mentve: energia_elemzes.png")
print("=" * 50)
plt.close()

# További elemzés: Extrém értékek
print("\n" + "=" * 50)
print("EXTRÉM ÉRTÉKEK")
print("=" * 50)
print(
    f"\nLegmagasabb fogyasztás: {df['fogyasztás'].max():.2f} ({df[df['fogyasztás'] == df['fogyasztás'].max()]['időbélyeg'].values[0]})"
)
print(
    f"Legalacsonyabb fogyasztás: {df['fogyasztás'].min():.2f} ({df[df['fogyasztás'] == df['fogyasztás'].min()]['időbélyeg'].values[0]})"
)
print(
    f"\nLegmagasabb hőmérséklet: {df['hőmérséklet'].max():.2f}°C ({df[df['hőmérséklet'] == df['hőmérséklet'].max()]['időbélyeg'].values[0]})"
)
print(
    f"Legalacsonyabb hőmérséklet: {df['hőmérséklet'].min():.2f}°C ({df[df['hőmérséklet'] == df['hőmérséklet'].min()]['időbélyeg'].values[0]})"
)
