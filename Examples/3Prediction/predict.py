import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Adatok betöltése
df = pd.read_csv("data_time_series.csv")
df["időbélyeg"] = pd.to_datetime(df["időbélyeg"])

print("=" * 60)
print("IDŐSOR ELŐREJELZÉS LSTM MODELLEL")
print("=" * 60)
print(f"Betöltött adatok száma: {len(df)}")
print(f"Időintervallum: {df['időbélyeg'].min()} - {df['időbélyeg'].max()}")
print(f"Termelés tartomány: {df['termelés'].min():.2f} - {df['termelés'].max():.2f}")

# Adatok normalizálása (LSTM jobban működik normalizált adatokkal)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df["termelés"].values.reshape(-1, 1))


# Időablak létrehozása (előző N nap alapján jósoljuk meg a következőt)
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i : i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


# Időablak mérete (hány előző napot használunk)
n_steps = 3
X, y = create_sequences(data_scaled, n_steps)

print(f"\nIdőablak mérete: {n_steps} nap")
print(f"Létrehozott minták száma: {len(X)}")

# Tanító és teszt adatok felosztása (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Tanító minták: {len(X_train)}")
print(f"Teszt minták: {len(X_test)}")

# LSTM modell építése
model = Sequential(
    [
        LSTM(50, activation="relu", return_sequences=True, input_shape=(n_steps, 1)),
        LSTM(50, activation="relu"),
        Dense(25),
        Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

print("\n" + "=" * 60)
print("MODELL TANÍTÁSA...")
print("=" * 60)

# Modell tanítása
history = model.fit(
    X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=0
)

print("Tanítás befejezve!")

# Predikció
y_pred_scaled = model.predict(X_test, verbose=0)

# Visszaskálázás eredeti értékekre
y_test_original = scaler.inverse_transform(y_test)
y_pred_original = scaler.inverse_transform(y_pred_scaled)

# Teljes tanító adat predikció (grafikonhoz)
y_train_pred_scaled = model.predict(X_train, verbose=0)
y_train_pred_original = scaler.inverse_transform(y_train_pred_scaled)
y_train_original = scaler.inverse_transform(y_train)

# Értékelési metrikák
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print("\n" + "=" * 60)
print("MODELL TELJESÍTMÉNY")
print("=" * 60)
print(f"MSE (Közepes Négyzetes Hiba): {mse:.2f}")
print(f"RMSE (Gyökös Közepes Négyzetes Hiba): {rmse:.2f}")
print(f"MAE (Átlagos Abszolút Hiba): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

print("\n" + "=" * 60)
print("PREDIKCIÓS MINTÁK (első 5 teszt adat)")
print("=" * 60)
for i in range(min(5, len(y_test_original))):
    print(
        f"#{i + 1} - Predikció: {y_pred_original[i][0]:.2f}, "
        f"Valós érték: {y_test_original[i][0]:.2f}, "
        f"Különbség: {abs(y_pred_original[i][0] - y_test_original[i][0]):.2f}"
    )

# Vizualizáció
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("LSTM Idősor Előrejelzés Elemzése", fontsize=16, fontweight="bold")

# 1. Teljes adatsor és előrejelzések
ax1 = axes[0, 0]

# Időpontok az eredeti DataFrame-ből
train_dates = df["időbélyeg"].iloc[n_steps : n_steps + len(y_train_original)]
test_dates = df["időbélyeg"].iloc[
    n_steps + len(y_train_original) : n_steps
    + len(y_train_original)
    + len(y_test_original)
]

ax1.plot(
    df["időbélyeg"],
    df["termelés"],
    "o-",
    alpha=0.5,
    label="Eredeti adatok",
    linewidth=1,
    markersize=4,
)
ax1.plot(
    train_dates,
    y_train_pred_original,
    "s-",
    alpha=0.7,
    label="Tanító predikció",
    linewidth=1.5,
    markersize=3,
)
ax1.plot(
    test_dates,
    y_pred_original,
    "^-",
    alpha=0.9,
    label="Teszt predikció",
    linewidth=2,
    markersize=5,
)
ax1.axvline(
    x=test_dates.iloc[0],
    color="red",
    linestyle="--",
    linewidth=2,
    alpha=0.5,
    label="Tanító/Teszt határ",
)

ax1.set_xlabel("Időbélyeg", fontsize=11)
ax1.set_ylabel("Termelés", fontsize=11)
ax1.set_title("Idősor és Előrejelzések", fontsize=12, fontweight="bold")
ax1.legend(loc="best")
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis="x", rotation=45)

# 2. Teszt adatok - Predikció vs Valós
ax2 = axes[0, 1]
ax2.scatter(
    y_test_original,
    y_pred_original,
    alpha=0.6,
    color="purple",
    s=80,
    edgecolors="black",
)

min_val = min(y_test_original.min(), y_pred_original.min())
max_val = max(y_test_original.max(), y_pred_original.max())
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

# 3. Tanítási veszteség (loss) alakulása
ax3 = axes[1, 0]
ax3.plot(history.history["loss"], label="Tanítási Loss", linewidth=2)
ax3.plot(history.history["val_loss"], label="Validációs Loss", linewidth=2)
ax3.set_xlabel("Epoch", fontsize=11)
ax3.set_ylabel("Loss (MSE)", fontsize=11)
ax3.set_title("Tanítási Folyamat", fontsize=12, fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_yscale("log")

# 4. Teljesítmény összefoglaló
ax4 = axes[1, 1]
ax4.axis("off")

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

summary_text = f"""
MODELL TELJESÍTMÉNY ÖSSZEFOGLALÓ

Értékelési Metrikák:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• MSE: {mse:.2f}
• RMSE: {rmse:.2f}
• MAE: {mae:.2f}
• R² Score: {r2:.4f}

Modell Konfiguráció:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Modell típus: LSTM (2 réteg)
• Időablak: {n_steps} nap
• Neurónok: 50 + 50 + 25
• Tanítási epochok: 100

Adatok:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Összes adat: {len(df)} nap
• Tanító minták: {len(X_train)}
• Teszt minták: {len(X_test)}
• Tanító/Teszt arány: 80/20

Modell Minősítés:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• {quality}

Átlagos eltérés: ±{mae:.2f}
"""

ax4.text(
    0.1,
    0.5,
    summary_text,
    fontsize=10,
    verticalalignment="center",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

# Minősítés színes háttérrel
ax4.text(
    0.55,
    0.15,
    quality,
    fontsize=14,
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor=color, alpha=0.6),
)

plt.tight_layout()
plt.savefig("lstm_time_series_prediction.png", bbox_inches="tight", dpi=300)
print("\n" + "=" * 60)
print("Vizualizáció mentve: lstm_time_series_prediction.png")
print("=" * 60)
plt.close()

# Jövőbeli előrejelzés (következő 7 nap)
print("\n" + "=" * 60)
print("JÖVŐBELI ELŐREJELZÉS (következő 7 nap)")
print("=" * 60)

last_sequence = data_scaled[-n_steps:].reshape(1, n_steps, 1)
future_predictions = []

for i in range(7):
    next_pred = model.predict(last_sequence, verbose=0)
    future_predictions.append(next_pred[0, 0])

    # Következő szekvencia frissítése
    last_sequence = np.append(
        last_sequence[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1
    )

# Visszaskálázás
future_predictions_original = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

last_date = df["időbélyeg"].max()
for i, pred in enumerate(future_predictions_original, 1):
    future_date = last_date + pd.Timedelta(days=i)
    print(f"{future_date.strftime('%Y-%m-%d')}: {pred[0]:.2f}")

print("=" * 60)
