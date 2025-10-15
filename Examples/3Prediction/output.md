python3 predict.py
2025-10-15 09:00:16.965724: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
2025-10-15 09:00:20.268241: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
2025-10-15 09:00:20.269277: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-10-15 09:00:24.137293: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
============================================================
IDŐSOR ELŐREJELZÉS LSTM MODELLEL
============================================================
Betöltött adatok száma: 200
Időintervallum: 2023-01-01 00:00:00 - 2023-07-19 00:00:00
Termelés tartomány: 323.47 - 686.87

Időablak mérete: 3 nap
Létrehozott minták száma: 197
Tanító minták: 157
Teszt minták: 40

============================================================
MODELL TANÍTÁSA...
============================================================
Tanítás befejezve!

============================================================
MODELL TELJESÍTMÉNY
============================================================
MSE (Közepes Négyzetes Hiba): 1867.58
RMSE (Gyökös Közepes Négyzetes Hiba): 43.22
MAE (Átlagos Abszolút Hiba): 34.05
R² Score: 0.7266

============================================================
PREDIKCIÓS MINTÁK (első 5 teszt adat)
============================================================
#1 - Predikció: 488.86, Valós érték: 524.82, Különbség: 35.96
#2 - Predikció: 509.38, Valós érték: 514.69, Különbség: 5.31
#3 - Predikció: 517.89, Valós érték: 501.05, Különbség: 16.84
#4 - Predikció: 509.79, Valós érték: 606.72, Különbség: 96.93
#5 - Predikció: 552.37, Valós érték: 527.31, Különbség: 25.06

============================================================
Vizualizáció mentve: lstm_time_series_prediction.png
============================================================

============================================================
JÖVŐBELI ELŐREJELZÉS (következő 7 nap)
============================================================
2023-07-20: 491.49
2023-07-21: 502.36
2023-07-22: 501.72
2023-07-23: 504.61
2023-07-24: 506.51
2023-07-25: 508.36
2023-07-26: 509.91
============================================================
