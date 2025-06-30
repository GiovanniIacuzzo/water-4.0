import pandas as pd
import matplotlib.pyplot as plt

# Carica il foglio Demands dal file Excel
file_path = "/Users/giovanni02/Desktop/Progetti/water-4.0/data/2018_SCADA.xlsx"
df = pd.read_excel(file_path, sheet_name='Demands (L_h)', engine='openpyxl')

# Pre-processing
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index('Timestamp')
df = df.sort_index()

# 1️⃣ Statistiche di base
print("\n📊 Statistiche base per ogni sensore:")
print(df.describe().transpose()[['mean', 'std', 'min', 'max']])

# 2️⃣ Analisi giorni tipici/anomali
daily = df.resample("D").sum()
daily_total = daily.sum(axis=1)

print("\n🕵️ Giorno con massimo consumo:", daily_total.idxmax(), "-", daily_total.max())
print("🕵️ Giorno con minimo consumo:", daily_total.idxmin(), "-", daily_total.min())

# Istogramma giorni
plt.figure(figsize=(8,4))
daily_total.plot(kind='hist', bins=40, title="Distribuzione consumo giornaliero [2018]", color='skyblue')
plt.xlabel("Consumo totale giornaliero (L)")
plt.grid()
plt.tight_layout()
plt.show()

# 3️⃣ Profilo orario medio (baseline)
df['hour'] = df.index.hour
baseline = df.groupby('hour').mean()

plt.figure(figsize=(8,4))
baseline.mean(axis=1).plot(title="Consumo medio per ora (Baseline 2018)", color='green')
plt.xlabel("Ora del giorno")
plt.ylabel("Media [L/h]")
plt.grid()
plt.tight_layout()
plt.show()

# Carica il foglio Demands dal file Excel
file_path = "/Users/giovanni02/Desktop/Progetti/water-4.0/data/2019_SCADA.xlsx"
df = pd.read_excel(file_path, sheet_name='Demands (L_h)', engine='openpyxl')

# Pre-processing
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index('Timestamp')
df = df.sort_index()

# 1️⃣ Statistiche di base
print("\n📊 Statistiche base per ogni sensore:")
print(df.describe().transpose()[['mean', 'std', 'min', 'max']])

# 2️⃣ Analisi giorni tipici/anomali
daily = df.resample("D").sum()
daily_total = daily.sum(axis=1)

print("\n🕵️ Giorno con massimo consumo:", daily_total.idxmax(), "-", daily_total.max())
print("🕵️ Giorno con minimo consumo:", daily_total.idxmin(), "-", daily_total.min())

# Istogramma giorni
plt.figure(figsize=(8,4))
daily_total.plot(kind='hist', bins=40, title="Distribuzione consumo giornaliero [2019]", color='skyblue')
plt.xlabel("Consumo totale giornaliero (L)")
plt.grid()
plt.tight_layout()
plt.show()

# 3️⃣ Profilo orario medio (baseline)
df['hour'] = df.index.hour
baseline = df.groupby('hour').mean()

plt.figure(figsize=(8,4))
baseline.mean(axis=1).plot(title="Consumo medio per ora (Baseline 2019)", color='green')
plt.xlabel("Ora del giorno")
plt.ylabel("Media [L/h]")
plt.grid()
plt.tight_layout()
plt.show()
