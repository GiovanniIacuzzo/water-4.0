# 💧 WATER 4.0 – Predizione delle Perdite Idriche con LSTM Ottimizzata tramite CPSO

WATER 4.0 è un progetto di ricerca orientato alla previsione continua delle perdite idriche in reti di distribuzione, tramite l’impiego di una rete neurale LSTM (Long Short-Term Memory) ottimizzata attraverso un innovativo algoritmo numerico chiamato Continuous CPSO, una variante del Particle Swarm Optimization (PSO).

---

## 📌 Obiettivo del Progetto

Predire in modo accurato e continuo la quantità di **acqua persa (m³/h)** nella rete idrica, sfruttando **misurazioni storiche multivariate** (pressioni, flussi, domande e livelli). Il modello si basa su:

- Un’architettura **LSTM deep learning** per modellare le dipendenze temporali.
- Un algoritmo di ottimizzazione **CPSO (Continuous Particle Swarm Optimization)** per il tuning automatico degli iperparametri del modello.

---

## Modello Predittivo

### Architettura LSTM

Il modello implementa:

- **2 strati LSTM** (stacked) da 128 unità ciascuno.
- **Dropout** intermedio per la regolarizzazione.
- **Dense layer finale** per produrre un output continuo.

L’input è una sequenza multivariata temporale (`(seq_len, n_features)`), l’output è la **perdita predetta al tempo t+1**.

### Scelta della variabile target

La perdita è calcolata come **somma delle perdite simulate su tutti i link** della rete, per ciascun timestamp. Il task è trattato come **regressione univariata**.

### Vantaggi dell'approccio LSTM

- Modellazione di dinamiche stagionali e giornaliere.
- Resistenza al rumore di misura.
- Scalabilità a reti reali o dataset più ampi.
- Facilità di integrazione in sistemi di early warning o controllo predittivo.

---

## Ottimizzazione con Continuous CPSO

Il tuning degli iperparametri LSTM è effettuato tramite **Continuous CPSO**, una variante del Particle Swarm Optimization in grado di:

- Esplorare lo spazio continuo degli iperparametri (es. learning rate, dimensione dei layer, dropout).
- Accelerare la convergenza tramite meccanismi evolutivi adattivi.
- Garantire migliori performance rispetto al tuning manuale o con griglia.

---

## Dataset: BattLeDIM 2020

Il dataset utilizzato proviene dalla competizione internazionale **BattLeDIM 2020**, ed è basato su simulazioni realistiche della rete idrica virtuale **L-Town**.

### Variabili disponibili:

1. **Domande (Demands)** – 82 nodi (l/h)
2. **Flussi (Flows)** – 3 sensori (m³/h)
3. **Pressioni (Pressures)** – 33 sensori (m)
4. **Livelli (Levels)** – 1 serbatoio (m)
5. **Perdite (Leakages)** – perdite simulate (m³/h)

> Dati registrati ogni **5 minuti**, per un intero anno (2018).

---

## Preprocessing e Preparazione dei Dati

A cura del modulo [`utils/dataset.py`](utils/dataset.py):

- **Parsing temporale**: indicizzazione per timestamp
- **Unione delle variabili**: concatenazione orizzontale delle features
- **Pulizia dei dati**: rimozione righe con NaN
- **Costruzione target**: somma delle perdite su tutti i link
- **Normalizzazione**: via `StandardScaler` (Scikit-learn)
- **Segmentazione sequenziale**: sliding window (`seq_len`, `target`)

---

## 📁 Struttura del Progetto

```bash
WATER-4.0/
│
├── CPSO/                   # Cartella Ottimizzazione
│ ├── CPSO.py               # Algoritmo di ottimizzazione CPSO
│ ├── f_obj.py              # Funzione obiettivo + Train
│ └── ottimizzazione.py     # File che esegue l'ottimizzazione
│
├── data/                   # Cartella dei dati
│
├── models/                 # Cartella del modello 
│ └── lstm_model.py         # Definizione della rete LSTM
│
├── utils/                  # Cartella utils
│ ├── dataset.py            # Preprocessing e dataset loader
│ ├── evaluate.py           # Metriche di valutazione
│ ├── plot.py               # Visualizzazioni
│ └── train.py              # Ciclo di training
│
├── config.yaml             # Parametri del modello
├── environment.yml         # Dipendenze Conda
│
├── main.py                 # Script principale per esecuzione
│
├── .gitignore
│
└── README.md
```

---

## Requisiti e Setup

1. **Crea un ambiente conda**:

```bash
conda env create -f environment.yml
conda activate water4
```

2. **Avvia l'allenamento del modello**:
```bash
python main.py
```
### Contatti
Per domande: [giovanni.iacuzzo@unikorestudent.it](mailto:giovanni.iacuzzo@unikorestudent.it)