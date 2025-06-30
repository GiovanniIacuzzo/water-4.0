import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os

REAL_PATH = "data/leak_data.csv"
GEN_PATH = "generated_samples.csv"
OUT_DIR = "visualizations"
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_prepare():
    real = pd.read_csv(REAL_PATH)
    gen = pd.read_csv(GEN_PATH)

    real['source'] = 'Real'
    gen['source'] = 'Generated'

    # Uniformare tipi
    gen['weekday'] = gen['weekday'].astype(str)
    gen['month'] = gen['month'].astype(str)
    gen['leak_type'] = gen['leak_type'].astype(str)

    return pd.concat([real, gen], axis=0, ignore_index=True)


def plot_distribution(data, column, title=None):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=data, x=column, hue='source', fill=True, common_norm=False, alpha=0.4)
    plt.title(title or f"Distribuzione di {column}")
    plt.savefig(f"{OUT_DIR}/dist_{column}.png")
    plt.close()


def plot_boxplot(data, column, by='source'):
    plt.figure(figsize=(6, 5))
    sns.boxplot(x=by, y=column, data=data)
    plt.title(f"Boxplot {column}")
    plt.savefig(f"{OUT_DIR}/box_{column}.png")
    plt.close()


def plot_scatter(data, x='duration', y='severity'):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=data, x=x, y=y, hue='source', alpha=0.6)
    plt.title("Scatterplot durata vs severità")
    plt.savefig(f"{OUT_DIR}/scatter_duration_severity.png")
    plt.close()


def run_ks_tests(real, gen):
    ks_results = {}
    for col in ['duration', 'severity']:
        stat, p = ks_2samp(real[col], gen[col])
        ks_results[col] = {'KS_stat': stat, 'p_value': p}
    return ks_results


def plot_generated_distributions():
    print("📊 Caricamento dati...")
    data = load_and_prepare()
    real = data[data['source'] == 'Real']
    gen = data[data['source'] == 'Generated']

    print("📈 Generazione grafici...")
    for col in ['duration', 'severity']:
        plot_distribution(data, col)
        plot_boxplot(data, col)

    plot_scatter(data)

    print("🧪 Test KS...")
    ks = run_ks_tests(real, gen)
    for col, res in ks.items():
        print(f"🧠 {col} → KS={res['KS_stat']:.4f} | p={res['p_value']:.4f}")

    print(f"✅ Visualizzazioni salvate in '{OUT_DIR}'")
