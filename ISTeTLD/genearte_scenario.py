import wntr
import random
import pandas as pd
import datetime
import os

# === CONFIGURAZIONE ===
INP_PATH = "/Users/giovanni02/Desktop/Progetti/water-4.0/data/L-TOWN.inp"
OUTPUT_PATH = "scenarios.csv"
NUM_SCENARIOS = 1000     # quante righe generare

LEAK_TYPES = ['orifice', 'valve', 'pipe']

def generate_random_time():
    hour = random.randint(0, 23)
    return f"{hour:02d}:00"

def generate_scenarios(inp_path, num_scenarios, output_path):
    print(f"📥 Caricamento rete da {inp_path}")
    wn = wntr.network.WaterNetworkModel(inp_path)
    nodes = list(wn.junction_name_list)

    scenarios = []

    for _ in range(num_scenarios):
        node = random.choice(nodes)
        start_time = generate_random_time()
        duration = random.randint(1, 6)
        severity = round(random.uniform(0.2, 2.5), 2)
        leak_type = random.choice(LEAK_TYPES)

        # Timestamp fittizio
        fake_date = datetime.datetime(
            year=2018,
            month=random.randint(1, 12),
            day=random.randint(1, 28)  # evitiamo problemi con febbraio
        )
        weekday = fake_date.strftime('%A')  # Monday, Tuesday...
        month = fake_date.strftime('%B')    # January, February...

        scenario = {
            "node_id": node,
            "start_time": start_time,
            "duration": duration,
            "severity": severity,
            "leak_type": leak_type,
            "weekday": weekday,
            "month": month
        }

        scenarios.append(scenario)

    df = pd.DataFrame(scenarios)
    df.to_csv(output_path, index=False)
    print(f"✅ {num_scenarios} scenari salvati in {output_path}")

if __name__ == "__main__":
    generate_scenarios(INP_PATH, NUM_SCENARIOS, OUTPUT_PATH)
