import wntr
import random

# === CONFIGURAZIONE ===
inp_file = "/Users/giovanni02/Desktop/Progetti/water-4.0/data/L-TOWN.inp"
leak_node_id = None      # Se None, verrà scelto casualmente
leak_start_hour = 2      # Ora di inizio perdita (in ore)
leak_duration_hr = 3     # Durata della perdita in ore
leak_flow_cmh = 1.0      # Severità perdita in m³/h (puoi testare altri valori)

# === CARICAMENTO MODELLO ===
print("🔄 Caricamento rete...")
wn = wntr.network.WaterNetworkModel(inp_file)

# === VERIFICA E SELEZIONE NODO ===
if leak_node_id is None:
    junctions = list(wn.junction_name_list)
    leak_node_id = random.choice(junctions)
    print(f"📍 Nodo selezionato per perdita: {leak_node_id}")
elif leak_node_id not in wn.node_name_list:
    raise ValueError(f"❌ Nodo {leak_node_id} non trovato nella rete.")

# === CREAZIONE DEL PATTERN TEMPORALE DELLA PERDITA ===
pattern_name = 'leak_pattern'
pattern_length = int(wn.options.time.duration / wn.options.time.hydraulic_timestep)
pattern = [0]*pattern_length

# Calcolo inizio e fine in step
start_step = int((leak_start_hour * 3600) / wn.options.time.hydraulic_timestep)
end_step = int(((leak_start_hour + leak_duration_hr) * 3600) / wn.options.time.hydraulic_timestep)

for i in range(start_step, min(end_step, len(pattern))):
    pattern[i] = 1.0

wn.add_pattern(pattern_name, pattern)

# === INIEZIONE DELLA PERDITA ===
leak_node = wn.get_node(leak_node_id)
leak_node.add_demand(base=leak_flow_cmh, pattern_name=pattern_name, category='LEAKAGE')
print(f"💧 Perdita iniettata su {leak_node_id}: {leak_flow_cmh} m³/h dalle {leak_start_hour}:00 per {leak_duration_hr}h")

# === CONFIGURAZIONE TEMPORALI ===
wn.options.time.hydraulic_timestep = 3600        # 1 ora
wn.options.time.report_timestep = 3600
wn.options.time.duration = 24 * 3600             # massimo 24 ore

# === SIMULAZIONE ===
print("▶️ Avvio simulazione...")
sim = wntr.sim.EpanetSimulator(wn)

try:
    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()
except Exception as e:
    print("❌ Errore durante la simulazione:", e)

# === ANALISI RISULTATI ===
pressure = results.node['pressure']
leak_pressures = pressure.loc[:, leak_node_id]

print("\n📊 Pressioni al nodo con perdita:")
print(leak_pressures.head(12))  # prime 12 ore

# Puoi salvare i risultati se vuoi:
leak_pressures.to_csv("leak_pressure_results.csv")

print("\n✅ Script terminato con successo.")
