import wntr
import wntr.network
import networkx as nx

# Carica il modello
wn = wntr.network.WaterNetworkModel('/Users/giovanni02/Desktop/Progetti/water-4.0/data/L-TOWN_Real.inp')

# Informazioni base
print(f"Numero nodi: {len(wn.junction_name_list)}")
print(f"Numero serbatoi: {len(wn.tank_name_list)}")
print(f"Numero condotte: {len(wn.pipe_name_list)}")

# Individua i nodi isolati (senza connessioni)
G = wn.get_graph()
isolated_nodes = list(nx.isolates(G))

if isolated_nodes:
    print(f"Nodi isolati trovati: {isolated_nodes}")
else:
    print("Nessun nodo isolato trovato.")