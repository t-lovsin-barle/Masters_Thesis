import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------
# 0) Setup
# -----------------------
np.random.seed(42)

# Sample point cloud (disk radius = 1)
n_points = 200
R = 1.0
r = R * np.sqrt(np.random.rand(n_points))
theta = 2 * np.pi * np.random.rand(n_points)
points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))

# Build δ-Rips graph
threshold = 0.25
G = nx.Graph()
for i, p in enumerate(points):
    G.add_node(i, pos=(p[0], p[1]))
for i in range(n_points):
    for j in range(i+1, n_points):
        if np.linalg.norm(points[i] - points[j]) <= threshold:
            G.add_edge(i, j)

degree = dict(G.degree())
nx.set_node_attributes(G, degree, "deg")
g_vals = {v: -d for v, d in degree.items()}  # g = -f

# -----------------------
# 1) Persistence (H0)
# -----------------------
use_gudhi = True
try:
    import gudhi as gd
except Exception:
    print("gudhi not available; using union-find.")
    use_gudhi = False

if use_gudhi:
    st = gd.SimplexTree()
    for v, gv in g_vals.items():
        st.insert([v], filtration=gv)
    for u, v in G.edges():
        st.insert([u, v], filtration=max(g_vals[u], g_vals[v]))
    st.initialize_filtration()
    persistence = st.persistence(homology_coeff_field=2, persistence_dim_max=True)
    dim0 = [p for p in persistence if p[0] == 0]
    births, deaths = [], []
    for _, (b, d) in dim0:
        births.append(b)
        if d == float("inf"):
            d = max(list(g_vals.values())) + 0.5
        deaths.append(d)
    births, deaths = np.array(births), np.array(deaths)
else:
    # union-find fallback
    events = []
    for v, gv in g_vals.items():
        events.append((gv, 0, 'v', (v,)))
    for u, v in G.edges():
        events.append((max(g_vals[u], g_vals[v]), 1, 'e', (u, v)))
    events.sort(key=lambda x: (x[0], x[1]))

    parent, rank, comp_birth = {}, {}, {}
    pers_pairs = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b, current_filt):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        ba, bb = comp_birth[ra], comp_birth[rb]
        if ba > bb:
            dead_rep, lived_rep = ra, rb
            dead_birth = ba
        else:
            dead_rep, lived_rep = rb, ra
            dead_birth = bb
        pers_pairs.append((dead_birth, current_filt))
        if rank[ra] < rank[rb]:
            parent[ra] = rb
            comp_birth[rb] = min(comp_birth[ra], comp_birth[rb])
        else:
            parent[rb] = ra
            comp_birth[ra] = min(comp_birth[ra], comp_birth[rb])
            if rank[ra] == rank[rb]:
                rank[ra] += 1

    for filt, _, etype, payload in events:
        if etype == 'v':
            v = payload[0]
            parent[v] = v
            rank[v] = 0
            comp_birth[v] = filt
        else:
            u, v = payload
            if u not in parent or v not in parent:
                continue
            union(u, v, filt)

    infinite_death_val = max([ev[0] for ev in events]) + 0.5
    reps = set(find(v) for v in G.nodes())
    for r in reps:
        pers_pairs.append((comp_birth[r], infinite_death_val))

    births = np.array([p[0] for p in pers_pairs])
    deaths = np.array([p[1] for p in pers_pairs])

# Flip back to positive coordinates for visualization
births_pos = -births
deaths_pos = -deaths

# -----------------------
# 2) Plots
# -----------------------

pos = {i: tuple(points[i]) for i in range(n_points)}

# (1) Point cloud
plt.figure(figsize=(6,6))
plt.scatter(points[:,0], points[:,1], s=30, color="blue")
plt.gca().set_aspect('equal')
plt.show()

# (2) Rips graph
plt.figure(figsize=(6,6))
nx.draw(G, pos=pos, with_labels=False, node_size=20, node_color="blue")
plt.gca().set_aspect('equal')
plt.title("Rips graph")
plt.show()

# (3) Filtration snapshots (decreasing threshold left -> right)
deg_values = np.array(list(degree.values()))
thresh_vals = np.unique(np.percentile(deg_values, [25, 50, 75, 100]).astype(int))[::-1]

xs, ys = points[:,0], points[:,1]
xlim = (xs.min()-0.1, xs.max()+0.1)
ylim = (ys.min()-0.1, ys.max()+0.1)

fig, axes = plt.subplots(1, len(thresh_vals), figsize=(5*len(thresh_vals), 5))
for ax, t in zip(axes, thresh_vals):
    selected_nodes = [v for v, d in degree.items() if d >= t]
    subG = G.subgraph(selected_nodes)
    nx.draw(subG, pos=pos, ax=ax, node_size=20, node_color="blue")
    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"f ≥ {t}")
plt.show()

# (4) Persistence diagram
plt.figure(figsize=(6,6))
plt.scatter(births_pos, deaths_pos, s=40, color="red")
lim = max(births_pos.max(), deaths_pos.max()) + 1
plt.plot([0, lim], [0, lim], 'k--')
plt.xlim(0, lim)
plt.ylim(0, lim)
plt.gca().set_aspect('equal')
plt.show()
