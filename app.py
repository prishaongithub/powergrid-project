import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx
import matplotlib.pyplot as plt
from datetime import date as dt_date

# -------------------
# Constants (keep consistent with generator)
# -------------------
SUB_ID = "SUB"
SUB_X, SUB_Y = 50, 50

st.set_page_config(page_title="Tower Risk & Routing", layout="wide")

@st.cache_data
def load_data():
    df_meta = pd.read_csv("towers_metadata.csv", parse_dates=["last_maintenance"])
    df_edges = pd.read_csv("graph_edges.csv")
    return df_meta, df_edges

@st.cache_resource
def load_model():
    model = joblib.load("fault_prediction_model.pkl")
    feature_order = joblib.load("model_features.pkl")
    return model, feature_order

def seed_from_date(d: dt_date) -> int:
    # stable randomness per selected day
    return int(d.strftime("%Y%m%d"))

def simulate_daily_sensors(df_meta: pd.DataFrame, sim_date: dt_date) -> pd.DataFrame:
    """Generate random daily sensor values for all towers (stable for the date)."""
    rng = np.random.default_rng(seed_from_date(sim_date))
    n = len(df_meta)

    rain = rng.integers(0, 100, n)
    wind = rng.integers(0, 120, n)
    temp = rng.integers(-5, 45, n)
    freq = rng.uniform(49, 51, n)
    volt = rng.uniform(200, 240, n)
    overload = rng.integers(50, 150, n)

    df_dyn = pd.DataFrame({
        "tower_id": df_meta["tower_id"].values,
        "date": pd.to_datetime(sim_date),
        "rain_intensity": rain,
        "wind_speed": wind,
        "temperature": temp,
        "frequency": freq,
        "voltage": volt,
        "overload_current_pct": overload
    })
    # anomalies
    df_dyn["freq_anomaly"] = ((df_dyn["frequency"] < 49.5) | (df_dyn["frequency"] > 50.5)).astype(int)
    df_dyn["volt_anomaly"] = ((df_dyn["voltage"] < 210) | (df_dyn["voltage"] > 230)).astype(int)
    df_dyn["anomaly_chances"] = df_dyn["freq_anomaly"] + df_dyn["volt_anomaly"]

    # days since maintenance
    df_dyn = df_dyn.merge(df_meta[["tower_id", "last_maintenance"]], on="tower_id", how="left")
    df_dyn["days_since_maintenance"] = (df_dyn["date"] - df_dyn["last_maintenance"]).dt.days

    return df_dyn.drop(columns=["last_maintenance"])

def prepare_features(df_dyn: pd.DataFrame, df_meta: pd.DataFrame, feature_order: list) -> pd.DataFrame:
    """Merge dynamic + static features and align columns to training order."""
    df = df_dyn.merge(
        df_meta[[
            "tower_id","region","region_risk_score","tower_age_years",
            "corrosion_level","x","y","distance_from_substation"
        ]],
        on="tower_id", how="left"
    )
    # build feature matrix like training
    drop_cols = ["date", "tower_id"]  # last_maintenance not present here
    X = df.drop(columns=drop_cols)
    # one-hot region exactly as before
    X = pd.get_dummies(X, columns=["region"], drop_first=True)

    # align to training feature order
    for col in feature_order:
        if col not in X.columns:
            X[col] = 0
    # extra cols in X that weren't in training -> drop
    X = X[feature_order]
    return df, X

def build_graph(df_meta: pd.DataFrame, df_edges: pd.DataFrame) -> nx.Graph:
    """Build NetworkX graph with substation + towers + weighted edges."""
    G = nx.Graph()
    # add nodes with coords
    for _, r in df_meta.iterrows():
        G.add_node(r["tower_id"], x=r["x"], y=r["y"])
    G.add_node(SUB_ID, x=SUB_X, y=SUB_Y)

    # add weighted edges
    for _, e in df_edges.iterrows():
        G.add_edge(e["source"], e["target"], weight=float(e["distance"]))
    return G

def dijkstra_route(G: nx.Graph, start: str, targets: list[str]) -> tuple[list[str], float, list[str]]:
    """
    Greedy multi-stop route:
    Start at `start`, repeatedly go to the unvisited target with the
    shortest path distance (via Dijkstra). Returns:
      - expanded sequence of nodes including intermediate hops
      - total path length
      - ordered list of target towers actually visited
    """
    current = start
    remaining = set(targets)
    visited_order = []
    path_nodes = [start]
    total = 0.0

    while remaining:
        # choose closest next target by shortest-path distance
        best_t, best_len, best_path = None, float("inf"), None
        for t in remaining:
            try:
                sp_len = nx.dijkstra_path_length(G, current, t, weight="weight")
            except nx.NetworkXNoPath:
                continue
            if sp_len < best_len:
                best_len = sp_len
                best_t = t
        if best_t is None:
            # no reachable targets remain
            break
        best_path = nx.dijkstra_path(G, current, best_t, weight="weight")
        # extend nodes (avoid duplicating the first node)
        path_nodes += best_path[1:]
        total += best_len
        visited_order.append(best_t)
        remaining.remove(best_t)
        current = best_t

    return path_nodes, total, visited_order

def plot_network(df_meta, risky_set, route_nodes):
    fig, ax = plt.subplots(figsize=(7, 7))
    # plot towers
    ax.scatter(df_meta["x"], df_meta["y"], s=30, alpha=0.8)
    # annotate towers
    for _, r in df_meta.iterrows():
        ax.text(r["x"] + 0.8, r["y"] + 0.8, r["tower_id"], fontsize=8)

    # substation
    ax.scatter([SUB_X], [SUB_Y], s=120, marker="s")
    ax.text(SUB_X + 1, SUB_Y + 1, SUB_ID, fontsize=10)

    # highlight risky towers
    risky_coords = df_meta[df_meta["tower_id"].isin(risky_set)][["x","y"]].values
    if len(risky_coords) > 0:
        ax.scatter(risky_coords[:,0], risky_coords[:,1], s=120, marker="^")

    # draw route if provided
    if route_nodes and len(route_nodes) > 1:
        # map node->coords
        coord = {r["tower_id"]:(r["x"], r["y"]) for _, r in df_meta.iterrows()}
        coord[SUB_ID] = (SUB_X, SUB_Y)
        xs, ys = [], []
        for n in route_nodes:
            x, y = coord[n]
            xs.append(x); ys.append(y)
        ax.plot(xs, ys)

    ax.set_title("Network (Squares=substation, Triangles=Top Risky)")
    ax.set_xlim(-5, 105); ax.set_ylim(-5, 105)
    ax.set_aspect("equal", adjustable="box")
    st.pyplot(fig)

# =======================
# UI
# =======================
st.title("⚡ Tower Fault Risk & Dijkstra Routing")

df_meta, df_edges = load_data()
model, feature_order = load_model()

col1, col2, col3 = st.columns([1,1,1])
with col1:
    chosen_date = st.date_input("Select date", value=dt_date.today())
with col2:
    top_n = st.slider("Top N risky towers to visit", min_value=3, max_value=min(10, len(df_meta)), value=5)
with col3:
    st.caption("Random sensor data is simulated per day to mimic live inputs.")

# Simulate & predict
df_dyn = simulate_daily_sensors(df_meta, chosen_date)
df_joined, X = prepare_features(df_dyn, df_meta, feature_order)

# Predict probabilities
probs = model.predict_proba(X)[:, 1]
df_results = df_joined.copy()
df_results["risk_score"] = probs
df_results = df_results.sort_values("risk_score", ascending=False)

st.subheader("Predicted Risk (today)")
st.dataframe(df_results[[
    "tower_id","risk_score","region","region_risk_score",
    "tower_age_years","corrosion_level",
    "rain_intensity","wind_speed","temperature","frequency","voltage",
    "anomaly_chances","days_since_maintenance"
]].head(20).style.format({
    "risk_score": "{:.3f}",
    "frequency": "{:.2f}",
    "voltage": "{:.2f}"
}))

# pick risky set
risky_towers = df_results.head(top_n)["tower_id"].tolist()

# Build graph and route
G = build_graph(df_meta, df_edges)
route_nodes, total_len, visit_order = dijkstra_route(G, SUB_ID, risky_towers)

left, right = st.columns([1,1])
with left:
    st.subheader("Top Risky Towers")
    st.write(", ".join(risky_towers))

with right:
    st.subheader("Dijkstra Visit Order")
    if visit_order:
        st.write(" → ".join([SUB_ID] + visit_order))
        st.write(f"Total route length: **{total_len:.2f}**")
    else:
        st.write("No reachable towers (check graph connectivity).")

st.subheader("Map & Route")
plot_network(df_meta, set(risky_towers), route_nodes)

# Download predictions
csv_bytes = df_results.sort_values("risk_score", ascending=False).to_csv(index=False).encode("utf-8")
st.download_button(
    "Download today's predictions (CSV)",
    data=csv_bytes,
    file_name=f"predictions_{chosen_date}.csv",
    mime="text/csv"
)

st.caption("Route uses a greedy multi-stop strategy with Dijkstra shortest paths on a sparse road graph.")
