import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
import streamlit.components.v1 as components
import plotly.graph_objects as go
import osmnx as ox
import networkx as nx
from folium.features import RegularPolygonMarker
from branca.element import Element


# =========================================
# CONFIGURACIÓN GENERAL DEL DASHBOARD
# =========================================
st.set_page_config(
    page_title="DASHBOARD DISEÑO CCTV — Topologías de Fibra",
    layout="wide"
)

st.title("DASHBOARD DISEÑO CCTV")
st.warning(
    "VERSIÓN FINAL 12 — Sin UNIFILAR — Mapa funcionando",
    icon="⚙️"
)
st.caption("Visualización didáctica de topologías: Punto a Punto, Anillo y FTTN")

st.markdown("""
Este tablero está pensado para el curso de **Diseño CCTV**, 
comparando tres modelos de implementación de fibra óptica:
- 🔹 Punto a Punto  
- 🔹 Topología en Anillo  
- 🔹 Distribución FTTN (Fibra hasta el Nodo)
""")

st.markdown("---")

# Color base FICOM para fibras
FICOM_COLOR = "#4FB4CA"

# Paleta estándar código de colores FO
FIBER_STANDARD_COLORS = [
    ("Fibra 1", "Azul",   "#0000FF"),
    ("Fibra 2", "Naranja", "#FF7F00"),
    ("Fibra 3", "Verde",   "#008000"),
]

PATCHCORD_COLOR = "#FFD700"  # amarillo patchcord


# =========================================
# FUNCIÓN DIAGRAMA LÓGICO GENERAL (PLOTLY)
# =========================================
def create_topology_diagram(topology: str) -> go.Figure:
    topo = topology.lower()

    # -------------------------------
    # PUNTO A PUNTO
    # -------------------------------
    if topo == "p2p":
        fig = go.Figure()

        core_x, core_y = -0.9, 0.5
        fig.add_trace(go.Scatter(
            x=[core_x],
            y=[core_y],
            mode="markers+text",
            marker=dict(size=22, symbol="circle", color="red"),
            text=["CORE / NVR"],
            textposition="bottom center",
            showlegend=False
        ))

        sw_core_x, sw_core_y = -0.3, 0.5
        fig.add_trace(go.Scatter(
            x=[sw_core_x],
            y=[sw_core_y],
            mode="markers+text",
            marker=dict(size=20, symbol="square", color="orange"),
            text=["TR01-SW00-DC-8P"],
            textposition="bottom center",
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[core_x, sw_core_x],
            y=[core_y, sw_core_y],
            mode="lines",
            line=dict(width=3, color=FICOM_COLOR),
            showlegend=False
        ))

        field_switches = [
            {"name": "TR01-SW01\nND01", "x": 0.3, "y": 0.8},
            {"name": "TR01-SW02\nND02", "x": 0.3, "y": 0.5},
            {"name": "TR01-SW03\nND03", "x": 0.3, "y": 0.2},
        ]

        cam_index = 1

        for fs in field_switches:
            sx, sy = fs["x"], fs["y"]

            fig.add_trace(go.Scatter(
                x=[sw_core_x, sx],
                y=[sw_core_y, sy],
                mode="lines",
                line=dict(width=2, color=FICOM_COLOR),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=[sx],
                y=[sy],
                mode="markers+text",
                marker=dict(size=18, symbol="square", color="green"),
                text=[fs["name"]],
                textposition="bottom center",
                showlegend=False
            ))

            cam_positions = [
                (sx + 0.35, sy + 0.12),
                (sx + 0.35, sy - 0.12),
            ]
            for (cx, cy) in cam_positions:
                fig.add_trace(go.Scatter(
                    x=[sx, cx],
                    y=[sy, cy],
                    mode="lines",
                    line=dict(width=1.8, dash="dot", color="gray"),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="markers+text",
                    marker=dict(size=12, symbol="triangle-up", color="black"),
                    text=[f"Cam {cam_index}"],
                    textposition="top center",
                    showlegend=False
                ))
                cam_index += 1

        fig.update_layout(
            title="Topología Punto a Punto (CORE → TR01-SW00-DC-8P → Sw de campo → Cámaras)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=60, b=20),
            height=400
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    # -------------------------------
    # ANILLO
    # -------------------------------
    if topo == "ring":
        n = 6
        radius = 0.6
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        switch_x = radius * np.cos(angles)
        switch_y = radius * np.sin(angles) + 0.1

        fig = go.Figure()

        for i in range(n):
            x0, y0 = switch_x[i], switch_y[i]
            x1, y1 = switch_x[(i + 1) % n], switch_y[(i + 1) % n]
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=2, color=FICOM_COLOR),
                showlegend=False
            ))

        fig.add_trace(go.Scatter(
            x=switch_x,
            y=switch_y,
            mode="markers+text",
            marker=dict(size=16, symbol="square", color="royalblue"),
            text=[f"TR01-SW0{i+1}" for i in range(n)],
            textposition="top center",
            showlegend=False
        ))

        cam_offset = 0.25
        for i in range(n):
            sx, sy = switch_x[i], switch_y[i]
            cx = sx * (1 + cam_offset)
            cy = sy * (1 + cam_offset)
            fig.add_trace(go.Scatter(
                x=[sx, cx],
                y=[sy, cy],
                mode="lines",
                line=dict(width=1.5, color="gray"),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers+text",
                marker=dict(size=12, symbol="triangle-up", color="black"),
                text=[f"Cam {i+1}"],
                textposition="top center",
                showlegend=False
            ))

        fig.update_layout(
            title="Topología en Anillo",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=60, b=20),
            height=400
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    # -------------------------------
    # FTTN
    # -------------------------------
    if topo == "fttn":
        fig = go.Figure()

        core_x, core_y = -0.9, 0.5
        fig.add_trace(go.Scatter(
            x=[core_x],
            y=[core_y],
            mode="markers+text",
            marker=dict(size=18, symbol="circle", color="red"),
            text=["CORE / NVR"],
            textposition="bottom center",
            showlegend=False
        ))

        node_x, node_y = -0.3, 0.5
        fig.add_trace(go.Scatter(
            x=[node_x],
            y=[node_y],
            mode="markers+text",
            marker=dict(size=18, symbol="square", color="royalblue"),
            text=["Nodo FTTN\n(FOSC+ONU)"],
            textposition="bottom center",
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[core_x, node_x],
            y=[core_y, node_y],
            mode="lines",
            line=dict(width=3, color=FICOM_COLOR),
            showlegend=False
        ))

        sw_positions = [
            (0.3, 0.8),
            (0.3, 0.5),
            (0.3, 0.2),
        ]

        for i, (sx, sy) in enumerate(sw_positions, start=1):
            fig.add_trace(go.Scatter(
                x=[node_x, sx],
                y=[node_y, sy],
                mode="lines",
                line=dict(width=2, color=FICOM_COLOR),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[sx],
                y=[sy],
                mode="markers+text",
                marker=dict(size=16, symbol="square", color="green"),
                text=[f"TR01-SW0{i+1}-ND0{i}"],
                textposition="bottom center",
                showlegend=False
            ))
            cam1 = (sx + 0.3, sy + 0.15)
            cam2 = (sx + 0.3, sy - 0.15)
            for j, (cx, cy) in enumerate([cam1, cam2], start=1):
                fig.add_trace(go.Scatter(
                    x=[sx, cx],
                    y=[sy, cy],
                    mode="lines",
                    line=dict(width=1.5, dash="dot", color="gray"),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="markers+text",
                    marker=dict(size=12, symbol="triangle-up", color="black"),
                    text=[f"Cam {i}.{j}"],
                    textposition="top center",
                    showlegend=False
                ))

        fig.update_layout(
            title="Topología FTTN (Fibra hasta el Nodo)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=60, b=20),
            height=400
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    return go.Figure()


# =========================================================
# MAPA EJEMPLO REAL — MENDOZA (P2P, OSMnx + Folium)
# =========================================================
def build_mendoza_p2p_map_osmnx(random_key: int = 0):
    rng = np.random.default_rng(random_key)

    center_lat = -32.8895
    center_lon = -68.8458

    G = ox.graph_from_point(
        (center_lat, center_lon),
        dist=2000,
        network_type="drive"
    )
    G_work = G.copy()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB dark_matter",
    )

    nodes = [
        {
            "name": "CORE / NVR",
            "type": "CORE",
            "lat": center_lat + 0.0003,
            "lon": center_lon - 0.0003,
            "descripcion": "Grabador / NVR en Datacenter",
        },
        {
            "name": "TR01-SW00-DC-8P",
            "type": "SW_CORE",
            "lat": center_lat,
            "lon": center_lon,
            "descripcion": "Switch de 8 puertos ópticos en Sala Técnica (troncal TR01)",
        },
    ]

    field_nodes = []
    for i in range(3):
        dlat = rng.uniform(-0.006, 0.006)
        dlon = rng.uniform(-0.006, 0.006)
        field_nodes.append({
            "name": f"TR01-SW0{i+1}-ND0{i+1}",
            "type": "SW_CAMPO",
            "lat": center_lat + dlat,
            "lon": center_lon + dlon,
            "descripcion": f"Switch de campo Nodo 0{i+1} (troncal TR01)",
        })
    nodes.extend(field_nodes)

    df_nodes = pd.DataFrame(nodes)

    df_nodes["mid_lat"] = df_nodes["lat"]
    df_nodes["mid_lon"] = df_nodes["lon"]
    df_nodes["corner1_lat"] = np.nan
    df_nodes["corner1_lon"] = np.nan
    df_nodes["corner2_lat"] = np.nan
    df_nodes["corner2_lon"] = np.nan

    for idx, row in df_nodes[df_nodes["type"] == "SW_CAMPO"].iterrows():
        u, v, key = ox.distance.nearest_edges(G, X=row["lon"], Y=row["lat"])
        y_u, x_u = G.nodes[u]["y"], G.nodes[u]["x"]
        y_v, x_v = G.nodes[v]["y"], G.nodes[v]["x"]

        mid_lat = (y_u + y_v) / 2
        mid_lon = (x_u + x_v) / 2

        df_nodes.loc[idx, "mid_lat"] = mid_lat
        df_nodes.loc[idx, "mid_lon"] = mid_lon
        df_nodes.loc[idx, "corner1_lat"] = y_u
        df_nodes.loc[idx, "corner1_lon"] = x_u
        df_nodes.loc[idx, "corner2_lat"] = y_v
        df_nodes.loc[idx, "corner2_lon"] = x_v

    dc_delta_lat = 0.0009
    dc_delta_lon = 0.00108
    folium.Rectangle(
        bounds=[
            [center_lat - dc_delta_lat, center_lon - dc_delta_lon],
            [center_lat + dc_delta_lat, center_lon + dc_delta_lon],
        ],
        color="white",
        weight=2,
        dash_array="5,5",
        fill=False,
        tooltip="DATACENTER (CORE / NVR + TR01-SW00-DC-8P)",
    ).add_to(m)

    def node_color(t):
        if t == "CORE":
            return "red"
        if t == "SW_CORE":
            return "orange"
        if t == "SW_CAMPO":
            return "lime"
        return "white"

    for _, row in df_nodes.iterrows():
        if row["type"] == "CORE":
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=8,
                color=node_color(row["type"]),
                fill=True,
                fill_color=node_color(row["type"]),
                fill_opacity=0.9,
                popup=f"{row['name']}<br>{row['descripcion']}",
                tooltip=row["name"],
            ).add_to(m)
        elif row["type"] == "SW_CORE":
            RegularPolygonMarker(
                location=[row["lat"], row["lon"]],
                number_of_sides=4,
                radius=9,
                color=node_color(row["type"]),
                fill=True,
                fill_color=node_color(row["type"]),
                fill_opacity=0.9,
                popup=f"{row['name']}<br>{row['descripcion']}",
                tooltip=row["name"],
            ).add_to(m)
        elif row["type"] == "SW_CAMPO":
            RegularPolygonMarker(
                location=[row["mid_lat"], row["mid_lon"]],
                number_of_sides=4,
                radius=9,
                color=node_color(row["type"]),
                fill=True,
                fill_color=node_color(row["type"]),
                fill_opacity=0.9,
                popup=f"{row['name']}<br>{row['descripcion']}",
                tooltip=row["name"],
            ).add_to(m)

    troncal_info = []

    def add_route_by_street(
        map_obj,
        G_work,
        G_full,
        lat0,
        lon0,
        lat1,
        lon1,
        label: str,
        tooltip: str,
        color: str,
    ):
        def _shortest_path(G_used):
            orig_node = ox.distance.nearest_nodes(G_used, X=lon0, Y=lat0)
            dest_node = ox.distance.nearest_nodes(G_used, X=lon1, Y=lat1)
            return nx.shortest_path(G_used, orig_node, dest_node, weight="length")

        route = None
        use_fallback = False

        try:
            route = _shortest_path(G_work)
        except nx.NetworkXNoPath:
            try:
                route = _shortest_path(G_full)
                use_fallback = True
            except nx.NetworkXNoPath:
                return

        total_len = 0.0
        for u, v in zip(route, route[1:]):
            data = G_full.get_edge_data(u, v)
            if data is None:
                data = G_full.get_edge_data(v, u)
            if data:
                edge_attr = list(data.values())[0]
                total_len += edge_attr.get("length", 0.0)

        route_coords = [(G_full.nodes[n]["y"], G_full.nodes[n]["x"]) for n in route]
        route_coords.insert(0, (lat0, lon0))
        route_coords.append((lat1, lon1))

        folium.PolyLine(
            locations=route_coords,
            color=color,
            weight=3,
            tooltip=tooltip,
        ).add_to(map_obj)

        troncal_info.append({
            "name": label,
            "distance_m": total_len
        })

        if not use_fallback:
            for u, v in zip(route, route[1:]):
                if G_work.has_edge(u, v):
                    G_work.remove_edge(u, v)
                if G_work.has_edge(v, u):
                    G_work.remove_edge(v, u)

    core = df_nodes[df_nodes["type"] == "CORE"].iloc[0]
    sw_core = df_nodes[df_nodes["type"] == "SW_CORE"].iloc[0]
    sw_campo = df_nodes[df_nodes["type"] == "SW_CAMPO"]

    folium.PolyLine(
        locations=[[core["lat"], core["lon"]], [sw_core["lat"], sw_core["lon"]]],
        color=FICOM_COLOR,
        weight=4,
        tooltip="FO CORE / NVR → TR01-SW00-DC-8P",
    ).add_to(m)

    route_colors = [c[2] for c in FIBER_STANDARD_COLORS]

    for i, (_, row) in enumerate(sw_campo.iterrows()):
        color = route_colors[i % len(route_colors)]
        add_route_by_street(
            m,
            G_work,
            G,
            sw_core["lat"],
            sw_core["lon"],
            row["mid_lat"],
            row["mid_lon"],
            label=row["name"],
            tooltip=f"FO {sw_core['name']} → {row['name']}",
            color=color,
        )

    for _, row in sw_campo.iterrows():
        mid_lat = row["mid_lat"]
        mid_lon = row["mid_lon"]
        c1_lat, c1_lon = row["corner1_lat"], row["corner1_lon"]
        c2_lat, c2_lon = row["corner2_lat"], row["corner2_lon"]

        folium.PolyLine(
            locations=[[mid_lat, mid_lon], [c1_lat, c1_lon]],
            color="white",
            weight=1,
            dash_array="4,4",
            tooltip=f"UTP desde {row['name']} a Cámara 1 (≤ 100 m aprox.)",
        ).add_to(m)

        RegularPolygonMarker(
            location=[c1_lat, c1_lon],
            number_of_sides=3,
            radius=7,
            rotation=0,
            color="yellow",
            fill=True,
            fill_color="yellow",
            fill_opacity=0.9,
            tooltip=f"Cámara IP 1 asociada a {row['name']}",
        ).add_to(m)

        folium.PolyLine(
            locations=[[mid_lat, mid_lon], [c2_lat, c2_lon]],
            color="white",
            weight=1,
            dash_array="4,4",
            tooltip=f"UTP desde {row['name']} a Cámara 2 (≤ 100 m aprox.)",
        ).add_to(m)

        RegularPolygonMarker(
            location=[c2_lat, c2_lon],
            number_of_sides=3,
            radius=7,
            rotation=0,
            color="yellow",
            fill=True,
            fill_color="yellow",
            fill_opacity=0.9,
            tooltip=f"Cámara IP 2 asociada a {row['name']}",
        ).add_to(m)

    legend_html = f"""
    <div style="
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        background-color: rgba(0, 0, 0, 0.75);
        padding: 10px 14px;
        border-radius: 8px;
        color: #ffffff;
        font-size: 11px;
        max-width: 260px;
    ">
      <div style="font-weight: bold; margin-bottom: 4px;">Referencias</div>
      <div style="margin-bottom: 4px;">
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:red;margin-right:4px;"></span>
        CORE / NVR
      </div>
      <div style="margin-bottom: 4px;">
        <span style="display:inline-block;width:10px;height:10px;background:orange;margin-right:4px;"></span>
        Sw troncal TR01-SW00-DC-8P
      </div>
      <div style="margin-bottom: 4px;">
        <span style="display:inline-block;width:10px;height:10px;background:lime;margin-right:4px;"></span>
        Sw de campo TR01-SW0x-ND0x
      </div>
      <div style="margin-bottom: 4px;">
        <span style="display:inline-block;width:22px;border-bottom:3px solid {FICOM_COLOR};margin-right:4px;"></span>
        FO CORE → Sw 8P
      </div>
      <div style="margin-bottom: 4px;">
        <span style="display:inline-block;width:22px;border-bottom:3px solid {route_colors[0]};margin-right:4px;"></span>
        FO Fibra 1 (Azul) → Sw campo
      </div>
      <div style="margin-bottom: 4px;">
        <span style="display:inline-block;width:22px;border-bottom:3px solid {route_colors[1]};margin-right:4px;"></span>
        FO Fibra 2 (Naranja) → Sw campo
      </div>
      <div style="margin-bottom: 4px;">
        <span style="display:inline-block;width:22px;border-bottom:3px solid {route_colors[2]};margin-right:4px;"></span>
        FO Fibra 3 (Verde) → Sw campo
      </div>
      <div style="margin-bottom: 4px;">
        <span style="
            display:inline-block;
            width: 0;
            height: 0;
            border-left: 10px solid yellow;
            border-top: 5px solid transparent;
            border-bottom: 5px solid transparent;
            margin-right:4px;
        "></span>
        Cámara IP
      </div>
      <hr style="border:0;border-top:1px solid #777;margin:4px 0;">
      <div style="font-weight: bold; margin-bottom: 2px;">Longitud troncales</div>
    """

    if troncal_info:
        for info in troncal_info:
            dist_m = info["distance_m"]
            legend_html += f"<div>{info['name']}: {dist_m:,.0f} m</div>"
    else:
        legend_html += "<div>Sin datos de troncales</div>"

    legend_html += "</div>"

    legend = Element(legend_html)
    m.get_root().html.add_child(legend)

    return m, troncal_info



# =========================================
# TABS PRINCIPALES
# =========================================
tab_p2p, tab_ring, tab_fttn, tab_comp = st.tabs(
    ["🔌 Punto a Punto", "⭕ Anillo", "🌿 FTTN (CCTV-IP)", "📊 Comparativo Global"]
)



# ============================================================
# FUNCIONES AUXILIARES — RECTÁNGULOS, ODF, SWITCH, CABLES
# ============================================================

def _add_rectangle(fig, x0, y0, x1, y1, label, color="#FFFFFF", line_color="#000000"):
    fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                  line=dict(color=line_color, width=1),
                  fillcolor=color)
    fig.add_annotation(x=(x0+x1)/2, y=y1+0.01, text=label, showarrow=False, font=dict(size=10))

def _add_odf(fig, x0, y0, h, label):
    w = 0.18
    x1 = x0 + w
    y1 = y0 + h

    fig.add_shape(
        type="rect",
        x0=x0, y0=y0, x1=x1, y1=y1,
        line=dict(color="black", width=1),
        fillcolor="#C8F7C5"
    )

    fig.add_annotation(
        x=(x0+x1)/2,
        y=y1 - 0.015,
        text=label,
        font=dict(size=10),
        showarrow=False
    )

    # Puertos: 6 puertos (solo P1 se usa)
    ports = []
    px = x0 + 0.02
    py = y0 + h*0.25
    for i in range(6):
        fig.add_shape(
            type="rect",
            x0=px, y0=py,
            x1=px+0.018, y1=py+0.03,
            line=dict(color="black", width=1),
            fillcolor="white"
        )
        ports.append((px+0.009, py+0.015))
        px += 0.026

    return ports
    
def _straight_cable(a, b):
    """Devuelve 3 segmentos horizontales/verticales, siempre visibles."""
    x0, y0 = a
    x1, y1 = b
    xm = (x0 + x1) / 2

    return [
        ((x0, y0), (xm, y0)),
        ((xm, y0), (xm, y1)),
        ((xm, y1), (x1, y1)),
    ]

def _add_switch(fig, x0, y0, width=0.18, height=0.14):
    x1 = x0 + width
    y1 = y0 + height

    fig.add_shape(
        type="rect",
        x0=x0, y0=y0, x1=x1, y1=y1,
        fillcolor="black",
        line=dict(color="white", width=1)
    )

    fig.add_annotation(
        x=(x0+x1)/2, y=y1 - 0.02,
        text="SW8P-CORE-NVR — 8×SFP",
        font=dict(size=10, color="white"),
        showarrow=False
    )

    ports = []
    px = x0 + 0.015
    pw = width/12
    py0 = y0 + 0.02
    py1 = py0 + 0.03

    for i in range(8):
        fig.add_shape(
            type="rect",
            x0=px, y0=py0, x1=px+pw, y1=py1,
            fillcolor="gray", line=dict(color="white")
        )
        ports.append((px+pw/2, (py0+py1)/2))
        px += pw + 0.01

    return ports

def _add_cable_curve(fig, p1, p2, width, color):
    (x1, y1) = p1
    (x2, y2) = p2
    r = 0.03

    # Punto medio
    xm = (x1 + x2)/2

    path = f"M {x1},{y1} Q {xm},{y1+r} {xm},{(y1+y2)/2} T {x2},{y2}"

    fig.add_shape(
        type="path",
        path=path,
        line=dict(color=color, width=width)
    )
    
def _add_cable_segments(fig, segments, width=3, color="#E3D873"):
    trace_ids = []
    for (a, b) in segments:
        t = fig.add_trace(go.Scatter(
            x=[a[0], b[0]],
            y=[a[1], b[1]],
            mode="lines",
            line=dict(color=color, width=width),
            showlegend=False
        ))
        trace_ids.append(len(fig.data)-1)
    return trace_ids


def _build_all_cables(fig):
    odf_tr = fig._odf_troncal
    odf_int = fig._odf_interconexion
    odf_core_int = fig._odf_core_int
    odf_core = fig._odf_core
    sfp = fig._sfp_ports

    # Patchcord TRONCAL → INTERCONEXIÓN (grueso)
    for tr, it in zip(odf_tr, odf_int):
        _add_cable_curve(fig, tr[0], it[0], width=4, color="#FFD700")

    # Patchcords TRONCAL → CORE-NVR (INT)
    _add_cable_curve(fig, odf_int[0][0], odf_core_int[0], 2, "#FFD700")
    _add_cable_curve(fig, odf_int[1][0], odf_core_int[1], 2, "#FFD700")
    _add_cable_curve(fig, odf_int[2][0], odf_core_int[2], 2, "#FFD700")

    # Cable grueso entre CORE-NVR (INT) → CORE-NVR (Rack CORE)
    _add_cable_curve(fig, odf_core_int[0], odf_core[0], width=6, color="#FFD700")

    # CORE → SFP Switch
    _add_cable_curve(fig, odf_core[0], sfp[0], 2, "#FFD700")
    _add_cable_curve(fig, odf_core[1], sfp[1], 2, "#FFD700")
    _add_cable_curve(fig, odf_core[2], sfp[2], 2, "#FFD700")

    # ======================================================
    # 1) TRONCAL → INTERCONEXIÓN (cables gruesos)
    # ======================================================
    fig._cables["troncal_1"] = {
        "segments": _straight_cable(p_tr1, p_int1),
        "trace_ids": _add_cable_segments(fig, _straight_cable(p_tr1, p_int1), width=5)
    }
    fig._cables["troncal_2"] = {
        "segments": _straight_cable(p_tr2, p_int2),
        "trace_ids": _add_cable_segments(fig, _straight_cable(p_tr2, p_int2), width=5)
    }
    fig._cables["troncal_3"] = {
        "segments": _straight_cable(p_tr3, p_int3),
        "trace_ids": _add_cable_segments(fig, _straight_cable(p_tr3, p_int3), width=5)
    }

    # ======================================================
    # 2) PATCHCORDS TRONCAL → ODF CORE-NVR (INT)
    # ======================================================
    fig._cables["patch1_tr_to_coreint"] = {
        "segments": [(p_tr1, p_core_int_1)],
        "trace_ids": _add_cable_segments(fig, [(p_tr1, p_core_int_1)], width=2)
    }
    fig._cables["patch2_tr_to_coreint"] = {
        "segments": [(p_tr2, p_core_int_2)],
        "trace_ids": _add_cable_segments(fig, [(p_tr2, p_core_int_2)], width=2)
    }
    fig._cables["patch3_tr_to_coreint"] = {
        "segments": [(p_tr3, p_core_int_3)],
        "trace_ids": _add_cable_segments(fig, [(p_tr3, p_core_int_3)], width=2)
    }

    # ======================================================
    # 3) CABLE GRUESO ENTRE ODF CORE INT → ODF CORE FINAL
    # ======================================================
    fig._cables["coreint_to_core"] = {
        "segments": _straight_cable(p_core_int_1, p_core_1),
        "trace_ids": _add_cable_segments(fig, _straight_cable(p_core_int_1, p_core_1), width=5)
    }

    # ======================================================
    # 4) CORE FINAL → SWITCH
    # ======================================================
    fig._cables["core1_to_sfp1"] = {
        "segments": [(p_core_1, sfp1)],
        "trace_ids": _add_cable_segments(fig, [(p_core_1, sfp1)], width=2)
    }
    fig._cables["core2_to_sfp2"] = {
        "segments": [(p_core_2, sfp2)],
        "trace_ids": _add_cable_segments(fig, [(p_core_2, sfp2)], width=2)
    }
    fig._cables["core3_to_sfp3"] = {
        "segments": [(p_core_3, sfp3)],
        "trace_ids": _add_cable_segments(fig, [(p_core_3, sfp3)], width=2)
    }

def _build_animation(fig):
    frames = []
    keys = list(fig._cables.keys())

    highlight = "#FFD700"
    normal = "#E3D873"

    for step, key in enumerate(keys):
        frame_data = []

        for k2 in keys:
            for tr in fig._cables[k2]["trace_ids"]:
                frame_data.append({"line": {"color": highlight if k2 == key else normal,
                                            "width": 4 if "troncal" in k2 else 2}})

        frames.append(go.Frame(data=frame_data, name=f"step_{step}"))

    fig.frames = frames


# ============================================================
# FUNCIÓN PRINCIPAL: DIAGRAMA COMPLETO
# ============================================================

# =========================================================
# NUEVO DIAGRAMA DE RACKS — TRONCAL / INTERCONEXIÓN / CORE
# =========================================================

def create_rack_connection_diagram():
    fig = go.Figure()

    # ----------------------------------------
    # DIMENSIONES BASE
    # ----------------------------------------
    rack_w = 0.22
    rack_h = 0.72   # Aumentado para que entren mejor los ODF
    odf_h = 0.12    # ALTURA DOBLE
    odf_gap = 0.10

    rack_troncal_x0 = 0.05
    rack_int_x0     = 0.38
    rack_core_x0    = 0.70
    rack_y0 = 0.10

    # ========================================
    # RACK TRONCAL
    # ========================================
    fig.add_shape(
        type="rect",
        x0=rack_troncal_x0, y0=rack_y0,
        x1=rack_troncal_x0 + rack_w,
        y1=rack_y0 + rack_h,
        line=dict(color="black", width=2),
        fillcolor="#F7F7F7"
    )
    fig.add_annotation(
        x=rack_troncal_x0 + rack_w/2,
        y=rack_y0 + rack_h + 0.03,
        text="RACK TRONCAL",
        showarrow=False,
        font=dict(size=12)
    )

    # ========================================
    # RACK INTERCONEXIÓN
    # ========================================
    fig.add_shape(
        type="rect",
        x0=rack_int_x0, y0=rack_y0,
        x1=rack_int_x0 + rack_w,
        y1=rack_y0 + rack_h,
        line=dict(color="black", width=2),
        fillcolor="#F7F7F7"
    )
    fig.add_annotation(
        x=rack_int_x0 + rack_w/2,
        y=rack_y0 + rack_h + 0.03,
        text="RACK DE INTERCONEXIÓN",
        showarrow=False,
        font=dict(size=12)
    )

    # ========================================
    # RACK CORE / NVR
    # ========================================
    fig.add_shape(
        type="rect",
        x0=rack_core_x0, y0=rack_y0,
        x1=rack_core_x0 + rack_w,
        y1=rack_y0 + rack_h,
        line=dict(color="black", width=2),
        fillcolor="#F7F7F7"
    )
    fig.add_annotation(
        x=rack_core_x0 + rack_w/2,
        y=rack_y0 + rack_h + 0.03,
        text="RACK CORE / NVR",
        showarrow=False,
        font=dict(size=12)
    )

    # ========================================
    # POSICIONES ODF (ALTURA DOBLE)
    # ========================================
    odf1_y = rack_y0 + rack_h - (odf_h + odf_gap)*1
    odf2_y = rack_y0 + rack_h - (odf_h + odf_gap)*2
    odf3_y = rack_y0 + rack_h - (odf_h + odf_gap)*3

    # ODF TRONCALES
    odf1_tr = _add_odf(fig, rack_troncal_x0+0.02, odf1_y, odf_h, "ODF TRONCAL 1")
    odf2_tr = _add_odf(fig, rack_troncal_x0+0.02, odf2_y, odf_h, "ODF TRONCAL 2")
    odf3_tr = _add_odf(fig, rack_troncal_x0+0.02, odf3_y, odf_h, "ODF TRONCAL 3")

    # ODF RACK INTERCONEXIÓN
    odf1_int = _add_odf(fig, rack_int_x0+0.02, odf1_y, odf_h, "ODF TRONCAL 1")
    odf2_int = _add_odf(fig, rack_int_x0+0.02, odf2_y, odf_h, "ODF TRONCAL 2")
    odf3_int = _add_odf(fig, rack_int_x0+0.02, odf3_y, odf_h, "ODF TRONCAL 3")

    # NUEVO ODF CORE–NVR (INT)
    odf_core_int_y = odf3_y - 0.14
    odf_core_int = _add_odf(fig, rack_int_x0+0.02, odf_core_int_y, odf_h, "ODF CORE–NVR (INT)")

    # ODF CORE / NVR del rack CORE
    odf_core = _add_odf(fig, rack_core_x0+0.02, odf1_y, odf_h, "ODF CORE–NVR")

    # Guardar para cableado
    fig._odf_troncal       = [odf1_tr, odf2_tr, odf3_tr]
    fig._odf_interconexion = [odf1_int, odf2_int, odf3_int]
    fig._odf_core_int      = odf_core_int
    fig._odf_core          = odf_core

    # ========================================
    # SWITCH (ALTURA DOBLE)
    # ========================================
    sw_x = rack_core_x0 + 0.02
    sw_y = odf1_y - 0.16

    fig._sfp_ports = _add_switch(fig, sw_x, sw_y, width=0.18, height=0.14)

    # ========================================
    # CABLEADO COMPLETO
    # ========================================
    _build_all_cables(fig)

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=700, margin=dict(l=20,r=20,t=20,b=20))

    return fig

# =========================================================
# TAB 1 — PUNTO A PUNTO
# =========================================================
with tab_p2p:
    st.subheader("Topología Punto a Punto")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("### Esquema lógico P2P (con switches de campo)")
            st.info(
                "CORE / NVR → TR01-SW00-DC-8P → Fibra a switches de campo "
                "(TR01-SW01-ND01 / ND02 / ND03) con 1 entrada óptica y varias salidas eléctricas → Cámaras por UTP."
            )
            st.markdown("**Flujo básico:**")
            st.markdown("- El CORE concentra el grabador / NVR y routing principal.")
            st.markdown("- `TR01-SW00-DC-8P` distribuye la troncal óptica de la troncal TR01.")
            st.markdown("- Cada puerto óptico alimenta un **switch de campo** (TR01-SW01-ND01 / SW02-ND02 / SW03-ND03).")
            st.markdown("- Desde cada switch de campo salen **2 o más cámaras** por UTP (últimos 100 m).")

            fig_p2p = create_topology_diagram("p2p")
            st.plotly_chart(fig_p2p, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown("### Indicadores P2P (ejemplo)")
            st.metric("Total de cámaras (ejemplo)", 12)
            st.metric("Puertos ópticos en CORE (Sw 8P)", 8)
            st.metric("Switches de campo (TR01-SW01/02/03)", 3)

            st.markdown("#### Ventajas / Desventajas")
            st.success("✔ Arquitectura intuitiva (CORE → Sw troncal → Sw de campo).")
            st.success("✔ Permite agrupar varias cámaras por cada punto de FO (Sw campo).")
            st.warning("✖ Sigue consumiendo varios puertos ópticos en el Sw troncal.")
            st.warning("✖ Si falla un switch de campo, caen todas las cámaras de ese punto.")

    st.markdown("---")

    with st.container(border=True):
        st.markdown("## Ejemplo real — Ciudad de Mendoza (P2P sobre mapa)")

        st.markdown("""
En este ejemplo se ubican los elementos sobre una troncal lógica **TR01**.
Podés **regenerar aleatoriamente** la ubicación de los nodos de campo (TR01-SW0x-ND0x)
para discutir distintas variantes de diseño.
""")

        if "p2p_rand_key" not in st.session_state:
            st.session_state["p2p_rand_key"] = 0

        cols_btn = st.columns([1, 3])
        with cols_btn[0]:
            if st.button("🔁 Regenerar nodos de campo"):
                st.session_state["p2p_rand_key"] += 1

        m, troncal_info = build_mendoza_p2p_map_osmnx(
            random_key=st.session_state["p2p_rand_key"]
        )
        components.html(m._repr_html_(), height=500)

        st.markdown("### Distancia de cada troncal FO (TR01-SW00-DC-8P → Sw Campo)")
        if troncal_info:
            for info in troncal_info:
                dist_m = info["distance_m"]
                dist_km = dist_m / 1000.0
                st.markdown(
                    f"- **{info['name']}**: {dist_m:,.0f} m (≈ {dist_km:.2f} km)",
                )
        else:
            st.caption("No se pudieron calcular las distancias en este ejemplo.")

# =========================================
# NUEVO DIAGRAMA DE RACKS — TRONCAL / INTERCONEXIÓN / CORE
# =========================================

def create_rack_connection_diagram():
    fig = go.Figure()

    # ----------------------------------------
    # DIMENSIONES BASE
    # ----------------------------------------
    rack_w = 0.22
    rack_h = 0.90

    rack_troncal_x0 = 0.05
    rack_int_x0     = 0.38
    rack_core_x0    = 0.70

    rack_y0 = 0.15

    # ========================================
    # RACK TRONCAL
    # ========================================
    fig.add_shape(type="rect",
        x0=rack_troncal_x0, y0=rack_y0,
        x1=rack_troncal_x0 + rack_w,
        y1=rack_y0 + rack_h,
        line=dict(color="black", width=2),
        fillcolor="#F7F7F7"
    )
    fig.add_annotation(
        x=rack_troncal_x0 + rack_w / 2,
        y=rack_y0 + rack_h + 0.03,
        text="RACK TRONCAL",
        showarrow=False,
        font=dict(size=12)
    )

    # ========================================
    # RACK INTERCONEXIÓN
    # ========================================
    fig.add_shape(type="rect",
        x0=rack_int_x0, y0=rack_y0,
        x1=rack_int_x0 + rack_w,
        y1=rack_y0 + rack_h,
        line=dict(color="black", width=2),
        fillcolor="#F7F7F7"
    )
    fig.add_annotation(
        x=rack_int_x0 + rack_w / 2,
        y=rack_y0 + rack_h + 0.03,
        text="RACK DE INTERCONEXIÓN",
        showarrow=False,
        font=dict(size=12)
    )

    # ========================================
    # RACK CORE
    # ========================================
    fig.add_shape(type="rect",
        x0=rack_core_x0, y0=rack_y0,
        x1=rack_core_x0 + rack_w,
        y1=rack_y0 + rack_h,
        line=dict(color="black", width=2),
        fillcolor="#F7F7F7"
    )
    fig.add_annotation(
        x=rack_core_x0 + rack_w / 2,
        y=rack_y0 + rack_h + 0.03,
        text="RACK CORE / NVR",
        showarrow=False,
        font=dict(size=12)
    )

    # ========================================
    # ODFs del RACK TRONCAL
    # ========================================
    odf_gap = 0.09
    odf_h   = 0.06

    odf1_y = rack_y0 + rack_h - (odf_h + odf_gap) * 1
    odf2_y = rack_y0 + rack_h - (odf_h + odf_gap) * 2
    odf3_y = rack_y0 + rack_h - (odf_h + odf_gap) * 3

    odf1_tr = _add_odf(fig, rack_troncal_x0 + 0.02, odf1_y, label="ODF TRONCAL 1")
    odf2_tr = _add_odf(fig, rack_troncal_x0 + 0.02, odf2_y, label="ODF TRONCAL 2")
    odf3_tr = _add_odf(fig, rack_troncal_x0 + 0.02, odf3_y, label="ODF TRONCAL 3")

    # ========================================
    # ODFs del RACK INTERCONEXIÓN
    # (Copias espejo de los troncales)
    # ========================================
    odf1_int = _add_odf(fig, rack_int_x0 + 0.02, odf1_y, label="ODF TRONCAL 1")
    odf2_int = _add_odf(fig, rack_int_x0 + 0.02, odf2_y, label="ODF TRONCAL 2")
    odf3_int = _add_odf(fig, rack_int_x0 + 0.02, odf3_y, label="ODF TRONCAL 3")

    # ========================================
    # NUEVO ODF CORE–NVR EN EL RACK INTERMEDIO
    # ========================================
    odf_core_int_y = odf3_y - 0.10
    odf_core_int = _add_odf(
        fig,
        rack_int_x0 + 0.02,
        odf_core_int_y,
        label="ODF CORE–NVR (INT)"
    )

    # ========================================
    # ODF CORE–NVR DEL RACK CORE
    # ========================================
    odf_core_y = odf1_y
    odf_core = _add_odf(fig, rack_core_x0 + 0.02, odf_core_y, label="ODF CORE – NVR")

    # Guardar para cableado
    fig._odf_troncal       = [odf1_tr, odf2_tr, odf3_tr]
    fig._odf_interconexion = [odf1_int, odf2_int, odf3_int]
    fig._odf_core          = odf_core
    fig._odf_core_int      = odf_core_int

    # ==================================================
    # SWITCH SW8P-CORE-NVR (BAJO EL ODF DEL CORE)
    # ==================================================
    sw_x = rack_core_x0 + 0.02
    sw_y = odf_core_y - 0.10

    sfp_ports = _add_switch(fig, sw_x, sw_y)
    fig._sfp_ports = sfp_ports

    # ==================================================
    # CABLES + ANIMACIÓN
    # ==================================================
    _build_all_cables(fig)
    _build_animation(fig)

    return fig


# =====================================================================
# PARTE SWITCH
# =====================================================================

def _add_switch(fig, x0, y0, width=0.18, height=0.08, label="SW8P-CORE-NVR"):
    x1 = x0 + width
    y1 = y0 + height

    fig.add_shape(
        type="rect",
        x0=x0, y0=y0, x1=x1, y1=y1,
        line=dict(color="black", width=1),
        fillcolor="black"
    )

    fig.add_annotation(
        x=(x0 + x1) / 2,
        y=y1 - 0.015,
        text=f"{label}<br>8 × SFP",
        font=dict(color="white", size=9),
        showarrow=False
    )

    # Puertos SFP
    sfp_positions = []
    port_w = width / 10
    gap = (width - (8 * port_w)) / 9

    px = x0 + gap
    py0 = y0 + 0.01
    py1 = py0 + 0.02

    for i in range(8):
        fig.add_shape(
            type="rect",
            x0=px, y0=py0, x1=px + port_w, y1=py1,
            line=dict(color="white", width=1),
            fillcolor="gray"
        )
        sfp_positions.append((px + port_w / 2, (py0 + py1) / 2))
        px += port_w + gap

    return sfp_positions


# =====================================================================
# PARTE CABLEADO (incluye los nuevos TR → CORE INT)
# =====================================================================

def _add_cable_segments(fig, segments, width=2):
    ids = []
    for (a, b) in segments:
        t = fig.add_trace(go.Scatter(
            x=[a[0], b[0]],
            y=[a[1], b[1]],
            mode="lines",
            line=dict(color="#E3D873", width=width),
            showlegend=False
        ))
        ids.append(len(fig.data) - 1)
    return ids


def _build_all_cables(fig):
    fig._cables = {}

    odf_tr = fig._odf_troncal
    odf_int = fig._odf_interconexion
    odf_core = fig._odf_core
    odf_core_int = fig._odf_core_int
    sfp = fig._sfp_ports

    # Puerto 1 de cada ODF
    p_tr1 = odf_tr[0][0]
    p_tr2 = odf_tr[1][0]
    p_tr3 = odf_tr[2][0]

    p_int1 = odf_int[0][0]
    p_int2 = odf_int[1][0]
    p_int3 = odf_int[2][0]

    p_core1 = odf_core[0]
    p_core2 = odf_core[1]
    p_core3 = odf_core[2]

    p_core_int_1 = odf_core_int[0]
    p_core_int_2 = odf_core_int[1]
    p_core_int_3 = odf_core_int[2]

    sfp1 = sfp[0]
    sfp2 = sfp[1]
    sfp3 = sfp[2]

    # -------------------------------
    # Conexión Troncal → Interconexión
    # -------------------------------

    def rect(a, b):
        xm = (a[0] + b[0]) / 2
        return [ (a, (xm, a[1])), ((xm, a[1]), (xm, b[1])), ((xm, b[1]), b) ]

    fig._cables["troncal_1"] = {
        "segments": rect(p_tr1, p_int1),
        "trace_ids": _add_cable_segments(fig, rect(p_tr1, p_int1), width=4)
    }
    fig._cables["troncal_2"] = {
        "segments": rect(p_tr2, p_int2),
        "trace_ids": _add_cable_segments(fig, rect(p_tr2, p_int2), width=4)
    }
    fig._cables["troncal_3"] = {
        "segments": rect(p_tr3, p_int3),
        "trace_ids": _add_cable_segments(fig, rect(p_tr3, p_int3), width=4)
    }

    # -------------------------------
    # NUEVO: Patchcords TRONCAL → CORE (INT)
    # -------------------------------
    fig._cables["patch_tr1_coreint"] = {
        "segments": [ (p_int1, p_core_int_1) ],
        "trace_ids": _add_cable_segments(fig, [(p_int1, p_core_int_1)], width=2)
    }
    fig._cables["patch_tr2_coreint"] = {
        "segments": [ (p_int2, p_core_int_2) ],
        "trace_ids": _add_cable_segments(fig, [(p_int2, p_core_int_2)], width=2)
    }
    fig._cables["patch_tr3_coreint"] = {
        "segments": [ (p_int3, p_core_int_3) ],
        "trace_ids": _add_cable_segments(fig, [(p_int3, p_core_int_3)], width=2)
    }

    # -------------------------------
    # CORE → SWITCH
    # -------------------------------
    fig._cables["core_1"] = {
        "segments": [ (p_core1, sfp1) ],
        "trace_ids": _add_cable_segments(fig, [(p_core1, sfp1)], width=2)
    }
    fig._cables["core_2"] = {
        "segments": [ (p_core2, sfp2) ],
        "trace_ids": _add_cable_segments(fig, [(p_core2, sfp2)], width=2)
    }
    fig._cables["core_3"] = {
        "segments": [ (p_core3, sfp3) ],
        "trace_ids": _add_cable_segments(fig, [(p_core3, sfp3)], width=2)
    }


# =====================================================================
# ANIMACIÓN (queda igual, pero luego la mejoramos)
# =====================================================================

def _build_animation(fig):
    frames = []
    cables = fig._cables
    keys = list(cables.keys())

    highlight = "#FFD700"
    normal = "#E3D873"

    for step, key in enumerate(keys):
        frame_data = []
        for k2 in keys:
            for trace_id in cables[k2]["trace_ids"]:
                frame_data.append({
                    "line": {
                        "color": highlight if k2 == key else normal,
                        "width": 4 if "troncal" in k2 else 2
                    }
                })
        frames.append(go.Frame(data=frame_data, name=f"step_{step}"))

    fig.frames = frames

    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "x": 0.10,
            "y": -0.12,
            "showactive": True,
            "buttons": [
                {
                    "label": "▶ Reproducir",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 800, "redraw": True}}]
                },
                {
                    "label": "⏹ Reset",
                    "method": "animate",
                    "args": [["step_0"], {"frame": {"duration": 0, "redraw": True}}]
                },
            ],
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {
                    "label": f"{i+1}",
                    "method": "animate",
                    "args": [[f"step_{i}"], {"frame": {"duration": 0, "redraw": True}}]
                }
                for i in range(len(keys))
            ],
            "x": 0.25,
            "y": -0.12,
            "len": 0.60
        }]
    )

# =========================================
# =========================================

st.markdown("### Diagrama físico de Racks con animación — Troncal / Interconexión / CORE")
fig_racks = create_rack_connection_diagram()
st.plotly_chart(fig_racks, use_container_width=True)

# =========================================================
# TAB 2 — ANILLO
# =========================================================
with tab_ring:
    st.subheader("Topología en Anillo")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("### Esquema lógico en Anillo")
            st.info(
                "Diagrama con **switches interconectados en anillo**, "
                "desde los cuales salen derivaciones hacia las cámaras."
            )

            fig_ring = create_topology_diagram("ring")
            st.plotly_chart(fig_ring, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown("### Indicadores Anillo (ejemplo)")
            st.metric("Total de cámaras", 32)
            st.metric("Fibra total estimada (m)", 3100)
            st.metric("N° de switches en anillo", 6)

            st.markdown("#### Ventajas / Desventajas")
            st.success("✔ Mejor redundancia ante cortes de fibra.")
            st.success("✔ Buen equilibrio entre cantidad de fibra y cobertura.")
            st.warning("✖ Mayor complejidad de diseño y configuración.")
            st.warning("✖ Requiere protocolos de anillo (STP/RSTP, ERPS, etc.).")

# =========================================================
# TAB 3 — FTTN
# =========================================================
with tab_fttn:
    st.subheader("Topología FTTN — Concepto general")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("### Esquema lógico FTTN")
            st.info(
                "Fibra hasta un **Nodo FTTN** (FOSC + divisor + ONU / switch), "
                "y desde allí distribución hacia varios puntos con UTP o FO secundaria."
            )

            fig_fttn = create_topology_diagram("fttn")
            st.plotly_chart(fig_fttn, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown("### Comentarios FTTN (ejemplo)")
            st.metric("Nodos FTTN", 3)
            st.metric("Cámaras promedio por nodo", 6)
            st.metric("Cobertura típica desde nodo", "200–400 m")

# =========================================================
# TAB 4 — COMPARATIVO GLOBAL
# =========================================================
with tab_comp:
    st.subheader("Comparativo Global de Topologías")

    data_comp = {
        "Topología": ["Punto a Punto", "Anillo", "FTTN"],
        "Cámaras (ej.)": [12, 32, 18],
        "Fibra total (m, ej.)": [3500, 3100, 2600],
        "Redundancia": ["Baja/Media", "Alta", "Media"],
        "Complejidad diseño": ["Baja/Media", "Media/Alta", "Media"],
        "Costo relativo": ["Medio/Alto", "Medio", "Medio/Bajo"],
        "Escalabilidad": ["Media", "Media", "Alta"],
    }

    df_comp = pd.DataFrame(data_comp)
    st.dataframe(df_comp, use_container_width=True)

