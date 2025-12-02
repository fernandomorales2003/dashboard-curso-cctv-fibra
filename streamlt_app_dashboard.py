import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
import streamlit.components.v1 as components

import osmnx as ox
import networkx as nx
from folium.features import RegularPolygonMarker

# =========================================
# CONFIGURACIÓN GENERAL DEL DASHBOARD
# =========================================
st.set_page_config(
    page_title="DASHBOARD DISEÑO CCTV — Topologías de Fibra",
    layout="wide"
)

st.title("DASHBOARD DISEÑO CCTV")
st.warning("VERSIÓN 3 — FIBRAS FICOM + SW CUADRADOS", icon="⚙️")
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


# =========================================
# FUNCIÓN DIAGRAMAS LÓGICOS (PLOTLY)
# =========================================
def create_topology_diagram(topology: str) -> go.Figure:
    """
    Genera un diagrama esquemático simple para:
      - 'p2p'
      - 'ring'
      - 'fttn'
    """
    topo = topology.lower()

    # -------------------------------
    # PUNTO A PUNTO (con switches de campo)
    # -------------------------------
    if topo == "p2p":
        fig = go.Figure()

        # CORE / NVR (círculo rojo) a la izquierda
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

        # Switch óptico de 8 bocas (cuadrado naranja) al medio
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

        # Enlace CORE → Sw 8P (fibra FICOM)
        fig.add_trace(go.Scatter(
            x=[core_x, sw_core_x],
            y=[core_y, sw_core_y],
            mode="lines",
            line=dict(width=3, color=FICOM_COLOR),
            showlegend=False
        ))

        # Switches de campo (cuadrados verdes, 1 entrada óptica, varias salidas eléctricas)
        field_switches = [
            {"name": "TR01-SW01\nND01", "x": 0.3, "y": 0.8},
            {"name": "TR01-SW02\nND02", "x": 0.3, "y": 0.5},
            {"name": "TR01-SW03\nND03", "x": 0.3, "y": 0.2},
        ]

        cam_index = 1

        for fs in field_switches:
            sx, sy = fs["x"], fs["y"]

            # Fibra óptica Sw 8P → Sw Campo (FICOM)
            fig.add_trace(go.Scatter(
                x=[sw_core_x, sx],
                y=[sw_core_y, sy],
                mode="lines",
                line=dict(width=2, color=FICOM_COLOR),
                showlegend=False
            ))

            # Switch de campo (cuadrado verde)
            fig.add_trace(go.Scatter(
                x=[sx],
                y=[sy],
                mode="markers+text",
                marker=dict(size=18, symbol="square", color="green"),
                text=[fs["name"]],
                textposition="bottom center",
                showlegend=False
            ))

            # Desde cada Sw de campo, 2 cámaras (UTP eléctrico)
            cam_positions = [
                (sx + 0.35, sy + 0.12),
                (sx + 0.35, sy - 0.12),
            ]
            for (cx, cy) in cam_positions:
                # Enlace eléctrico (UTP) – gris punteado
                fig.add_trace(go.Scatter(
                    x=[sx, cx],
                    y=[sy, cy],
                    mode="lines",
                    line=dict(width=1.8, dash="dot", color="gray"),
                    showlegend=False
                ))
                # Cámara (triángulo)
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
        # 6 switches en círculo
        n = 6
        radius = 0.6
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        switch_x = radius * np.cos(angles)
        switch_y = radius * np.sin(angles) + 0.1  # un poquito arriba

        fig = go.Figure()

        # Enlaces del anillo (fibra FICOM)
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

        # Switches (cuadrados azules) – nombres genéricos TR01-SW01..06
        fig.add_trace(go.Scatter(
            x=switch_x,
            y=switch_y,
            mode="markers+text",
            marker=dict(size=16, symbol="square", color="royalblue"),
            text=[f"TR01-SW0{i+1}" for i in range(n)],
            textposition="top center",
            showlegend=False
        ))

        # Cámaras “colgando” de cada switch (UTP)
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

        # CORE / NVR círculo rojo
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

        # Nodo FTTN cuadrado azul
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

        # Enlace CORE → Nodo (fibra FICOM)
        fig.add_trace(go.Scatter(
            x=[core_x, node_x],
            y=[core_y, node_y],
            mode="lines",
            line=dict(width=3, color=FICOM_COLOR),
            showlegend=False
        ))

        # 3 switches a la derecha (cuadrados verdes)
        sw_positions = [
            (0.3, 0.8),
            (0.3, 0.5),
            (0.3, 0.2),
        ]

        for i, (sx, sy) in enumerate(sw_positions, start=1):
            # Enlace nodo → switch (fibra FICOM)
            fig.add_trace(go.Scatter(
                x=[node_x, sx],
                y=[node_y, sy],
                mode="lines",
                line=dict(width=2, color=FICOM_COLOR),
                showlegend=False
            ))
            # Switch
            fig.add_trace(go.Scatter(
                x=[sx],
                y=[sy],
                mode="markers+text",
                marker=dict(size=16, symbol="square", color="green"),
                text=[f"TR01-SW0{i+1}-ND0{i}"],
                textposition="bottom center",
                showlegend=False
            ))
            # 2 cámaras colgando de cada switch (UTP corto)
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

    # Fallback
    return go.Figure()


# =========================================
# MAPA EJEMPLO REAL — MENDOZA (P2P, OSMnx + Folium)
# =========================================
def build_mendoza_p2p_map_osmnx() -> folium.Map:
    """
    Mapa de ejemplo en la ciudad de Mendoza (P2P) usando:
    - OSMnx para obtener la red vial real y calcular rutas por las calles.
    - Folium + CartoDB Dark Matter como fondo (calles sobre fondo oscuro).
    - Switches de campo ubicados a mitad de cuadra.
    - 2 cámaras por cada switch de campo, en las esquinas de la cuadra (UTP).
    """

    # Centro aproximado de Mendoza
    center_lat = -32.8895
    center_lon = -68.8458

    # Grafo de calles en un radio de ~2 km (solo para ruteo)
    G = ox.graph_from_point(
        (center_lat, center_lon),
        dist=2000,
        network_type="drive"
    )

    # Copia para ir "gastando" edges y lograr caminos distintos
    G_work = G.copy()

    # Mapa base: fondo negro con calles claras (simple)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB dark_matter",
    )

    # Nodos lógicos de nuestra red CCTV (lat/lon "referencia")
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
        {
            "name": "TR01-SW01-ND01",
            "type": "SW_CAMPO",
            "lat": center_lat + 0.004,
            "lon": center_lon,
            "descripcion": "Switch de campo Nodo 01 (troncal TR01)",
        },
        {
            "name": "TR01-SW02-ND02",
            "type": "SW_CAMPO",
            "lat": center_lat + 0.005,
            "lon": center_lon - 0.006,
            "descripcion": "Switch de campo Nodo 02 (troncal TR01)",
        },
        {
            "name": "TR01-SW03-ND03",
            "type": "SW_CAMPO",
            "lat": center_lat - 0.004,
            "lon": center_lon + 0.006,
            "descripcion": "Switch de campo Nodo 03 (troncal TR01)",
        },
    ]

    df_nodes = pd.DataFrame(nodes)

    # Coordenadas reales para dibujo:
    # - mid_lat/mid_lon = posición donde se dibuja el equipo
    # - corner1/2 = esquinas de la cuadra para las cámaras (solo SW_CAMPO)
    df_nodes["mid_lat"] = df_nodes["lat"]
    df_nodes["mid_lon"] = df_nodes["lon"]
    df_nodes["corner1_lat"] = np.nan
    df_nodes["corner1_lon"] = np.nan
    df_nodes["corner2_lat"] = np.nan
    df_nodes["corner2_lon"] = np.nan

    # Para cada SW_CAMPO: buscar la cuadra (edge) más cercana y colocar el switch a mitad de cuadra
    for idx, row in df_nodes[df_nodes["type"] == "SW_CAMPO"].iterrows():
        u, v, key = ox.distance.nearest_edges(G, X=row["lon"], Y=row["lat"])
        y_u, x_u = G.nodes[u]["y"], G.nodes[u]["x"]
        y_v, x_v = G.nodes[v]["y"], G.nodes[v]["x"]

        # Punto medio de la cuadra
        mid_lat = (y_u + y_v) / 2
        mid_lon = (x_u + x_v) / 2

        df_nodes.loc[idx, "mid_lat"] = mid_lat
        df_nodes.loc[idx, "mid_lon"] = mid_lon
        df_nodes.loc[idx, "corner1_lat"] = y_u
        df_nodes.loc[idx, "corner1_lon"] = x_u
        df_nodes.loc[idx, "corner2_lat"] = y_v
        df_nodes.loc[idx, "corner2_lon"] = x_v

    # Recuadro del DATACENTER (ligeramente más chico)
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

    # Colores por tipo (brillantes para fondo negro)
    def node_color(t):
        if t == "CORE":
            return "red"
        if t == "SW_CORE":
            return "orange"
        if t == "SW_CAMPO":
            return "lime"
        return "white"

    # Marcadores:
    # - CORE: círculo en su lat/lon original (dentro del recuadro)
    # - SW_CORE: cuadrado en su lat/lon original
    # - SW_CAMPO: cuadrados verdes a mitad de cuadra (mid_lat/mid_lon)
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
                fill_color=node_color(row["type"]],
                fill_opacity=0.9,
                popup=f"{row['name']}<br>{row['descripcion']}",
                tooltip=row["name"],
            ).add_to(m)
        elif row["type"] == "SW_CAMPO":
            RegularPolygonMarker(
                location=[row["mid_lat"], row["mid_lon"]],
                number_of_sides=4,
                radius=9,
                color=node_color(row["type"]],
                fill=True,
                fill_color=node_color(row["type"]],
                fill_opacity=0.9,
                popup=f"{row['name']}<br>{row['descripcion']}",
                tooltip=row["name"],
            ).add_to(m)

    # Helper para dibujar ruta real por calles con fallback y color propio
    def add_route_by_street(
        map_obj,
        G_work,
        G_full,
        lat0,
        lon0,
        lat1,
        lon1,
        tooltip: str,
        color: str,
    ):
        """
        Intenta calcular la ruta más corta en G_work.
        Si no hay camino, hace fallback a G_full (permitiendo compartir tramos).
        Dibuja la traza en el color indicado.
        """
        def _shortest_path(G_used):
            orig_node = ox.distance.nearest_nodes(G_used, X=lon0, Y=lat0)
            dest_node = ox.distance.nearest_nodes(G_used, X=lon1, Y=lat1)
            return nx.shortest_path(G_used, orig_node, dest_node, weight="length")

        route = None
        use_fallback = False

        try:
            route = _shortest_path(G_work)
        except nx.NetworkXNoPath:
            # Fallback al grafo completo
            try:
                route = _shortest_path(G_full)
                use_fallback = True
            except nx.NetworkXNoPath:
                return  # ni siquiera en el grafo completo hay camino

        route_coords = [(G_full.nodes[n]["y"], G_full.nodes[n]["x"]) for n in route]

        # Arrancar y terminar en los equipos exactos
        route_coords.insert(0, (lat0, lon0))
        route_coords.append((lat1, lon1))

        folium.PolyLine(
            locations=route_coords,
            color=color,
            weight=3,
            tooltip=tooltip,
        ).add_to(map_obj)

        # Si usamos G_work, gastamos edges; si usamos fallback, no tocamos nada
        if not use_fallback:
            for u, v in zip(route, route[1:]):
                if G_work.has_edge(u, v):
                    G_work.remove_edge(u, v)
                if G_work.has_edge(v, u):
                    G_work.remove_edge(v, u)

    # Recuperamos nodos clave
    core = df_nodes[df_nodes["type"] == "CORE"].iloc[0]      # solo visual
    sw_core = df_nodes[df_nodes["type"] == "SW_CORE"].iloc[0]
    sw_campo = df_nodes[df_nodes["type"] == "SW_CAMPO"]

    # CORE → Sw 8P (intra-edificio, recto y FICOM)
    folium.PolyLine(
        locations=[[core["lat"], core["lon"]], [sw_core["lat"], sw_core["lon"]]],
        color=FICOM_COLOR,
        weight=4,
        tooltip="FO CORE / NVR → TR01-SW00-DC-8P",
    ).add_to(m)

    # Colores distintos para cada traza Sw 8P → Sw Campo
    route_colors = ["#4FB4CA", "#00CC83", "#3260EA"]  # paleta FICOM/derivados

    # Sw 8P → cada Sw de campo, siguiendo calles y terminando a mitad de cuadra
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
            tooltip=f"FO {sw_core['name']} → {row['name']}",
            color=color,
        )

    # -------------------------------
    # UTP desde cada Sw Campo a las dos esquinas de la cuadra (2 cámaras)
    # -------------------------------
    for _, row in sw_campo.iterrows():
        mid_lat = row["mid_lat"]
        mid_lon = row["mid_lon"]
        c1_lat, c1_lon = row["corner1_lat"], row["corner1_lon"]
        c2_lat, c2_lon = row["corner2_lat"], row["corner2_lon"]

        # Cable UTP punteado (mucho más fino que la fibra) hacia esquina 1
        folium.PolyLine(
            locations=[[mid_lat, mid_lon], [c1_lat, c1_lon]],
            color="white",
            weight=1,
            dash_array="4,4",
            tooltip=f"UTP desde {row['name']} a Cámara 1 (≤ 100 m aprox.)",
        ).add_to(m)

        # Cámara 1 como triángulo en esquina 1
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

        # Cable UTP punteado hacia esquina 2
        folium.PolyLine(
            locations=[[mid_lat, mid_lon], [c2_lat, c2_lon]],
            color="white",
            weight=1,
            dash_array="4,4",
            tooltip=f"UTP desde {row['name']} a Cámara 2 (≤ 100 m aprox.)",
        ).add_to(m)

        # Cámara 2 como triángulo en esquina 2
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

    return m


# =========================================
# TABS PRINCIPALES
# =========================================
tab_p2p, tab_ring, tab_fttn, tab_comp = st.tabs(
    ["🔌 Punto a Punto", "⭕ Anillo", "🌿 FTTN (CCTV-IP)", "📊 Comparativo Global"]
)

# =========================================================
# TAB 1 — PUNTO A PUNTO
# =========================================================
with tab_p2p:
    st.subheader("Topología Punto a Punto")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
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

    st.markdown("## Ejemplo real — Ciudad de Mendoza (P2P sobre mapa)")

    st.markdown("""
En este ejemplo se ubican los elementos sobre una troncal lógica **TR01**:

- **CORE / NVR** (círculo rojo) dentro de un **Datacenter**.
- `TR01-SW00-DC-8P` (cuadrado naranja) como switch de 8 puertos ópticos en Sala Técnica.
- Tres **switches de campo** (cuadrados verdes), ubicados a **mitad de cuadra**, todos a menos de ~1 km del CORE:
  - `TR01-SW01-ND01`  
  - `TR01-SW02-ND02`  
  - `TR01-SW03-ND03`  

Las líneas representan:
- Enlaces de **fibra óptica** (con colores distintos) desde `TR01-SW00-DC-8P` hacia cada Sw Campo.
- Desde cada Sw Campo, **2 UTP punteados finos** (≤ 100 m aprox.) hacia las **dos esquinas de esa cuadra**, 
  donde se marcan las **cámaras** con triángulos amarillos.

La nomenclatura `TR01-SWxx-NDyy` ayuda a los técnicos a:
- Identificar la **troncal** (TR01).
- Saber qué **switch** están trabajando (`SW01/02/03`).
- Diferenciar el **nodo lógico** dentro de la troncal (`ND01/02/03`).
""")

    m = build_mendoza_p2p_map_osmnx()
    components.html(m._repr_html_(), height=500)

# =========================================================
# TAB 2 — ANILLO
# =========================================================
with tab_ring:
    st.subheader("Topología en Anillo")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("### Esquema lógico en Anillo")
        st.info(
            "Diagrama con **switches interconectados en anillo**, "
            "desde los cuales salen derivaciones hacia las cámaras."
        )
        st.markdown("**Idea visual:**")
        st.markdown("- Anillo de switches interconectados (fibra en color FICOM).")
        st.markdown("- Derivaciones hacia cámaras en cada nodo (UTP).")
        st.markdown("- Soporta redundancia por camino alternativo ante cortes.")

        fig_ring = create_topology_diagram("ring")
        st.plotly_chart(fig_ring, use_container_width=True)

    with col2:
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
# TAB 3 — FTTN (conceptual)
# =========================================================
with tab_fttn:
    st.subheader("Topología FTTN — Concepto general")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("### Esquema lógico FTTN")
        st.info(
            "Fibra hasta un **Nodo FTTN** (FOSC + divisor + ONU / switch), "
            "y desde allí distribución hacia varios puntos con UTP o FO secundaria."
        )
        st.markdown("**Flujo básico:**")
        st.markdown("- CORE / NVR en un punto central (datacenter).")
        st.markdown("- Fibra troncal hasta nodos FTTN estratégicos.")
        st.markdown("- En cada nodo: elementos de acceso (ONU / switch).")
        st.markdown("- Desde el nodo, cámaras cercanas por UTP o FO corta.")

        fig_fttn = create_topology_diagram("fttn")
        st.plotly_chart(fig_fttn, use_container_width=True)

    with col2:
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
