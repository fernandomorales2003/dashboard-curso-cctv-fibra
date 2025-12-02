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
st.caption("Visualización didáctica de topologías: Punto a Punto, Anillo y FTTN")

st.markdown("""
Este tablero está pensado para el curso de **Diseño CCTV**, 
comparando tres modelos de implementación de fibra óptica:
- 🔹 Punto a Punto  
- 🔹 Topología en Anillo  
- 🔹 Distribución FTTN (Fibra hasta el Nodo)
""")

st.markdown("---")


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

        # Switch óptico de 8 bocas (cuadrado azul) al medio
        sw_core_x, sw_core_y = -0.3, 0.5
        fig.add_trace(go.Scatter(
            x=[sw_core_x],
            y=[sw_core_y],
            mode="markers+text",
            marker=dict(size=20, symbol="square", color="royalblue"),
            text=["Sw 8P ópticas"],
            textposition="bottom center",
            showlegend=False
        ))

        # Enlace CORE → Sw 8P
        fig.add_trace(go.Scatter(
            x=[core_x, sw_core_x],
            y=[core_y, sw_core_y],
            mode="lines",
            line=dict(width=3),
            showlegend=False
        ))

        # Switches de campo (cuadrados verdes, 1 entrada óptica, varias salidas eléctricas)
        field_switches = [
            {"name": "Sw Campo A", "x": 0.3, "y": 0.8},
            {"name": "Sw Campo B", "x": 0.3, "y": 0.5},
            {"name": "Sw Campo C", "x": 0.3, "y": 0.2},
        ]

        cam_index = 1

        for fs in field_switches:
            sx, sy = fs["x"], fs["y"]

            # Fibra óptica Sw 8P → Sw Campo
            fig.add_trace(go.Scatter(
                x=[sw_core_x, sx],
                y=[sw_core_y, sy],
                mode="lines",
                line=dict(width=2),
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
                # Enlace eléctrico (UTP)
                fig.add_trace(go.Scatter(
                    x=[sx, cx],
                    y=[sy, cy],
                    mode="lines",
                    line=dict(width=1.8, dash="dot"),
                    showlegend=False
                ))
                # Cámara
                fig.add_trace(go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="markers+text",
                    marker=dict(size=12, symbol="circle"),
                    text=[f"Cam {cam_index}"],
                    textposition="top center",
                    showlegend=False
                ))
                cam_index += 1

        fig.update_layout(
            title="Topología Punto a Punto (CORE → Sw 8P → Sw de campo → Cámaras)",
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

        # Enlaces del anillo (líneas entre switches)
        for i in range(n):
            x0, y0 = switch_x[i], switch_y[i]
            x1, y1 = switch_x[(i + 1) % n], switch_y[(i + 1) % n]
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=2),
                showlegend=False
            ))

        # Switches (cuadrados azules)
        fig.add_trace(go.Scatter(
            x=switch_x,
            y=switch_y,
            mode="markers+text",
            marker=dict(size=16, symbol="square", color="royalblue"),
            text=[f"Sw {i+1}" for i in range(n)],
            textposition="top center",
            showlegend=False
        ))

        # Cámaras “colgando” de cada switch
        cam_offset = 0.25
        for i in range(n):
            sx, sy = switch_x[i], switch_y[i]
            cx = sx * (1 + cam_offset)
            cy = sy * (1 + cam_offset)
            fig.add_trace(go.Scatter(
                x=[sx, cx],
                y=[sy, cy],
                mode="lines",
                line=dict(width=1.5),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers+text",
                marker=dict(size=12, symbol="circle"),
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

        # Enlace CORE → Nodo
        fig.add_trace(go.Scatter(
            x=[core_x, node_x],
            y=[core_y, node_y],
            mode="lines",
            line=dict(width=3),
            showlegend=False
        ))

        # 3 switches a la derecha (cuadrados verdes)
        sw_positions = [
            (0.3, 0.8),
            (0.3, 0.5),
            (0.3, 0.2),
        ]

        for i, (sx, sy) in enumerate(sw_positions, start=1):
            # Enlace nodo → switch (fibra)
            fig.add_trace(go.Scatter(
                x=[node_x, sx],
                y=[node_y, sy],
                mode="lines",
                line=dict(width=2),
                showlegend=False
            ))
            # Switch
            fig.add_trace(go.Scatter(
                x=[sx],
                y=[sy],
                mode="markers+text",
                marker=dict(size=16, symbol="square", color="green"),
                text=[f"Sw {i}"],
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
                    line=dict(width=1.5, dash="dot"),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="markers+text",
                    marker=dict(size=12, symbol="circle"),
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
    - Rutas de FO sin compartir tramos (edge-disjoint) entre switches de campo.
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

    # Creamos una copia para ir "gastando" edges y lograr caminos distintos
    G_work = G.copy()

    # Mapa base: fondo negro con calles claras (simple)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB dark_matter",
    )

    # Nodos lógicos de nuestra red CCTV
    # Nota: CORE y Sw 8P están en la misma zona (datacenter), dentro del recuadro punteado
    nodes = [
        {
            "name": "CORE / NVR",
            "type": "CORE",
            "lat": center_lat + 0.0003,
            "lon": center_lon - 0.0003,
            "descripcion": "NVR dentro del Datacenter",
        },
        {
            "name": "Sw 8P ópticas (Sala Técnica)",
            "type": "SW_CORE",
            "lat": center_lat,
            "lon": center_lon,
            "descripcion": "Switch de distribución óptica principal dentro del Datacenter",
        },
        {
            "name": "Sw Campo A — Plaza Independencia",
            "type": "SW_CAMPO",
            "lat": center_lat + 0.004,
            "lon": center_lon,
            "descripcion": "Switch de campo alimentando 4 cámaras de la plaza",
        },
        {
            "name": "Sw Campo B — Parque Central",
            "type": "SW_CAMPO",
            "lat": center_lat + 0.005,
            "lon": center_lon - 0.006,
            "descripcion": "Switch de campo alimentando cámaras del Parque Central",
        },
        {
            "name": "Sw Campo C — Terminal de Ómnibus",
            "type": "SW_CAMPO",
            "lat": center_lat - 0.004,
            "lon": center_lon + 0.006,
            "descripcion": "Switch de campo alimentando cámaras en accesos a la Terminal",
        },
    ]

    df_nodes = pd.DataFrame(nodes)

    # Dibujamos el rectángulo punteado del DATACENTER alrededor del CORE / Sw 8P
    dc_delta_lat = 0.001
    dc_delta_lon = 0.0012
    folium.Rectangle(
        bounds=[
            [center_lat - dc_delta_lat, center_lon - dc_delta_lon],
            [center_lat + dc_delta_lat, center_lon + dc_delta_lon],
        ],
        color="white",
        weight=2,
        dash_array="5,5",
        fill=False,
        tooltip="DATACENTER (CORE / NVR + Sw 8P)",
    ).add_to(m)

    # Colores por tipo (todos brillantes para fondo negro)
    def node_color(t):
        if t == "CORE":
            return "red"
        if t == "SW_CORE":
            return "orange"
        if t == "SW_CAMPO":
            return "lime"
        return "white"

    # Marcadores de nodos:
    # - CORE: círculo
    # - SW_CORE y SW_CAMPO: cuadrados
    for _, row in df_nodes.iterrows():
        if row["type"] == "CORE":
            # NVR / CORE círculo rojo
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=7,
                color=node_color(row["type"]),
                fill=True,
                fill_color=node_color(row["type"]),
                fill_opacity=0.9,
                popup=f"{row['name']}<br>{row['descripcion']}",
                tooltip=row["name"],
            ).add_to(m)
        else:
            # Switches cuadrados (RegularPolygonMarker con 4 lados)
            folium.RegularPolygonMarker(
                location=[row["lat"], row["lon"]],
                number_of_sides=4,
                radius=7,
                color=node_color(row["type"]),
                fill=True,
                fill_color=node_color(row["type"]),
                fill_opacity=0.9,
                popup=f"{row['name']}<br>{row['descripcion']}",
                tooltip=row["name"],
            ).add_to(m)

    # Helper para dibujar ruta real por calles y "gastar" los edges usados
    def add_route_by_street(map_obj, G_work, lat0, lon0, lat1, lon1, tooltip: str):
        """
        Calcula la ruta más corta por la red vial entre (lat0, lon0) y (lat1, lon1)
        y la dibuja en el mapa. La polilínea se extiende hasta el punto exacto
        de origen y destino (CORE / Sw).

        Además, elimina del grafo de trabajo los edges usados en esta ruta,
        para que las siguientes rutas no compartan tramos (edge-disjoint).
        """
        try:
            # nearest_nodes espera X=lon, Y=lat
            orig_node = ox.distance.nearest_nodes(G_work, X=lon0, Y=lat0)
            dest_node = ox.distance.nearest_nodes(G_work, X=lon1, Y=lat1)

            route = nx.shortest_path(G_work, orig_node, dest_node, weight="length")
        except nx.NetworkXNoPath:
            # Si no encuentra camino, no dibuja nada
            return

        # Coordenadas de la ruta sobre calles
        route_coords = [(G_work.nodes[n]["y"], G_work.nodes[n]["x"]) for n in route]

        # Aseguramos que la línea empieza y termina en los equipos
        route_coords.insert(0, (lat0, lon0))      # origen exacto
        route_coords.append((lat1, lon1))         # destino exacto

        folium.PolyLine(
            locations=route_coords,
            color="white",   # fibra sobre fondo negro
            weight=3,
            tooltip=tooltip,
        ).add_to(map_obj)

        # "Gastamos" los edges del camino para que el próximo no los use
        # (evitamos compartir tramos entre diferentes rutas)
        for u, v in zip(route, route[1:]):
            if G_work.has_edge(u, v):
                G_work.remove_edge(u, v)
            if G_work.has_edge(v, u):
                G_work.remove_edge(v, u)

    # Recuperamos nodos clave
    core = df_nodes[df_nodes["type"] == "CORE"].iloc[0]      # no se usa para ruteo, solo visual
    sw_core = df_nodes[df_nodes["type"] == "SW_CORE"].iloc[0]
    sw_campo = df_nodes[df_nodes["type"] == "SW_CAMPO"]

    # CORE → Sw 8P (intra-edificio, recto y blanco)
    folium.PolyLine(
        locations=[[core["lat"], core["lon"]], [sw_core["lat"], sw_core["lon"]]],
        color="white",
        weight=4,
        tooltip="FO CORE → Sw 8P",
    ).add_to(m)

    # Sw 8P → cada Sw de campo, siguiendo calles y terminando en el SW
    # Las rutas se calculan sobre G_work, que vamos modificando
    for _, row in sw_campo.iterrows():
        add_route_by_street(
            m,
            G_work,
            sw_core["lat"],
            sw_core["lon"],
            row["lat"],
            row["lon"],
            tooltip=f"FO Sw 8P → {row['name']}",
        )

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

    # ---------------------------------
    # ESQUEMA LÓGICO + DIAGRAMA
    # ---------------------------------
    with col1:
        st.markdown("### Esquema lógico P2P (con switches de campo)")
        st.info(
            "CORE / NVR → Switch de 8 bocas ópticas → Fibra a switches de campo "
            "con 1 entrada óptica y varias salidas eléctricas → Cámaras por UTP."
        )
        st.markdown("**Flujo básico:**")
        st.markdown("- El CORE concentra el grabador / NVR y routing principal.")
        st.markdown("- Un switch con **8 puertos ópticos** distribuye la troncal.")
        st.markdown("- Cada puerto óptico alimenta un **switch de campo**.")
        st.markdown("- Desde cada switch de campo salen **2 o más cámaras** por UTP.")

        fig_p2p = create_topology_diagram("p2p")
        st.plotly_chart(fig_p2p, use_container_width=True)

    with col2:
        st.markdown("### Indicadores P2P (ejemplo)")
        st.metric("Total de cámaras (ejemplo)", 12)
        st.metric("Puertos ópticos en CORE", 8)
        st.metric("Switches de campo", 3)

        st.markdown("#### Ventajas / Desventajas")
        st.success("✔ Arquitectura intuitiva (CORE → distribución → campo).")
        st.success("✔ Permite agrupar varias cámaras en un mismo punto de FO.")
        st.warning("✖ Sigue consumiendo varios puertos ópticos en el CORE.")
        st.warning("✖ Si falla un switch de campo, caen todas las cámaras de ese punto.")

    st.markdown("---")

    # ---------------------------------
    # EJEMPLO REAL — MAPA CIUDAD DE MENDOZA
    # ---------------------------------
    st.markdown("## Ejemplo real — Ciudad de Mendoza (P2P sobre mapa)")

    st.markdown("""
En este ejemplo se ubican los elementos en la **ciudad de Mendoza**:

- **CORE / NVR** (círculo rojo) dentro de un **Datacenter**.
- Un **switch de 8 puertos ópticos** (cuadrado naranja) en la misma sala técnica.
- Tres **switches de campo** (cuadrados verdes), todos a menos de ~1 km del CORE:
  - `Sw Campo A — Plaza Independencia`
  - `Sw Campo B — Parque Central`
  - `Sw Campo C — Terminal de Ómnibus`

Las líneas representan los **enlaces de fibra**:
- CORE → Sw 8P (intra-edificio).
- Sw 8P → cada switch de campo (FO urbana), siguiendo rutas reales por las calles
  según la red vial de OpenStreetMap, con fondo oscuro simplificado.

Además, cada ruta hacia un switch de campo toma **un recorrido distinto**,
evitando compartir tramos entre sí, para poder discutir diferentes alternativas
de tendido.
""")

    m = build_mendoza_p2p_map_osmnx()
    components.html(m._repr_html_(), height=500)

    st.markdown("""
**Actividad sugerida para los alumnos:**

- Identificar sobre el mapa:
  - Dónde está el **Datacenter** (recuadro punteado).
  - Dónde está el **CORE / NVR** y el **Sw 8P** dentro del Datacenter.
  - La ubicación de cada **switch de campo** (plaza, parque, terminal).
- Analizar por qué el algoritmo eligió ese recorrido por calles (mínima distancia
  con la restricción de no reutilizar tramos).
- Discutir por dónde **realmente canalizarías** la fibra (postes, ductos, vereda, etc.)
  y si cambiarías el recorrido.
""")

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
        st.markdown("- Anillo de switches interconectados (fibra).")
        st.markdown("- Derivaciones hacia cámaras en cada nodo.")
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

    st.markdown("---")
    st.info("Luego podemos sumar un **ejemplo real de anillo en Mendoza** (por ejemplo, un anillo rodeando el microcentro y parques principales).")

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

        st.markdown("#### Ventajas / Desventajas")
        st.success("✔ Reduce la cantidad de fibra troncal desde el CORE.")
        st.success("✔ Permite escalar agregando nodos en nuevas zonas.")
        st.warning("✖ Más elementos activos en campo (más puntos de falla).")
        st.warning("✖ Requiere buen diseño de alimentación eléctrica y housing.")

    st.markdown("---")
    st.info("Más adelante podemos armar también un **mapa FTTN en Mendoza**, con nodos distribuidos por barrios.")

# =========================================================
# TAB 4 — COMPARATIVO GLOBAL
# =========================================================
with tab_comp:
    st.subheader("Comparativo Global de Topologías")

    st.markdown("""
    Vista comparativa (ejemplo) de las tres topologías:
    **Punto a Punto, Anillo y FTTN**.
    """)

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

    st.markdown("### Tabla comparativa")
    st.dataframe(df_comp, use_container_width=True)

    st.markdown("### Disparadores para la discusión en clase")
    st.markdown("- ¿En qué tipo de sitio conviene P2P con switches de campo? (ej: pocos nodos bien concentrados).")
    st.markdown("- ¿Cuándo justifica un anillo? (ej: corredores críticos y necesidad de alta disponibilidad).")
    st.markdown("- ¿Cuándo FTTN equilibra costo, escalabilidad y mantenimiento en CCTV urbano?")
