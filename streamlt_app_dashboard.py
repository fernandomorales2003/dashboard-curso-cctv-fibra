import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
import streamlit.components.v1 as components

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

        # CORE a la izquierda
        core_x, core_y = -0.9, 0.5
        fig.add_trace(go.Scatter(
            x=[core_x],
            y=[core_y],
            mode="markers+text",
            marker=dict(size=22, symbol="square"),
            text=["CORE / NVR"],
            textposition="bottom center",
            showlegend=False
        ))

        # Switch óptico de 8 bocas (switch central)
        sw_core_x, sw_core_y = -0.3, 0.5
        fig.add_trace(go.Scatter(
            x=[sw_core_x],
            y=[sw_core_y],
            mode="markers+text",
            marker=dict(size=20, symbol="hexagon"),
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

        # Switches de campo (1 entrada óptica, varias salidas eléctricas)
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

            # Switch de campo
            fig.add_trace(go.Scatter(
                x=[sx],
                y=[sy],
                mode="markers+text",
                marker=dict(size=18, symbol="square"),
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
            title="Topología Punto a Punto (CORE → Sw óptico → Sw de campo → Cámaras)",
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

        # Switches
        fig.add_trace(go.Scatter(
            x=switch_x,
            y=switch_y,
            mode="markers+text",
            marker=dict(size=16, symbol="square"),
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
                marker=dict(size=12),
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

        # CORE a la izquierda
        core_x, core_y = -0.9, 0.5
        fig.add_trace(go.Scatter(
            x=[core_x],
            y=[core_y],
            mode="markers+text",
            marker=dict(size=18, symbol="square"),
            text=["CORE / NVR"],
            textposition="bottom center",
            showlegend=False
        ))

        # Nodo FTTN al centro
        node_x, node_y = -0.3, 0.5
        fig.add_trace(go.Scatter(
            x=[node_x],
            y=[node_y],
            mode="markers+text",
            marker=dict(size=18, symbol="diamond"),
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

        # 3 switches a la derecha
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
                marker=dict(size=16, symbol="square"),
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
                    marker=dict(size=12),
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
# MAPA EJEMPLO REAL — MENDOZA (P2P, con FOLIUM)
# =========================================
def build_mendoza_p2p_map_folium() -> folium.Map:
    """
    Mapa de ejemplo en la ciudad de Mendoza (P2P) usando Folium + OpenStreetMap:
    - CORE / NVR en Microcentro
    - Sw 8P ópticas en el mismo edificio
    - 3 switches de campo (Plaza Independencia, Parque Central, Terminal)
      todos a menos de ~1 km del CORE.
    No requiere tokens ni keys.
    """

    # Coordenadas aproximadas del centro de Mendoza
    center_lat = -32.8895
    center_lon = -68.8458

    # NODOS (coordenadas aprox y distancias < ~1km)
    nodes = [
        {
            "name": "CORE / NVR",
            "type": "CORE",
            "lat": center_lat,
            "lon": center_lon,
            "descripcion": "Datacenter / Municipalidad (Microcentro)"
        },
        {
            "name": "Sw 8P ópticas (Sala Técnica)",
            "type": "SW_CORE",
            "lat": center_lat + 0.0003,   # ~30 m
            "lon": center_lon + 0.0003,
            "descripcion": "Switch de distribución óptica principal"
        },
        {
            "name": "Sw Campo A — Plaza Independencia",
            "type": "SW_CAMPO",
            "lat": center_lat + 0.004,    # ~450 m
            "lon": center_lon,
            "descripcion": "Switch de campo alimentando 4 cámaras de la plaza"
        },
        {
            "name": "Sw Campo B — Parque Central",
            "type": "SW_CAMPO",
            "lat": center_lat + 0.005,    # ~550 m
            "lon": center_lon - 0.006,    # ~650 m O
            "descripcion": "Switch de campo alimentando cámaras del Parque Central"
        },
        {
            "name": "Sw Campo C — Terminal de Ómnibus",
            "type": "SW_CAMPO",
            "lat": center_lat - 0.004,    # ~450 m S
            "lon": center_lon + 0.006,    # ~650 m E
            "descripcion": "Switch de campo alimentando cámaras en accesos a la Terminal"
        },
    ]

    df_nodes = pd.DataFrame(nodes)

    # Mapa base OSM
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # Colores por tipo de nodo
    def node_color(t):
        if t == "CORE":
            return "black"
        if t == "SW_CORE":
            return "blue"
        if t == "SW_CAMPO":
            return "green"
        return "gray"

    # Marcadores de nodos
    for _, row in df_nodes.iterrows():
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

    # Enlaces de FO:
    # CORE -> Sw 8P
    # Sw 8P -> cada Sw de campo
    core = df_nodes[df_nodes["type"] == "CORE"].iloc[0]
    sw_core = df_nodes[df_nodes["type"] == "SW_CORE"].iloc[0]
    sw_campo = df_nodes[df_nodes["type"] == "SW_CAMPO"]

    # CORE → Sw 8P
    folium.PolyLine(
        locations=[[core["lat"], core["lon"]], [sw_core["lat"], sw_core["lon"]]],
        color="black",
        weight=4,
        tooltip="FO CORE → Sw 8P",
    ).add_to(m)

    # Sw 8P → cada Sw de campo
    for _, row in sw_campo.iterrows():
        folium.PolyLine(
            locations=[[sw_core["lat"], sw_core["lon"]], [row["lat"], row["lon"]]],
            color="black",
            weight=3,
            tooltip=f"FO Sw 8P → {row['name']}",
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

- **CORE / NVR** en el Microcentro (datacenter / edificio municipal).
- Un **switch de 8 puertos ópticos** en la misma sala técnica.
- Tres **switches de campo**, todos a menos de ~1 km del CORE:
  - `Sw Campo A — Plaza Independencia`
  - `Sw Campo B — Parque Central`
  - `Sw Campo C — Terminal de Ómnibus`

Las líneas representan los **enlaces de fibra**:
- CORE → Sw 8P (intra-edificio).
- Sw 8P → cada switch de campo (FO urbana).
""")

    m = build_mendoza_p2p_map_folium()
    components.html(m._repr_html_(), height=500)

    st.markdown("""
**Actividad sugerida para los alumnos:**

- Identificar sobre el mapa:
  - Dónde está el **CORE** y el **switch de 8P**.
  - La ubicación de cada **switch de campo** (plaza, parque, terminal).
- Estimar la longitud de los enlaces de FO (todos menores a ~1 km).
- Proponer cuántas cámaras conectarías en cada switch de campo y qué zonas cubrirían.
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

    st.markmarkdown("### Disparadores para la discusión en clase")
    st.markdown("- ¿En qué tipo de sitio conviene P2P con switches de campo? (ej: pocos nodos bien concentrados).")
    st.markdown("- ¿Cuándo justifica un anillo? (ej: corredores críticos y necesidad de alta disponibilidad).")
    st.markdown("- ¿Cuándo FTTN equilibra costo, escalabilidad y mantenimiento en CCTV urbano?")
