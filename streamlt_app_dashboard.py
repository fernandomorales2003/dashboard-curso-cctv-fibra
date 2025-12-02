import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
        st.metric("Total de cámaras", 12)
        st.metric("Puertos ópticos en CORE", 8)
        st.metric("Switches de campo", 3)

        st.markdown("#### Ventajas / Desventajas")
        st.success("✔ Arquitectura intuitiva (CORE → distribución → campo).")
        st.success("✔ Permite agrupar varias cámaras en un mismo punto de FO.")
        st.warning("✖ Sigue consumiendo varios puertos ópticos en el CORE.")
        st.warning("✖ Si falla un switch de campo, caen todas las cámaras de ese punto.")

    st.markdown("---")

    # ---------------------------------
    # EJEMPLO REAL — CIUDAD DE MENDOZA
    # ---------------------------------
    st.markdown("## Ejemplo real — Ciudad de Mendoza (P2P)")

    st.markdown("""
Imaginemos un diseño real en **Mendoza capital** para monitoreo urbano:

- El **CORE / NVR** está en un datacenter del municipio en la zona de **Microcentro**.
- Desde allí sale fibra hacia un **switch óptico de 8 bocas**.
- Cada puerto óptico alimenta un **punto de distribución** en la ciudad: plazas y nodos estratégicos.
- En cada punto de distribución hay un **switch de campo** (1 entrada óptica, varias salidas eléctricas) 
  que alimenta 2–4 cámaras IP con tramos cortos de UTP.
""")

    data_mza_p2p = {
        "Punto": [
            "CORE / NVR",
            "Sw 8P ópticas (Sala Técnica)",
            "Sw Campo A — Plaza Independencia",
            "Sw Campo B — Parque Central",
            "Sw Campo C — Terminal de Ómnibus",
        ],
        "Ubicación aproximada": [
            "Zona Microcentro (Municipalidad / Datacenter)",
            "Mismo edificio CORE",
            "Plaza Independencia (centro histórico)",
            "Parque Central (zona norte ciudad)",
            "Terminal de Ómnibus (acceso este)",
        ],
        "Rol en la red": [
            "Procesamiento, grabación y gestión",
            "Distribución óptica principal (8 puertos FO)",
            "Switch de campo (1 FO in, 4 UTP out)",
            "Switch de campo (1 FO in, 3 UTP out)",
            "Switch de campo (1 FO in, 3 UTP out)",
        ],
        "N° cámaras asociadas": [
            "-",  # CORE
            "-",  # Sw 8P
            "4 cámaras perimetrales plaza",
            "3 cámaras parque",
            "3 cámaras andenes / accesos",
        ],
        "Distancia FO aprox. desde CORE": [
            "—",
            "10–20 m (intra-edificio)",
            "800–1000 m",
            "1200–1500 m",
            "1500–1800 m",
        ],
        "Distancia típica UTP (cámara–switch)": [
            "—",
            "—",
            "30–60 m",
            "30–70 m",
            "20–50 m",
        ],
    }

    df_mza_p2p = pd.DataFrame(data_mza_p2p)
    st.markdown("### Tabla de ejemplo — nodos y cámaras en Mendoza")
    st.dataframe(df_mza_p2p, use_container_width=True)

    st.markdown("""
**Idea didáctica para el curso:**

- Podés pedir a los alumnos que:
  - Identifiquen cuáles enlaces son **FO** y cuáles son **UTP**.
  - Estimen el **presupuesto óptico** desde el CORE hasta cada switch de campo.
  - Verifiquen que las distancias de UTP cumplan con los límites de Ethernet.
  - Propongan **dónde agregar redundancia** (por ejemplo, un segundo enlace FO a la Terminal).

En los próximos pasos podemos armar ejemplos similares para:
- 🔁 La topología en **Anillo** (por ejemplo, bordeando el centro y zona oeste).  
- 🌿 La topología **FTTN**, usando nodos intermedios para barrios más alejados.
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
    st.info("Más adelante podemos sumar un **ejemplo real de anillo en Mendoza** (por ejemplo, un anillo que una Microcentro, Parque Central, La Alameda y Terminal).")

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
        st.warning("✖ Requiere buen diseño de alimentación eléctrica y alojamiento.")

    st.markdown("---")
    st.info("Luego podemos agregar un **caso real FTTN en Mendoza**, por ejemplo nodos en barrios periféricos con varias cámaras por nodo.")

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
