import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="DASHBOARD DISEÑO CCTV — Topologías de Fibra",
    layout="wide"
)

# =========================
# ENCABEZADO
# =========================
st.title("DASHBOARD DISEÑO CCTV")
st.caption("Visualización didáctica de topologías: Punto a Punto, Anillo y FTTN")

st.markdown("""
Este tablero está pensado para usarlo en el curso de **Diseño CCTV**, 
comparando tres modelos de implementación de fibra óptica:
- 🔹 Punto a Punto  
- 🔹 Topología en Anillo  
- 🔹 Distribución FTTN (Fibra hasta el Nodo)
""")

st.markdown("---")

# =========================
# TABS PRINCIPALES
# =========================
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
        st.markdown("### Esquema lógico P2P")
        st.info(
            "Aquí iría un diagrama tipo **estrella**, donde desde el NVR/CORE "
            "salen enlaces directos de fibra hacia cada cámara o hacia pequeños switches remotos."
        )
        st.markdown("**Idea visual:**")
        st.markdown("- Centro: NVR / Core")
        st.markdown("- Ramas: enlaces directos de fibra hacia cada punto remoto")
        st.markdown("- Últimos metros: UTP hacia la cámara (si aplica)")
        st.image(
            "https://via.placeholder.com/600x300.png?text=Esquema+Punto+a+Punto",
            caption="Placeholder de diagrama Punto a Punto",
            use_column_width=True
        )

    with col2:
        st.markdown("### Indicadores P2P (ejemplo)")
        st.metric("Total de cámaras", 32)
        st.metric("Fibra total estimada (m)", 4200)
        st.metric("N° de enlaces directos", 32)

        st.markdown("#### Ventajas / Desventajas")
        st.success("✔ Arquitectura simple, fácil de entender")
        st.warning("✖ Mayor consumo de fibra y puertos en el core")
        st.warning("✖ Escalabilidad limitada en grandes sitios")

# =========================================================
# TAB 2 — ANILLO
# =========================================================
with tab_ring:
    st.subheader("Topología en Anillo")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("### Esquema lógico en Anillo")
        st.info(
            "Aquí iría un diagrama con **switches interconectados en anillo**, "
            "desde los cuales salen derivaciones hacia las cámaras."
        )
        st.markdown("**Idea visual:**")
        st.markdown("- Anillo de switches interconectados")
        st.markdown("- Derivaciones (spurs) hacia cámaras o pequeños grupos")
        st.markdown("- Soporta redundancia por camino alternativo")
        st.image(
            "https://via.placeholder.com/600x300.png?text=Esquema+Anillo",
            caption="Placeholder de diagrama en Anillo",
            use_column_width=True
        )

    with col2:
        st.markdown("### Indicadores Anillo (ejemplo)")
        st.metric("Total de cámaras", 32)
        st.metric("Fibra total estimada (m)", 3100)
        st.metric("N° de switches en anillo", 6)

        st.markdown("#### Ventajas / Desventajas")
        st.success("✔ Mejor redundancia ante cortes de fibra")
        st.success("✔ Mejor uso de fibra que P2P en sitios grandes")
        st.warning("✖ Mayor complejidad de diseño y configuración")

# =========================================================
# TAB 3 — FTTN (CCTV-IP) — USANDO KMZ
# =========================================================
with tab_fttn:
    st.subheader("Topología FTTN — CCTV-IP FTTN")

    st.markdown("""
    En este tab se trabaja sobre el diseño real del archivo **KMZ** (CCTV-IP FTTN), 
    mostrando la troncal de fibra, los nodos FTTN (FOSC + divisores + ONU), 
    y la distribución hacia cámaras internas y externas.
    """)

    # -------------------------
    # FILA 1 — MAPA + ESQUEMA
    # -------------------------
    col_map, col_scheme = st.columns([2, 1], gap="large")

    with col_map:
        st.markdown("### Mapa del sitio (KMZ)")
        st.caption("Aquí se visualiza el KMZ con las cadenas de fibra, FOSC/divisores, cámaras y switches.")

        st.markdown("**Capas a mostrar:**")
        col_layers1, col_layers2 = st.columns(2)

        with col_layers1:
            st.checkbox("Cadena 1", value=True)
            st.checkbox("Cadena 2", value=True)
            st.checkbox("Cadena 3", value=False)
            st.checkbox("Cadena 4", value=False)

        with col_layers2:
            st.checkbox("FOSC / Divisores / ONU", value=True)
            st.checkbox("Cámaras internas", value=True)
            st.checkbox("Cámaras externas", value=True)
            st.checkbox("Switches", value=True)

        st.warning(
            "🔧 Aquí iría el mapa interactivo (folium/leafmap/pydeck) con los datos parseados del KMZ."
        )
        st.image(
            "https://via.placeholder.com/700x350.png?text=Mapa+FTTN+desde+KMZ",
            caption="Placeholder mapa FTTN (KMZ)",
            use_column_width=True
        )

    with col_scheme:
        st.markdown("### Esquema lógico FTTN")
        st.info(
            "Diagrama en forma de **árbol**: troncal de fibra → nodos FTTN (FOSC + divisor + ONU) "
            "→ switches por zona → cámaras IP."
        )

        st.markdown("**Resumen conceptual:**")
        st.markdown("- Fibra troncal desde el core hacia las Cadenas (1–4)")
        st.markdown("- En cada nodo FTTN: FOSC + divisor + ONU")
        st.markdown("- Desde el nodo: UTP corto hacia cámaras, vía switches")

        st.image(
            "https://via.placeholder.com/500x280.png?text=Esquema+FTTN",
            caption="Placeholder de diagrama FTTN",
            use_column_width=True
        )

    st.markdown("---")

    # -------------------------
    # FILA 2 — INDICADORES FTTN
    # -------------------------
    st.markdown("### Indicadores del diseño FTTN (ejemplo)")

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

    with col_kpi1:
        st.markdown("#### Cámaras")
        st.metric("Cámaras polígono interno", 18)
        st.metric("Cámaras polígono externo", 14)
        st.metric("Total cámaras", 32)

    with col_kpi2:
        st.markdown("#### Switches / Nodos")
        st.metric("Switches internos", 4)
        st.metric("Switches externos", 4)
        st.metric("Total switches", 8)

    with col_kpi3:
        st.markdown("#### Divisores / Potencias (ejemplo)")
        st.metric("N° de FOSC + divisores", 6)
        st.metric("Tipos de splitters", "5:95 / 10:90 / 1:4")
        st.metric("Potencia mín. en ONUs (dBm)", -21.5)

    st.markdown("---")

    # -------------------------
    # FILA 3 — SIMULADOR DE FALLAS (DIDÁCTICO)
    # -------------------------
    st.markdown("### Simulación de falla en la troncal FTTN")

    col_sim1, col_sim2 = st.columns([2, 1], gap="large")

    with col_sim1:
        st.markdown("#### Selección de falla")
        cadena_seleccionada = st.selectbox(
            "Seleccionar cadena donde ocurre el corte:",
            ["Cadena 1", "Cadena 2", "Cadena 3", "Cadena 4"]
        )
        ubicacion_falla = st.slider(
            "Ubicación aproximada del corte sobre la cadena (0 = inicio, 100 = fin)",
            min_value=0,
            max_value=100,
            value=40,
            step=5
        )

        st.warning(
            "Aquí se podría recalcular qué cámaras quedan **online** y cuáles quedan **offline** "
            "según la ubicación de la falla en la cadena seleccionada."
        )

        st.image(
            "https://via.placeholder.com/700x300.png?text=Camaras+ONLINE+vs+OFFLINE",
            caption="Placeholder: visualización cámaras online/offline ante un corte",
            use_column_width=True
        )

    with col_sim2:
        st.markdown("#### Resumen del impacto (ejemplo)")
        st.metric("Cámaras online", 24)
        st.metric("Cámaras offline", 8)
        st.metric("Porcentaje operativo", "75%")

        st.markdown("**Interpretación didáctica:**")
        st.markdown("- ¿Qué tan crítico es el corte según su ubicación?")
        st.markdown("- ¿Conviene segmentar de otra forma las cadenas?")
        st.markdown("- ¿Dónde conviene ubicar nodos y FOSC?")

# =========================================================
# TAB 4 — COMPARATIVO GLOBAL
# =========================================================
with tab_comp:
    st.subheader("Comparativo Global de Topologías")

    st.markdown("""
    Esta vista permite comparar, de forma didáctica, los tres modelos de implementación:
    **Punto a Punto, Anillo y FTTN**.
    """)

    # Tabla comparativa de ejemplo
    data_comp = {
        "Topología": ["Punto a Punto", "Anillo", "FTTN"],
        "Cámaras (ej.)": [32, 32, 32],
        "Fibra total (m, ej.)": [4200, 3100, 2600],
        "Redundancia": ["Baja", "Alta", "Media"],
        "Complejidad diseño": ["Baja", "Media", "Media/Alta"],
        "Costo relativo": ["Alto", "Medio", "Medio/Bajo"],
        "Escalabilidad": ["Baja", "Media", "Alta"],
    }

    df_comp = pd.DataFrame(data_comp)

    st.markdown("### Tabla comparativa")
    st.dataframe(df_comp, use_container_width=True)

    st.markdown("### Comentarios para discusión en clase")
    st.markdown("- ¿En qué tipo de sitio conviene P2P? (Ej: pocos puntos, distancias cortas).")
    st.markdown("- ¿Cuándo justifica un anillo? (Ej: misión crítica, necesidad de redundancia fuerte).")
    st.markdown("- ¿Cuándo FTTN equilibra costo, escalabilidad y facilidad de mantenimiento?")

