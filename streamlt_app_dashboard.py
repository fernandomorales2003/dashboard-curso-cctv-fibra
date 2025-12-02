import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from xml.etree import ElementTree as ET
import unicodedata
import pydeck as pdk

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
# FUNCIONES AUXILIARES PARA EL KMZ
# =========================================
def strip_accents(s: str) -> str:
    if s is None:
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def parse_kmz_points_lines(kmz_file) -> pd.DataFrame:
    """
    Convierte un KMZ de CCTV (como 'CCTV-IP FTTN.kmz') en un DataFrame
    con puntos para:
      - Fibra (puntos de las LineString)
      - Cámaras
      - Switches
      - UTP
      - Nodos FTTN (FOSC + divisor + ONU)
    """
    # Aseguramos que está al inicio
    kmz_file.seek(0)

    # Abrimos el ZIP y buscamos el primer .kml
    with zipfile.ZipFile(kmz_file) as z:
        kml_name = next(
            name for name in z.namelist()
            if name.lower().endswith(".kml")
        )
        kml_data = z.read(kml_name)

    # Parseamos el XML (KML)
    ns = {'k': 'http://www.opengis.net/kml/2.2'}
    root = ET.fromstring(kml_data)
    doc = root.find('k:Document', ns)

    rows = []

    def traverse_folder(folder, parent_path=""):
        name_el = folder.find('k:name', ns)
        folder_name = name_el.text.strip() if (name_el is not None and name_el.text) else ""
        path = f"{parent_path}/{folder_name}" if parent_path else folder_name

        # Placemark dentro de esta carpeta
        for pm in folder.findall('k:Placemark', ns):
            pm_name_el = pm.find('k:name', ns)
            pm_name = pm_name_el.text.strip() if (pm_name_el is not None and pm_name_el.text) else ""

            desc_el = pm.find('k:description', ns)
            desc = desc_el.text.strip() if (desc_el is not None and desc_el.text) else ""

            # --- POINT ---
            point_el = pm.find('.//k:Point/k:coordinates', ns)
            if point_el is not None and point_el.text:
                coord_text = point_el.text.strip()
                lon, lat, *_ = [float(x) for x in coord_text.split(',')]
                rows.append({
                    "name": pm_name,
                    "folder_path": path,
                    "geom_type": "Point",
                    "lon": lon,
                    "lat": lat,
                    "description": desc,
                    "segment_index": None,
                })

            # --- LINESTRING (ruta de fibra) ---
            line_el = pm.find('.//k:LineString/k:coordinates', ns)
            if line_el is not None and line_el.text:
                coords = line_el.text.strip().split()
                for idx, coord in enumerate(coords):
                    lon, lat, *_ = [float(x) for x in coord.split(',')]
                    rows.append({
                        "name": pm_name or "LineString",
                        "folder_path": path,
                        "geom_type": "LineStringPoint",
                        "lon": lon,
                        "lat": lat,
                        "description": desc,
                        "segment_index": idx,
                    })

        # Subcarpetas
        for sub in folder.findall('k:Folder', ns):
            traverse_folder(sub, path)

    # Recorremos las carpetas top-level
    if doc is None:
        return pd.DataFrame()

    for folder in doc.findall('k:Folder', ns):
        traverse_folder(folder)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # --- Categorización por carpeta / nombre ---
    def categorize(row):
        fp_raw = row.get("folder_path", "") or ""
        fp = strip_accents(fp_raw.lower())
        name = strip_accents((row.get("name", "") or "").lower())

        if "fibra" in fp:
            return "Fibra"
        if "fosc" in fp or "divisor" in fp or "onu" in fp:
            return "Nodo FTTN"
        if "camaras" in fp or name.startswith("c "):
            return "Camara"
        if "switch" in fp or name.startswith("s "):
            return "Switch"
        if "utp" in fp:
            return "UTP"
        return "Otro"

    df["category"] = df.apply(categorize, axis=1)

    # Zona interna / externa en base al path original (sin strip)
    def get_zone(path: str) -> str:
        path = path or ""
        if "Poligono interno" in path:
            return "Poligono interno"
        if "Polig externo" in path:
            return "Poligono externo"
        return ""

    df["zone"] = df["folder_path"].apply(get_zone)

    # Cadena 1/2/3/4
    def get_cadena(path: str) -> str:
        path = path or ""
        for i in range(1, 5):
            if f"Cadena {i}" in path:
                return f"Cadena {i}"
        return ""

    df["cadena"] = df["folder_path"].apply(get_cadena)

    return df


def build_pydeck_map(df: pd.DataFrame) -> pdk.Deck:
    """
    Construye un objeto pydeck.Deck con:
      - PathLayer para fibra
      - ScatterLayer para cámaras, switches, nodos FTTN, UTP
    """
    if df.empty:
        # Algo mínimo si no hay datos
        return pdk.Deck()

    # Vista inicial centrada en el promedio
    mean_lat = df["lat"].mean()
    mean_lon = df["lon"].mean()

    view_state = pdk.ViewState(
        latitude=mean_lat,
        longitude=mean_lon,
        zoom=16,
        pitch=45,
        bearing=0,
    )

    layers = []

    # ---------------------------
    # LÍNEAS DE FIBRA (PathLayer)
    # ---------------------------
    df_fibra_pts = df[(df["category"] == "Fibra") & (df["geom_type"] == "LineStringPoint")].copy()
    if not df_fibra_pts.empty:
        # Agrupamos por nombre+cadena para formar las rutas
        df_paths = (
            df_fibra_pts
            .sort_values(["name", "cadena", "segment_index"])
            .groupby(["name", "cadena"], dropna=False)
            .apply(lambda g: g[["lon", "lat"]].values.tolist())
            .reset_index(name="path")
        )

        fiber_layer = pdk.Layer(
            "PathLayer",
            df_paths,
            get_path="path",
            get_color=[0, 0, 0],  # negro
            width_scale=2,
            width_min_pixels=2,
            get_width=4,
        )
        layers.append(fiber_layer)

    # ---------------------------
    # PUNTOS: cámaras, switches, nodos, UTP
    # ---------------------------
    def add_scatter_layer(df_cat, color, radius=5, name=""):
        if df_cat.empty:
            return None
        return pdk.Layer(
            "ScatterplotLayer",
            df_cat,
            get_position=["lon", "lat"],
            get_fill_color=color,
            get_radius=radius,
            pickable=True,
        )

    df_points = df[df["geom_type"] == "Point"].copy()

    df_cams = df_points[df_points["category"] == "Camara"]
    df_switch = df_points[df_points["category"] == "Switch"]
    df_nodos = df_points[df_points["category"] == "Nodo FTTN"]
    df_utp = df_points[df_points["category"] == "UTP"]

    layer_cams = add_scatter_layer(df_cams, [0, 128, 255], radius=8, name="Cámaras")     # azul
    layer_switch = add_scatter_layer(df_switch, [255, 165, 0], radius=8, name="Switches") # naranja
    layer_nodos = add_scatter_layer(df_nodos, [0, 200, 0], radius=9, name="Nodos FTTN")   # verde
    layer_utp = add_scatter_layer(df_utp, [150, 150, 150], radius=5, name="UTP")          # gris

    for lyr in [layer_cams, layer_switch, layer_nodos, layer_utp]:
        if lyr is not None:
            layers.append(lyr)

    tooltip = {
        "html": "<b>{name}</b><br/>Tipo: {category}<br/>Zona: {zone}<br/>Cadena: {cadena}",
        "style": {"backgroundColor": "white", "color": "black"}
    }

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=layers,
        tooltip=tooltip,
    )
    return deck


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
        st.markdown("### Esquema lógico P2P")
        st.info(
            "Diagrama tipo **estrella**, donde desde el NVR/CORE salen enlaces directos "
            "de fibra hacia cada cámara o hacia pequeños switches remotos."
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
            "Diagrama con **switches interconectados en anillo**, "
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
# TAB 3 — FTTN (CCTV-IP) — USANDO KMZ + PYDECK
# =========================================================
with tab_fttn:
    st.subheader("Topología FTTN — CCTV-IP FTTN")

    st.markdown("""
    Subí el archivo **KMZ** del diseño CCTV para visualizar:
    - Las cadenas de fibra
    - Los nodos FTTN (FOSC + divisores + ONU)
    - Cámaras, switches y tramos UTP
    """)

    kmz_file = st.file_uploader(
        "📂 Subir archivo KMZ",
        type=["kmz", "kml"],
        help="Ejemplo: CCTV-IP FTTN.kmz"
    )

    if kmz_file is None:
        st.info("Subí el archivo KMZ para ver el diseño FTTN.")
    else:
        # Parseamos el KMZ → DataFrame
        df_kmz = parse_kmz_points_lines(kmz_file)

        if df_kmz.empty:
            st.error("No se encontraron elementos en el KMZ.")
        else:
            st.success(f"Se cargaron {len(df_kmz)} puntos desde el KMZ.")

            with st.expander("Ver muestra de datos parseados"):
                st.dataframe(df_kmz.head(50), use_container_width=True)

            # ==========================
            # FILA 1 — MAPA (PYDECK) + RESUMEN
            # ==========================
            col_map, col_scheme = st.columns([2, 1], gap="large")

            with col_map:
                st.markdown("### Mapa del sitio (KMZ)")

                col_l1, col_l2 = st.columns(2)
                with col_l1:
                    show_fibra = st.checkbox("Fibra (trazado)", value=True)
                    show_nodos = st.checkbox("Nodos FTTN (FOSC/ONU)", value=True)
                    show_cams = st.checkbox("Cámaras", value=True)
                with col_l2:
                    show_switch = st.checkbox("Switches", value=True)
                    show_utp = st.checkbox("Tramos UTP (puntos)", value=True)

                # Filtramos según checkboxes, pero respetando geom_type para la construcción del mapa
                df_filtered = df_kmz.copy()

                # Si desactivás fibra, marcamos category != Fibra
                if not show_fibra:
                    df_filtered = df_filtered[df_filtered["category"] != "Fibra"]
                if not show_nodos:
                    df_filtered = df_filtered[df_filtered["category"] != "Nodo FTTN"]
                if not show_cams:
                    df_filtered = df_filtered[df_filtered["category"] != "Camara"]
                if not show_switch:
                    df_filtered = df_filtered[df_filtered["category"] != "Switch"]
                if not show_utp:
                    df_filtered = df_filtered[df_filtered["category"] != "UTP"]

                st.caption("Mapa interactivo con pydeck (fibra + nodos + cámaras + switches + UTP).")
                deck = build_pydeck_map(df_filtered)
                st.pydeck_chart(deck)

            with col_scheme:
                st.markdown("### Esquema lógico FTTN (resumen)")

                total_cams = int((df_kmz["category"] == "Camara").sum())
                cams_int = int(((df_kmz["category"] == "Camara") &
                                (df_kmz["zone"] == "Poligono interno")).sum())
                cams_ext = int(((df_kmz["category"] == "Camara") &
                                (df_kmz["zone"] == "Poligono externo")).sum())

                total_switch = int((df_kmz["category"] == "Switch").sum())
                total_utp = int((df_kmz["category"] == "UTP").sum())
                total_nodos = int((df_kmz["category"] == "Nodo FTTN").sum())

                st.metric("Total cámaras", total_cams)
                st.metric("Cámaras internas", cams_int)
                st.metric("Cámaras externas", cams_ext)
                st.metric("Nodos FTTN (FOSC/ONU)", total_nodos)

                st.markdown("#### Otros elementos")
                st.write(f"- Switches totales: **{total_switch}**")
                st.write(f"- Puntos UTP (segmentos): **{total_utp}**")

            st.markdown("---")

            # ==========================
            # FILA 2 — RESUMEN POR ZONA
            # ==========================
            st.markdown("### Resumen por zona")

            col_z1, col_z2, col_z3 = st.columns(3)

            with col_z1:
                st.markdown("#### Cámaras por zona")
                cams_by_zone = (
                    df_kmz[df_kmz["category"] == "Camara"]["zone"]
                    .value_counts()
                    .rename_axis("Zona")
                    .reset_index(name="Cámaras")
                )
                st.dataframe(cams_by_zone, use_container_width=True)

            with col_z2:
                st.markdown("#### Switches por zona")
                sw_by_zone = (
                    df_kmz[df_kmz["category"] == "Switch"]["zone"]
                    .value_counts()
                    .rename_axis("Zona")
                    .reset_index(name="Switches")
                )
                st.dataframe(sw_by_zone, use_container_width=True)

            with col_z3:
                st.markdown("#### UTP por zona")
                utp_by_zone = (
                    df_kmz[df_kmz["category"] == "UTP"]["zone"]
                    .value_counts()
                    .rename_axis("Zona")
                    .reset_index(name="Puntos UTP")
                )
                st.dataframe(utp_by_zone, use_container_width=True)


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

    st.markdown("### Disparadores para la discusión en clase")
    st.markdown("- ¿En qué tipo de sitio conviene P2P? (pocos puntos, distancias cortas).")
    st.markdown("- ¿Cuándo justifica un anillo? (misión crítica, alta disponibilidad).")
    st.markdown("- ¿Cuándo FTTN equilibra costo, escalabilidad y mantenimiento en CCTV?")
