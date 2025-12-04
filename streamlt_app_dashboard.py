import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
import streamlit.components.v1 as components
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
    "VERSIÓN 7 — Racks + Unifilar detallado + Colores FO + Mapa real",
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

# Paleta estándar de código de colores de fibras (primeras fibras)
FIBER_STANDARD_COLORS = [
    ("Fibra 1", "Azul",   "#0000FF"),
    ("Fibra 2", "Naranja", "#FF7F00"),
    ("Fibra 3", "Verde",   "#008000"),
]

PATCHCORD_COLOR = "#FFD700"  # amarillo patchcord


# =======================================================
# FUNCIÓN DIAGRAMA LÓGICO GENERAL (PLOTLY)
# =======================================================
def create_topology_diagram(topology: str) -> go.Figure:
    topo = topology.lower()

    # -------------------------------
    # PUNTO A PUNTO
    # -------------------------------
    if topo == "p2p":
        fig = go.Figure()

        core_x, core_y = -0.9, 0.5
        fig.add_trace(go.Scatter(
            x=[core_x], y=[core_y],
            mode="markers+text",
            marker=dict(size=22, symbol="circle", color="red"),
            text=["CORE / NVR"], textposition="bottom center",
            showlegend=False
        ))

        sw_core_x, sw_core_y = -0.3, 0.5
        fig.add_trace(go.Scatter(
            x=[sw_core_x], y=[sw_core_y],
            mode="markers+text",
            marker=dict(size=20, symbol="square", color="orange"),
            text=["TR01-SW00-DC-8P"], textposition="bottom center",
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[core_x, sw_core_x], y=[core_y, sw_core_y],
            mode="lines", line=dict(width=3, color=FICOM_COLOR),
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
                x=[sw_core_x, sx], y=[sw_core_y, sy],
                mode="lines", line=dict(width=2, color=FICOM_COLOR),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=[sx], y=[sy],
                mode="markers+text",
                marker=dict(size=18, symbol="square", color="green"),
                text=[fs["name"]], textposition="bottom center",
                showlegend=False
            ))

            cam_positions = [(sx + 0.35, sy + 0.12), (sx + 0.35, sy - 0.12)]
            for (cx, cy) in cam_positions:
                fig.add_trace(go.Scatter(
                    x=[sx, cx], y=[sy, cy],
                    mode="lines", line=dict(width=1.8, dash="dot", color="gray"),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=[cx], y=[cy],
                    mode="markers+text",
                    marker=dict(size=12, symbol="triangle-up", color="black"),
                    text=[f"Cam {cam_index}"], textposition="top center",
                    showlegend=False
                ))
                cam_index += 1

        fig.update_layout(
            title="Topología Punto a Punto",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
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
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)

        switch_x = radius * np.cos(angles)
        switch_y = radius * np.sin(angles) + 0.1

        fig = go.Figure()

        for i in range(n):
            x0, y0 = switch_x[i], switch_y[i]
            x1, y1 = switch_x[(i+1) % n], switch_y[(i+1) % n]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines", line=dict(width=2, color=FICOM_COLOR),
                showlegend=False
            ))

        fig.add_trace(go.Scatter(
            x=switch_x, y=switch_y,
            mode="markers+text",
            marker=dict(size=16, symbol="square", color="royalblue"),
            text=[f"TR01-SW0{i+1}" for i in range(n)],
            textposition="top center",
            showlegend=False
        ))

        fig.update_layout(
            title="Topología en Anillo",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
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
            x=[core_x], y=[core_y],
            mode="markers+text",
            marker=dict(size=18, symbol="circle", color="red"),
            text=["CORE / NVR"], textposition="bottom center",
            showlegend=False
        ))

        node_x, node_y = -0.3, 0.5
        fig.add_trace(go.Scatter(
            x=[node_x], y=[node_y],
            mode="markers+text",
            marker=dict(size=18, symbol="square", color="royalblue"),
            text=["Nodo FTTN"], textposition="bottom center",
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[core_x, node_x], y=[core_y, node_y],
            mode="lines", line=dict(width=3, color=FICOM_COLOR),
            showlegend=False
        ))

        sw_positions = [(0.3, 0.8), (0.3, 0.5), (0.3, 0.2)]
        for i, (sx, sy) in enumerate(sw_positions, start=1):
            fig.add_trace(go.Scatter(
                x=[node_x, sx], y=[node_y, sy],
                mode="lines", line=dict(width=2, color=FICOM_COLOR),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[sx], y=[sy],
                mode="markers+text",
                marker=dict(size=16, symbol="square", color="green"),
                text=[f"TR01-SW0{i+1}-ND0{i}"], textposition="bottom center",
                showlegend=False
            ))

        fig.update_layout(
            title="Topología FTTN",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor="white",
            height=400
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    return go.Figure()


# =======================================================
# RECTÁNGULO Y ODF
# =======================================================
def _add_rectangle(fig, x0, y0, x1, y1, label, color="#FFFFFF", line_color="#000000"):
    fig.add_shape(
        type="rect",
        x0=x0, y0=y0, x1=x1, y1=y1,
        line=dict(color=line_color, width=1),
        fillcolor=color
    )
    fig.add_annotation(
        x=(x0 + x1)/2,
        y=y1 + 0.01,
        text=label,
        showarrow=False,
        font=dict(size=10)
    )


def _add_odf(fig, x0, y0, width=0.18, height=0.06, label="ODF", ports=12):
    x1 = x0 + width
    y1 = y0 + height

    fig.add_shape(
        type="rect",
        x0=x0, y0=y0,
        x1=x1, y1=y1,
        line=dict(color="black"),
        fillcolor="#E8E8E8"
    )

    fig.add_annotation(
        x=(x0 + x1)/2,
        y=y1 - 0.01,
        text=label,
        showarrow=False,
        font=dict(size=9)
    )

    port_dx = width / ports
    ports_xy = []

    for i in range(ports):
        px0 = x0 + i * port_dx
        px1 = px0 + port_dx * 0.8
        py0 = y0 + 0.005
        py1 = py0 + 0.02

        fig.add_shape(
            type="rect",
            x0=px0, y0=py0,
            x1=px1, y1=py1,
            line=dict(color="black", width=1),
            fillcolor="white"
        )
        ports_xy.append(((px0 + px1)/2, (py0 + py1)/2))

    return ports_xy
# ================================================================
# SWITCH CORE / NVR
# ================================================================
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
        x=(x0+x1)/2,
        y=y1 - 0.015,
        text=f"{label}<br>8 × SFP",
        showarrow=False,
        font=dict(color="white", size=9)
    )

    sfp_positions = []
    port_w = width / 10
    gap = (width - (8*port_w)) / 9

    px = x0 + gap
    py0 = y0 + 0.01
    py1 = py0 + 0.02

    for _ in range(8):
        fig.add_shape(
            type="rect",
            x0=px, y0=py0, x1=px+port_w, y1=py1,
            line=dict(color="white", width=1),
            fillcolor="gray"
        )
        sfp_positions.append((px + port_w/2, (py0 + py1)/2))
        px += port_w + gap

    return sfp_positions


# ================================================================
# CABLES — SEGMENTOS RECTOS
# ================================================================
def _add_cable_segments(fig, segments, width=2, color="#E3D873"):
    trace_ids = []
    for (p0, p1) in segments:
        (x0, y0) = p0
        (x1, y1) = p1

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=width),
            showlegend=False
        ))
        trace_ids.append(len(fig.data) - 1)

    return trace_ids


# ================================================================
# CABLES COMPLETOS — TRONCAL, TR→SW, CORE→SW
# ================================================================
def _build_all_cables(fig):
    fig._cables = {}

    odf_tr      = fig._odf_troncal
    odf_int     = fig._odf_interconexion
    odf_core    = fig._odf_core
    sfp         = fig._sfp_ports

    p_tr1 = odf_tr[0][0]
    p_tr2 = odf_tr[1][0]
    p_tr3 = odf_tr[2][0]

    p_int1 = odf_int[0][0]
    p_int2 = odf_int[1][0]
    p_int3 = odf_int[2][0]

    p_core1 = odf_core[0]
    p_core2 = odf_core[1]
    p_core3 = odf_core[2]

    sfp1, sfp2, sfp3 = sfp[0], sfp[1], sfp[2]

    def rect_path(a, b):
        (x0, y0) = a
        (x1, y1) = b
        xm = (x0 + x1) / 2
        return [
            ((x0, y0), (xm, y0)),
            ((xm, y0), (xm, y1)),
            ((xm, y1), (x1, y1))
        ]

    cables_tr1 = rect_path(p_tr1, p_int1)
    cables_tr2 = rect_path(p_tr2, p_int2)
    cables_tr3 = rect_path(p_tr3, p_int3)

    fig._cables["troncal_1"] = {
        "segments": cables_tr1,
        "trace_ids": _add_cable_segments(fig, cables_tr1, width=4)
    }
    fig._cables["troncal_2"] = {
        "segments": cables_tr2,
        "trace_ids": _add_cable_segments(fig, cables_tr2, width=4)
    }
    fig._cables["troncal_3"] = {
        "segments": cables_tr3,
        "trace_ids": _add_cable_segments(fig, cables_tr3, width=4)
    }

    def h_path(a, b): return [(a, b)]

    fig._cables["patch_tr_1"] = {
        "segments": h_path(p_int1, sfp1),
        "trace_ids": _add_cable_segments(fig, h_path(p_int1, sfp1), width=2)
    }
    fig._cables["patch_tr_2"] = {
        "segments": h_path(p_int2, sfp2),
        "trace_ids": _add_cable_segments(fig, h_path(p_int2, sfp2), width=2)
    }
    fig._cables["patch_tr_3"] = {
        "segments": h_path(p_int3, sfp3),
        "trace_ids": _add_cable_segments(fig, h_path(p_int3, sfp3), width=2)
    }

    fig._cables["core_1"] = {
        "segments": h_path(p_core1, sfp1),
        "trace_ids": _add_cable_segments(fig, h_path(p_core1, sfp1), width=2)
    }
    fig._cables["core_2"] = {
        "segments": h_path(p_core2, sfp2),
        "trace_ids": _add_cable_segments(fig, h_path(p_core2, sfp2), width=2)
    }
    fig._cables["core_3"] = {
        "segments": h_path(p_core3, sfp3),
        "trace_ids": _add_cable_segments(fig, h_path(p_core3, sfp3), width=2)
    }


# ================================================================
# ANIMACIÓN COMPLETA
# ================================================================
def _build_animation(fig):
    frames = []
    cables = fig._cables
    cable_keys = list(cables.keys())

    highlight = "#FFD700"
    normal    = "#E3D873"

    for step_idx, key in enumerate(cable_keys):
        frame_data = []
        for k2 in cable_keys:
            for trace_id in cables[k2]["trace_ids"]:
                frame_data.append({
                    "line": {
                        "color": highlight if k2 == key else normal,
                        "width": 4 if "troncal" in k2 else 2
                    }
                })
        frames.append(go.Frame(data=frame_data, name=f"step_{step_idx}"))

    fig.frames = frames

    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "x": 0.10, "y": -0.12,
            "buttons": [
                {
                    "label": "▶ Reproducir",
                    "method": "animate",
                    "args": [
                        None,
                        {"frame": {"duration": 800, "redraw": True},
                         "fromcurrent": True,
                         "transition": {"duration": 200}}
                    ]
                },
                {
                    "label": "⏹ Reset",
                    "method": "animate",
                    "args": [["step_0"], {"frame": {"duration": 0, "redraw": True}}]
                }
            ]
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {"label": f"{i+1}",
                 "method": "animate",
                 "args": [[f"step_{i}"], {"frame": {"duration": 0, "redraw": True}}]}
                for i in range(len(cable_keys))
            ],
            "x": 0.25, "y": -0.12, "len": 0.60
        }]
    )


# ================================================================
# RACKS — DIAGRAMA COMPLETO
# ================================================================
def create_rack_connection_diagram():
    fig = go.Figure()

    rack_w = 0.22
    rack_h = 0.62

    rack_troncal_x0 = 0.05
    rack_int_x0     = 0.38
    rack_core_x0    = 0.70
    rack_y0 = 0.15

    # ----------------- RACKS -----------------
    fig.add_shape(
        type="rect",
        x0=rack_troncal_x0, y0=rack_y0,
        x1=rack_troncal_x0 + rack_w, y1=rack_y0 + rack_h,
        line=dict(color="black", width=2), fillcolor="#F7F7F7"
    )
    fig.add_annotation(
        x=rack_troncal_x0 + rack_w/2, y=rack_y0 + rack_h + 0.03,
        text="RACK TRONCAL", showarrow=False
    )

    fig.add_shape(
        type="rect",
        x0=rack_int_x0, y0=rack_y0,
        x1=rack_int_x0 + rack_w, y1=rack_y0 + rack_h,
        line=dict(color="black", width=2), fillcolor="#F7F7F7"
    )
    fig.add_annotation(
        x=rack_int_x0 + rack_w/2, y=rack_y0 + rack_h + 0.03,
        text="RACK DE INTERCONEXIÓN", showarrow=False
    )

    fig.add_shape(
        type="rect",
        x0=rack_core_x0, y0=rack_y0,
        x1=rack_core_x0 + rack_w, y1=rack_y0 + rack_h,
        line=dict(color="black", width=2), fillcolor="#F7F7F7"
    )
    fig.add_annotation(
        x=rack_core_x0 + rack_w/2, y=rack_y0 + rack_h + 0.03,
        text="RACK CORE / NVR", showarrow=False
    )

    # ----------------- ODFS -----------------
    odf_gap = 0.09
    odf_h = 0.06

    odf1_y = rack_y0 + rack_h - (odf_h + odf_gap)*1
    odf2_y = rack_y0 + rack_h - (odf_h + odf_gap)*2
    odf3_y = rack_y0 + rack_h - (odf_h + odf_gap)*3

    odf1_tr = _add_odf(fig, rack_troncal_x0 + 0.02, odf1_y, label="ODF TRONCAL 1")
    odf2_tr = _add_odf(fig, rack_troncal_x0 + 0.02, odf2_y, label="ODF TRONCAL 2")
    odf3_tr = _add_odf(fig, rack_troncal_x0 + 0.02, odf3_y, label="ODF TRONCAL 3")

    odf1_int = _add_odf(fig, rack_int_x0 + 0.02, odf1_y, label="ODF TRONCAL 1")
    odf2_int = _add_odf(fig, rack_int_x0 + 0.02, odf2_y, label="ODF TRONCAL 2")
    odf3_int = _add_odf(fig, rack_int_x0 + 0.02, odf3_y, label="ODF TRONCAL 3")

    odf_core_y = odf1_y
    odf_core = _add_odf(fig, rack_core_x0 + 0.02, odf_core_y, label="ODF CORE – NVR")

    fig._odf_troncal = [odf1_tr, odf2_tr, odf3_tr]
    fig._odf_interconexion = [odf1_int, odf2_int, odf3_int]
    fig._odf_core = odf_core

    # ----------------- SWITCH -----------------
    sw_x = rack_core_x0 + 0.02
    sw_y = odf_core_y - 0.10

    sfp_ports = _add_switch(fig, sw_x, sw_y)
    fig._sfp_ports = sfp_ports

    # ----------------- CABLES + ANIMACIÓN -----------------
    _build_all_cables(fig)
    _build_animation(fig)

    return fig
    # ================================================================
# SWITCH CORE / NVR
# ================================================================
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
        x=(x0+x1)/2,
        y=y1 - 0.015,
        text=f"{label}<br>8 × SFP",
        showarrow=False,
        font=dict(color="white", size=9)
    )

    sfp_positions = []
    port_w = width / 10
    gap = (width - (8*port_w)) / 9

    px = x0 + gap
    py0 = y0 + 0.01
    py1 = py0 + 0.02

    for _ in range(8):
        fig.add_shape(
            type="rect",
            x0=px, y0=py0, x1=px+port_w, y1=py1,
            line=dict(color="white", width=1),
            fillcolor="gray"
        )
        sfp_positions.append((px + port_w/2, (py0 + py1)/2))
        px += port_w + gap

    return sfp_positions


# ================================================================
# CABLES — SEGMENTOS RECTOS
# ================================================================
def _add_cable_segments(fig, segments, width=2, color="#E3D873"):
    trace_ids = []
    for (p0, p1) in segments:
        (x0, y0) = p0
        (x1, y1) = p1

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=width),
            showlegend=False
        ))
        trace_ids.append(len(fig.data) - 1)

    return trace_ids


# ================================================================
# CABLES COMPLETOS — TRONCAL, TR→SW, CORE→SW
# ================================================================
def _build_all_cables(fig):
    fig._cables = {}

    odf_tr      = fig._odf_troncal
    odf_int     = fig._odf_interconexion
    odf_core    = fig._odf_core
    sfp         = fig._sfp_ports

    p_tr1 = odf_tr[0][0]
    p_tr2 = odf_tr[1][0]
    p_tr3 = odf_tr[2][0]

    p_int1 = odf_int[0][0]
    p_int2 = odf_int[1][0]
    p_int3 = odf_int[2][0]

    p_core1 = odf_core[0]
    p_core2 = odf_core[1]
    p_core3 = odf_core[2]

    sfp1, sfp2, sfp3 = sfp[0], sfp[1], sfp[2]

    def rect_path(a, b):
        (x0, y0) = a
        (x1, y1) = b
        xm = (x0 + x1) / 2
        return [
            ((x0, y0), (xm, y0)),
            ((xm, y0), (xm, y1)),
            ((xm, y1), (x1, y1))
        ]

    cables_tr1 = rect_path(p_tr1, p_int1)
    cables_tr2 = rect_path(p_tr2, p_int2)
    cables_tr3 = rect_path(p_tr3, p_int3)

    fig._cables["troncal_1"] = {
        "segments": cables_tr1,
        "trace_ids": _add_cable_segments(fig, cables_tr1, width=4)
    }
    fig._cables["troncal_2"] = {
        "segments": cables_tr2,
        "trace_ids": _add_cable_segments(fig, cables_tr2, width=4)
    }
    fig._cables["troncal_3"] = {
        "segments": cables_tr3,
        "trace_ids": _add_cable_segments(fig, cables_tr3, width=4)
    }

    def h_path(a, b): return [(a, b)]

    fig._cables["patch_tr_1"] = {
        "segments": h_path(p_int1, sfp1),
        "trace_ids": _add_cable_segments(fig, h_path(p_int1, sfp1), width=2)
    }
    fig._cables["patch_tr_2"] = {
        "segments": h_path(p_int2, sfp2),
        "trace_ids": _add_cable_segments(fig, h_path(p_int2, sfp2), width=2)
    }
    fig._cables["patch_tr_3"] = {
        "segments": h_path(p_int3, sfp3),
        "trace_ids": _add_cable_segments(fig, h_path(p_int3, sfp3), width=2)
    }

    fig._cables["core_1"] = {
        "segments": h_path(p_core1, sfp1),
        "trace_ids": _add_cable_segments(fig, h_path(p_core1, sfp1), width=2)
    }
    fig._cables["core_2"] = {
        "segments": h_path(p_core2, sfp2),
        "trace_ids": _add_cable_segments(fig, h_path(p_core2, sfp2), width=2)
    }
    fig._cables["core_3"] = {
        "segments": h_path(p_core3, sfp3),
        "trace_ids": _add_cable_segments(fig, h_path(p_core3, sfp3), width=2)
    }


# ================================================================
# ANIMACIÓN COMPLETA
# ================================================================
def _build_animation(fig):
    frames = []
    cables = fig._cables
    cable_keys = list(cables.keys())

    highlight = "#FFD700"
    normal    = "#E3D873"

    for step_idx, key in enumerate(cable_keys):
        frame_data = []
        for k2 in cable_keys:
            for trace_id in cables[k2]["trace_ids"]:
                frame_data.append({
                    "line": {
                        "color": highlight if k2 == key else normal,
                        "width": 4 if "troncal" in k2 else 2
                    }
                })
        frames.append(go.Frame(data=frame_data, name=f"step_{step_idx}"))

    fig.frames = frames

    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "x": 0.10, "y": -0.12,
            "buttons": [
                {
                    "label": "▶ Reproducir",
                    "method": "animate",
                    "args": [
                        None,
                        {"frame": {"duration": 800, "redraw": True},
                         "fromcurrent": True,
                         "transition": {"duration": 200}}
                    ]
                },
                {
                    "label": "⏹ Reset",
                    "method": "animate",
                    "args": [["step_0"], {"frame": {"duration": 0, "redraw": True}}]
                }
            ]
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {"label": f"{i+1}",
                 "method": "animate",
                 "args": [[f"step_{i}"], {"frame": {"duration": 0, "redraw": True}}]}
                for i in range(len(cable_keys))
            ],
            "x": 0.25, "y": -0.12, "len": 0.60
        }]
    )


# ================================================================
# RACKS — DIAGRAMA COMPLETO
# ================================================================
def create_rack_connection_diagram():
    fig = go.Figure()

    rack_w = 0.22
    rack_h = 0.62

    rack_troncal_x0 = 0.05
    rack_int_x0     = 0.38
    rack_core_x0    = 0.70
    rack_y0 = 0.15

    # ----------------- RACKS -----------------
    fig.add_shape(
        type="rect",
        x0=rack_troncal_x0, y0=rack_y0,
        x1=rack_troncal_x0 + rack_w, y1=rack_y0 + rack_h,
        line=dict(color="black", width=2), fillcolor="#F7F7F7"
    )
    fig.add_annotation(
        x=rack_troncal_x0 + rack_w/2, y=rack_y0 + rack_h + 0.03,
        text="RACK TRONCAL", showarrow=False
    )

    fig.add_shape(
        type="rect",
        x0=rack_int_x0, y0=rack_y0,
        x1=rack_int_x0 + rack_w, y1=rack_y0 + rack_h,
        line=dict(color="black", width=2), fillcolor="#F7F7F7"
    )
    fig.add_annotation(
        x=rack_int_x0 + rack_w/2, y=rack_y0 + rack_h + 0.03,
        text="RACK DE INTERCONEXIÓN", showarrow=False
    )

    fig.add_shape(
        type="rect",
        x0=rack_core_x0, y0=rack_y0,
        x1=rack_core_x0 + rack_w, y1=rack_y0 + rack_h,
        line=dict(color="black", width=2), fillcolor="#F7F7F7"
    )
    fig.add_annotation(
        x=rack_core_x0 + rack_w/2, y=rack_y0 + rack_h + 0.03,
        text="RACK CORE / NVR", showarrow=False
    )

    # ----------------- ODFS -----------------
    odf_gap = 0.09
    odf_h = 0.06

    odf1_y = rack_y0 + rack_h - (odf_h + odf_gap)*1
    odf2_y = rack_y0 + rack_h - (odf_h + odf_gap)*2
    odf3_y = rack_y0 + rack_h - (odf_h + odf_gap)*3

    odf1_tr = _add_odf(fig, rack_troncal_x0 + 0.02, odf1_y, label="ODF TRONCAL 1")
    odf2_tr = _add_odf(fig, rack_troncal_x0 + 0.02, odf2_y, label="ODF TRONCAL 2")
    odf3_tr = _add_odf(fig, rack_troncal_x0 + 0.02, odf3_y, label="ODF TRONCAL 3")

    odf1_int = _add_odf(fig, rack_int_x0 + 0.02, odf1_y, label="ODF TRONCAL 1")
    odf2_int = _add_odf(fig, rack_int_x0 + 0.02, odf2_y, label="ODF TRONCAL 2")
    odf3_int = _add_odf(fig, rack_int_x0 + 0.02, odf3_y, label="ODF TRONCAL 3")

    odf_core_y = odf1_y
    odf_core = _add_odf(fig, rack_core_x0 + 0.02, odf_core_y, label="ODF CORE – NVR")

    fig._odf_troncal = [odf1_tr, odf2_tr, odf3_tr]
    fig._odf_interconexion = [odf1_int, odf2_int, odf3_int]
    fig._odf_core = odf_core

    # ----------------- SWITCH -----------------
    sw_x = rack_core_x0 + 0.02
    sw_y = odf_core_y - 0.10

    sfp_ports = _add_switch(fig, sw_x, sw_y)
    fig._sfp_ports = sfp_ports

    # ----------------- CABLES + ANIMACIÓN -----------------
    _build_all_cables(fig)
    _build_animation(fig)

    return fig
