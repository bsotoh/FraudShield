"""
FraudShield | Centro de decisiones de riesgo.

Ejecutar:
    python -m streamlit run app.py

Los modelos se entrenan en el notebook.
Esta aplicación utiliza resultados agregados previamente exportados.
"""

from pathlib import Path
from dataclasses import dataclass

import html
import json
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# 1. CONFIGURACIÓN GENERAL
# =============================================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(
    os.environ.get(
        "FRAUDSHIELD_DATA_DIR",
        str(ROOT / "dashboard_data")
    )
)

NAVY = "#142C43"
TEAL = "#007F73"
BLUE = "#3674C4"
RED = "#BC4651"
AMBER = "#AC690B"
MUTED = "#64768A"

SECTIONS = [
    "Panorama",
    "Simulador",
    "Estrategias",
    "Sensibilidad",
    "Modelo y método"
]


# =============================================================================
# 2. DISEÑO VISUAL
# =============================================================================

CSS = """
<style>
.stApp {
    background: #F4F7FA;
    color: #142C43;
}

.block-container {
    max-width: 1460px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: #EAF0F5;
    border-right: 1px solid #D8E2EB;
}

h1, h2, h3 {
    letter-spacing: -.035em;
    color: #142C43;
}

h1 {
    font-size: 2.6rem !important;
    line-height: 1.1 !important;
}

h2 {
    font-size: 1.55rem !important;
}

h3 {
    font-size: 1.12rem !important;
}

.brand {
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -.06em;
    margin-bottom: .1rem;
}

.brand span {
    color: #007F73;
}

.brand-sub {
    font-size: .72rem;
    letter-spacing: .16em;
    color: #52677C;
    margin-bottom: 1.6rem;
}

.hero {
    background: #142C43;
    border-radius: 20px;
    padding: 30px 34px;
    margin: 0 0 22px;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 28px;
    border-left: 5px solid #35C8AE;
}

.hero h1 {
    color: white !important;
    margin: 9px 0 !important;
    padding: 0 !important;
    font-size: 2.3rem !important;
}

.hero p {
    color: #CAD8E6;
    max-width: 720px;
    margin: 0;
    font-size: 1rem;
    line-height: 1.6;
}

.eyebrow {
    font-size: .7rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    font-weight: 700;
    color: #91E1D2;
}

.hero-side {
    border-left: 1px solid #395268;
    padding-left: 26px;
    min-width: 170px;
}

.hero-side strong {
    font-size: 2rem;
    font-weight: 700;
    display: block;
    color: #91E1D2;
}

.hero-side small {
    color: #CAD8E6;
    font-size: .8rem;
}

.kpi {
    min-height: 158px;
    background: white;
    border: 1px solid #DDE6EE;
    border-radius: 15px;
    padding: 19px 20px;
    margin: 0 0 10px;
}

.kpi .label {
    font-size: .8rem;
    color: #53677B;
    font-weight: 600;
    min-height: 20px;
}

.kpi .value {
    font-size: clamp(1.3rem, 2.1vw, 2rem);
    font-weight: 750;
    letter-spacing: -.045em;
    margin: 10px 0 7px;
    line-height: 1.15;
    overflow-wrap: anywhere;
}

.kpi .note {
    font-size: .75rem;
    color: #61758A;
    line-height: 1.45;
}

.kpi.positive {
    border-top: 3px solid #007F73;
}

.kpi.negative {
    border-top: 3px solid #BC4651;
}

.kpi.positive .value {
    color: #007F73;
}

.kpi.negative .value {
    color: #BC4651;
}

.insight {
    border-radius: 13px;
    background: #E4F3EF;
    border: 1px solid #B8DED3;
    padding: 17px 21px;
    line-height: 1.65;
    margin: 12px 0 20px;
}

.insight.negative {
    background: #F9ECEE;
    border-color: #E9C5CA;
}

.step {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .14em;
    color: #007F73;
    text-transform: uppercase;
}

.footer {
    margin-top: 22px;
    padding-top: 16px;
    border-top: 1px solid #D8E2EB;
    font-size: .74rem;
    color: #61758A;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 15px !important;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
}

div.stButton > button {
    border-radius: 9px;
    font-weight: 600;
}

@media(max-width: 750px) {
    .hero {
        padding: 24px;
        display: block;
    }

    .hero-side {
        border-left: 0;
        border-top: 1px solid #395268;
        padding: 15px 0 0;
        margin-top: 15px;
    }

    .hero h1 {
        font-size: 1.9rem !important;
    }

    .block-container {
        padding: 1.1rem;
    }

    .kpi {
        min-height: 125px;
    }
}

@media print {
    [data-testid="stSidebar"],
    header,
    footer,
    [data-testid="stToolbar"] {
        display: none !important;
    }

    .block-container {
        padding: 0 !important;
    }

    .hero, .kpi {
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
}
</style>
"""


# =============================================================================
# 3. FUNCIONES DE FORMATO
# =============================================================================

def fmt(value, digits=0):
    """Formato numérico con punto de miles y coma decimal."""
    if value is None or not np.isfinite(value):
        return "No definida"

    return (
        f"{float(value):,.{digits}f}"
        .replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )


def money(value):
    return f"${fmt(value)}"


def percent(value):
    if value is None:
        return "No definida"

    return f"{fmt(100 * value, 2)} %"


def card(label, value, note="", tone=""):
    """Tarjeta de indicador."""
    st.markdown(
        f"""
        <div class="kpi {tone}">
            <div class="label">{html.escape(label)}</div>
            <div class="value">{html.escape(str(value))}</div>
            <div class="note">{html.escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def chart_style(fig, height=370, ytitle=None):
    """Aplicar un estilo uniforme a los gráficos."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color=NAVY
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=16, r=20, t=35, b=25),
        legend=dict(
            orientation="h",
            y=-0.24,
            x=0,
            font=dict(size=11)
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12
        ),
        separators=",."
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False
    )

    fig.update_yaxes(
        gridcolor="#E5ECF2",
        zeroline=False,
        title=ytitle
    )

    return fig


def plot(fig, key):
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "scale": 2
            }
        }
    )


# =============================================================================
# 4. CARGA Y VALIDACIÓN DE DATOS
# =============================================================================

@dataclass
class Bundle:
    meta: dict
    curvas: pd.DataFrame
    ttcv: pd.DataFrame
    distribucion: dict
    modelos: pd.DataFrame
    refinamiento: pd.DataFrame


def validate_costs(c_fp, c_fn):
    validos = all(
        math.isfinite(float(x)) and float(x) >= 0
        for x in [c_fp, c_fn]
    )

    if not validos:
        raise ValueError(
            "Los costos deben ser números finitos no negativos."
        )

    if c_fp + c_fn == 0:
        raise ValueError(
            "Con ambos costos en cero, todos los umbrales tienen "
            "costo cero; no existe un óptimo económico único."
        )


def load_bundle(directory):
    path = Path(directory)

    required = [
        "metadatos.json",
        "curvas_umbrales.csv",
        "curva_ttcv.csv",
        "distribucion_test.npz"
    ]

    missing = [
        name
        for name in required
        if not (path / name).is_file()
    ]

    if missing:
        raise ValueError(
            "Faltan archivos en dashboard_data: "
            + ", ".join(missing)
        )

    meta = json.loads(
        (path / "metadatos.json").read_text(encoding="utf-8")
    )

    count_keys = [
        "n_train_0",
        "n_train_1",
        "n_test_0",
        "n_test_1"
    ]

    for key in count_keys:
        value = meta.get(key)

        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"Metadato inválido: {key}."
            )

    validate_costs(
        meta["c_fp_base"],
        meta["c_fn_base"]
    )

    if not 0 <= float(meta["umbral_cv_base"]) <= 1:
        raise ValueError("Umbral base inválido.")

    curves = pd.read_csv(
        path / "curvas_umbrales.csv"
    )

    tuning = pd.read_csv(
        path / "curva_ttcv.csv"
    )

    estructuras = [
        (
            curves,
            [
                "umbral",
                "fp_cv",
                "fn_cv",
                "fp_test",
                "fn_test"
            ]
        ),
        (
            tuning,
            [
                "umbral",
                "fp_medio",
                "fn_medio"
            ]
        )
    ]

    for frame, cols in estructuras:
        if not set(cols).issubset(frame.columns) or len(frame) < 2:
            raise ValueError(
                "Una curva no tiene las columnas o filas necesarias."
            )

        values = frame[cols].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "Las curvas contienen valores vacíos o no finitos."
            )

        if (values < 0).any():
            raise ValueError(
                "Las curvas contienen valores negativos."
            )

        if (
            frame["umbral"].max() > 1
            or np.any(np.diff(frame["umbral"]) <= 0)
        ):
            raise ValueError(
                "Los umbrales deben ser únicos, ordenados "
                "y estar entre 0 y 1."
            )

    folds = int(meta.get("n_folds", 3))

    if folds < 2:
        raise ValueError(
            "Se requieren al menos dos folds de validación."
        )

    for suffix in ["cv", "test"]:
        split = "train" if suffix == "cv" else "test"

        for error, clase in [("fp", 0), ("fn", 1)]:
            values = curves[f"{error}_{suffix}"].to_numpy()

            if (
                np.any(values != np.floor(values))
                or np.any(values > meta[f"n_{split}_{clase}"])
            ):
                raise ValueError(
                    "Conteos de errores incompatibles "
                    "con el tamaño del conjunto."
                )

            if error == "fp" and np.any(np.diff(values) > 0):
                raise ValueError(
                    "La curva de falsos positivos no es monótona."
                )

            if error == "fn" and np.any(np.diff(values) < 0):
                raise ValueError(
                    "La curva de falsos negativos no es monótona."
                )

    for col, clase in [("fp_medio", 0), ("fn_medio", 1)]:
        limite = math.ceil(
            meta[f"n_train_{clase}"] / folds
        )

        if tuning[col].max() > limite:
            raise ValueError(
                "Curva de validación incompatible con los folds."
            )

    with np.load(
        path / "distribucion_test.npz",
        allow_pickle=False
    ) as data:
        dist = {
            key: data[key].copy()
            for key in ["p0", "n0", "p1", "n1"]
        }

    for clase in [0, 1]:
        probs = dist[f"p{clase}"]
        freq = dist[f"n{clase}"]

        if (
            probs.ndim != 1
            or freq.ndim != 1
            or len(probs) != len(freq)
            or not len(probs)
        ):
            raise ValueError(
                "Distribución de prueba inválida."
            )

        if (
            not np.isfinite(probs).all()
            or not np.isfinite(freq).all()
        ):
            raise ValueError(
                "Distribución de prueba con valores no finitos."
            )

        if (
            (probs < 0).any()
            or (probs > 1).any()
            or np.any(np.diff(probs) <= 0)
        ):
            raise ValueError(
                "Probabilidades de prueba inválidas o desordenadas."
            )

        if (
            np.any(freq < 1)
            or np.any(freq != np.floor(freq))
            or int(freq.sum()) != meta[f"n_test_{clase}"]
        ):
            raise ValueError(
                "Frecuencias de prueba incompatibles "
                "con los metadatos."
            )

        dist[f"cum{clase}"] = np.r_[
            0,
            np.cumsum(freq, dtype=np.int64)
        ]

    weighted = meta.get("modelo_con_pesos", {})

    if any(
        not isinstance(weighted.get(k), int)
        or weighted[k] < 0
        for k in ["tn", "fp", "fn", "tp"]
    ):
        raise ValueError(
            "La matriz del modelo con pesos está incompleta."
        )

    if (
        weighted["tn"] + weighted["fp"] != meta["n_test_0"]
        or weighted["tp"] + weighted["fn"] != meta["n_test_1"]
    ):
        raise ValueError(
            "El modelo con pesos fue evaluado en otro conjunto."
        )

    def optional_csv(name):
        archivo = path / name

        if archivo.exists():
            return pd.read_csv(archivo)

        return pd.DataFrame()

    result = Bundle(
        meta=meta,
        curvas=curves,
        ttcv=tuning,
        distribucion=dist,
        modelos=optional_csv("modelos.csv"),
        refinamiento=optional_csv("refinamiento.csv")
    )

    reproduced = optimal_threshold(
        result,
        meta["c_fp_base"],
        meta["c_fn_base"]
    )

    if not np.isclose(
        reproduced,
        meta["umbral_cv_base"],
        atol=1e-12,
        rtol=0
    ):
        raise ValueError(
            "Las curvas no reproducen el umbral base del notebook."
        )

    fp, fn = test_errors(
        result,
        curves["umbral"].to_numpy()
    )

    if (
        not np.array_equal(fp, curves["fp_test"])
        or not np.array_equal(fn, curves["fn_test"])
    ):
        raise ValueError(
            "Los archivos de prueba no corresponden "
            "a la misma exportación."
        )

    return result


@st.cache_data(show_spinner=False)
def cached_bundle(directory, signature):
    return load_bundle(directory)


# =============================================================================
# 5. CÁLCULOS DE UMBRALES, MÉTRICAS Y COSTOS
# =============================================================================

def test_errors(bundle, thresholds):
    """
    Predicción positiva cuando probabilidad >= umbral.

    side='left' conserva correctamente los casos empatados
    exactamente con el umbral.
    """
    thresholds = np.asarray(
        thresholds,
        dtype=float
    )

    if (
        not np.isfinite(thresholds).all()
        or np.any((thresholds < 0) | (thresholds > 1))
    ):
        raise ValueError(
            "El umbral debe estar entre 0 y 1."
        )

    dist = bundle.distribucion

    index_0 = np.searchsorted(
        dist["p0"],
        thresholds,
        side="left"
    )

    index_1 = np.searchsorted(
        dist["p1"],
        thresholds,
        side="left"
    )

    fp = (
        bundle.meta["n_test_0"]
        - dist["cum0"][index_0]
    )

    fn = dist["cum1"][index_1]

    return fp, fn


def optimal_threshold(bundle, c_fp, c_fn):
    """Seleccionar el umbral exclusivamente con validación."""
    validate_costs(c_fp, c_fn)

    costs = (
        c_fp * bundle.ttcv["fp_medio"].to_numpy()
        + c_fn * bundle.ttcv["fn_medio"].to_numpy()
    )

    # En empates se utiliza el primer mínimo, igual que TTCV.
    index = int(np.argmin(costs))

    return float(
        bundle.ttcv["umbral"].iloc[index]
    )


def metrics(
    bundle,
    threshold,
    c_fp,
    c_fn,
    weighted=False,
    all_negative=False
):
    validate_costs(c_fp, c_fn)

    n0 = bundle.meta["n_test_0"]
    n1 = bundle.meta["n_test_1"]

    if weighted:
        fp = bundle.meta["modelo_con_pesos"]["fp"]
        fn = bundle.meta["modelo_con_pesos"]["fn"]

    elif all_negative:
        fp = 0
        fn = n1

    else:
        fp_value, fn_value = test_errors(
            bundle,
            threshold
        )

        fp = int(fp_value)
        fn = int(fn_value)

    tp = n1 - fn
    tn = n0 - fp

    total = fp * c_fp + fn * c_fn

    fp05, fn05 = test_errors(
        bundle,
        0.5
    )

    base = (
        int(fp05) * c_fp
        + int(fn05) * c_fn
    )

    saving = base - total
    alerts = tp + fp

    return {
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
        "cost": total,
        "base": base,
        "saving": saving,
        "saving_pct": saving / base if base else None,
        "recall": tp / n1 if n1 else None,
        "precision": tp / alerts if alerts else None,
        "f1": (
            2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn
            else None
        ),
        "alerts": alerts,
        "alert_rate": alerts / (n0 + n1)
    }


def strategy_table(bundle, c_fp, c_fn, manual):
    opt = optimal_threshold(
        bundle,
        c_fp,
        c_fn
    )

    theoretical = c_fp / (c_fp + c_fn)

    specifications = [
        (
            "Sin detección",
            "Regla: todo 0",
            None,
            False,
            True
        ),
        (
            "Tradicional",
            "LightGBM sin pesos",
            0.5,
            False,
            False
        ),
        (
            "Teórico",
            "LightGBM sin pesos",
            theoretical,
            False,
            False
        ),
        (
            "Óptimo CV",
            "LightGBM sin pesos",
            opt,
            False,
            False
        ),
        (
            "Con pesos",
            "LightGBM refinado con pesos",
            0.5,
            True,
            False
        ),
        (
            "Manual",
            "LightGBM sin pesos",
            manual,
            False,
            False
        )
    ]

    rows = []

    for name, model, threshold, weighted, all_negative in specifications:
        result = metrics(
            bundle,
            threshold,
            c_fp,
            c_fn,
            weighted=weighted,
            all_negative=all_negative
        )

        rows.append({
            "Estrategia": name,
            "Modelo": model,
            "Umbral": threshold,
            "Costo total (CLP)": result["cost"],
            "Ahorro (CLP)": result["saving"],
            "Recall": result["recall"],
            "Precisión": result["precision"],
            "F1": result["f1"],
            "FP": result["fp"],
            "FN": result["fn"],
            "TP": result["tp"],
            "TN": result["tn"]
        })

    return pd.DataFrame(rows)


def sensitivity_table(bundle, c_fp, c_fn):
    rows = []

    for factor in [0.5, 0.75, 1.0, 1.5, 2.0]:
        scenario_fn = float(c_fn * factor)

        threshold = optimal_threshold(
            bundle,
            c_fp,
            scenario_fn
        )

        result = metrics(
            bundle,
            threshold,
            c_fp,
            scenario_fn
        )

        theoretical = c_fp / (c_fp + scenario_fn)

        theoretical_result = metrics(
            bundle,
            theoretical,
            c_fp,
            scenario_fn
        )

        rows.append({
            "Escenario": f"{fmt(factor, 2)}× FN",
            "Costo FP (CLP)": c_fp,
            "Costo FN (CLP)": scenario_fn,
            "Razón FN/FP": (
                scenario_fn / c_fp
                if c_fp
                else np.nan
            ),
            "Umbral óptimo CV": threshold,
            "Umbral teórico": theoretical,
            "Costo óptimo (CLP)": result["cost"],
            "Costo 0,5 (CLP)": result["base"],
            "Costo teórico (CLP)": theoretical_result["cost"],
            "Ahorro (CLP)": result["saving"],
            "FP": result["fp"],
            "FN": result["fn"]
        })

    return pd.DataFrame(rows)


# =============================================================================
# 6. GRÁFICOS
# =============================================================================

def cost_chart(
    bundle,
    c_fp,
    c_fn,
    current,
    opt,
    theoretical,
    split="test"
):
    marks = [
        (0.5, "Tradicional 0,5", MUTED),
        (theoretical, "Teórico", BLUE),
        (opt, "Óptimo CV", TEAL),
        (current, "Seleccionado", AMBER)
    ]

    if split == "test":
        grid = np.unique(
            np.r_[
                np.linspace(0, 1, 1601),
                [mark[0] for mark in marks]
            ]
        )

        fp, fn = test_errors(
            bundle,
            grid
        )

        values = fp * c_fp + fn * c_fn

        subtitle = "Costo en test (CLP)"

    else:
        grid = bundle.ttcv["umbral"].to_numpy()

        values = (
            (
                bundle.ttcv["fp_medio"] * c_fp
                + bundle.ttcv["fn_medio"] * c_fn
            ).to_numpy()
            * bundle.meta.get("n_folds", 3)
        )

        subtitle = "Costo CV agregado estimado (CLP)"

    fig = go.Figure(
        go.Scatter(
            x=grid,
            y=values,
            mode="lines",
            name="Costo total",
            line=dict(
                color=NAVY,
                width=2.4
            ),
            fill="tozeroy",
            fillcolor="rgba(20,44,67,.035)",
            hovertemplate=(
                "Umbral %{x:.6f}"
                "<br>Costo $%{y:,.0f} CLP"
                "<extra></extra>"
            )
        )
    )

    for threshold, name, color in marks:
        if split == "test":
            fp, fn = test_errors(
                bundle,
                threshold
            )

            value = float(
                fp * c_fp + fn * c_fn
            )

        else:
            value = float(
                np.interp(
                    threshold,
                    grid,
                    values
                )
            )

        fig.add_vline(
            x=threshold,
            line=dict(
                color=color,
                width=1,
                dash="dot"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[threshold],
                y=[value],
                mode="markers",
                name=f"{name} · {fmt(threshold, 6)}",
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(
                        color="white",
                        width=1
                    )
                ),
                hovertemplate=(
                    "%{x:.6f}"
                    "<br>$%{y:,.0f} CLP"
                    "<extra>" + name + "</extra>"
                )
            )
        )

    fig.update_xaxes(
        title="Umbral de decisión",
        range=[0, 1],
        tickformat=".2f"
    )

    fig.update_yaxes(
        tickprefix="$",
        tickformat="~s"
    )

    return chart_style(
        fig,
        height=400,
        ytitle=subtitle
    )


def confusion_chart(m, c_fp, c_fn):
    """Matriz de confusión con etiquetas breves y costos legibles."""

    cantidades = [
        [int(m["tn"]), int(m["fp"])],
        [int(m["fn"]), int(m["tp"])],
    ]

    costos = [
        [0, m["fp"] * c_fp],
        [m["fn"] * c_fn, 0],
    ]

    etiquetas = [
        ["Aprobación correcta", "Falsa alarma"],
        ["Fraude omitido", "Fraude detectado"],
    ]

    siglas = [
        ["TN", "FP"],
        ["FN", "TP"],
    ]

    colores_texto = [
        ["#166054", "#855300"],
        ["#A3263D", "#166054"],
    ]

    def numero(valor):
        return f"{valor:,.0f}".replace(",", ".")

    fig = go.Figure(
        go.Heatmap(
            z=[[0, 1], [2, 0]],
            x=[0, 1],
            y=[0, 1],
            zmin=0,
            zmax=2,
            colorscale=[
                [0.00, "#E5F3EE"],
                [0.49, "#E5F3EE"],
                [0.50, "#FFF1D6"],
                [0.74, "#FFF1D6"],
                [0.75, "#FBE7EC"],
                [1.00, "#FBE7EC"],
            ],
            showscale=False,
            xgap=8,
            ygap=8,
            customdata=[
                [
                    [cantidades[f][c], costos[f][c], siglas[f][c]]
                    for c in range(2)
                ]
                for f in range(2)
            ],
            hovertemplate=(
                "<b>%{customdata[2]}</b>"
                "<br>Casos: %{customdata[0]:,.0f}"
                "<br>Costo: $%{customdata[1]:,.0f} CLP"
                "<extra></extra>"
            ),
        )
    )

    for fila in range(2):
        for columna in range(2):
            color = colores_texto[fila][columna]

            # Cantidad de operaciones.
            fig.add_annotation(
                x=columna,
                y=fila,
                text=f"<b>{numero(cantidades[fila][columna])}</b>",
                showarrow=False,
                yshift=34,
                font=dict(size=22, color=color),
            )

            # Descripción corta y tipo de resultado.
            fig.add_annotation(
                x=columna,
                y=fila,
                text=(
                    f"{etiquetas[fila][columna]}"
                    f"<br><b>{siglas[fila][columna]}</b>"
                ),
                showarrow=False,
                yshift=-2,
                align="center",
                font=dict(size=11, color="#24374B"),
            )

            # Costo monetario, en una línea independiente.
            fig.add_annotation(
                x=columna,
                y=fila,
                text=f"${numero(costos[fila][columna])} CLP",
                showarrow=False,
                yshift=-40,
                font=dict(size=11, color=color),
            )

    fig.update_layout(
        height=390,
        autosize=True,
        margin=dict(l=70, r=8, t=12, b=65),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            color="#24374B",
            size=12,
        ),
        separators=",.",
        dragmode=False,
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Aprobar<br>(clase 0)", "Alertar<br>(clase 1)"],
        range=[-0.5, 1.5],
        title=dict(text="Decisión del modelo", standoff=12),
        tickfont=dict(size=11),
        tickangle=0,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        automargin=True,
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Legítima<br>(clase 0)", "Fraude<br>(clase 1)"],
        range=[1.5, -0.5],
        title=None,
        tickfont=dict(size=11),
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        automargin=True,
    )

    return fig


# =============================================================================
# 7. ELEMENTOS COMPARTIDOS DE LA INTERFAZ
# =============================================================================

def headline(section, bundle):
    subtitles = {
        "Panorama": (
            "Menos pérdidas. Decisiones informadas.",
            "Evalúa el costo del fraude y de las falsas alarmas "
            "sobre las mismas transacciones."
        ),
        "Simulador": (
            "El costo cambia. El umbral también.",
            "Ajusta los costos y compara tu decisión con la "
            "recomendación obtenida en entrenamiento."
        ),
        "Estrategias": (
            "Seis decisiones, una misma base.",
            "Cinco estrategias del certamen y tu umbral manual, "
            "evaluados bajo los mismos costos."
        ),
        "Sensibilidad": (
            "Anticipa distintos escenarios.",
            "Observa qué ocurre cuando aumenta o disminuye "
            "el costo de un fraude no detectado."
        ),
        "Modelo y método": (
            "La evidencia detrás de la decisión.",
            "Datos, comparación de modelos, validación "
            "y límites del caso de estudio."
        )
    }

    title, description = subtitles[section]

    n_test = (
        bundle.meta["n_test_0"]
        + bundle.meta["n_test_1"]
    )

    st.markdown(
        f"""
        <div class="hero">
            <div>
                <div class="eyebrow">
                    FraudShield / {section}
                </div>
                <h1>{title}</h1>
                <p>{description}</p>
            </div>
            <div class="hero-side">
                <small>CONJUNTO DE PRUEBA</small>
                <strong>{fmt(n_test)}</strong>
                <small>transacciones evaluadas</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def summary_cards(result):
    columns = st.columns(4)

    with columns[0]:
        card(
            "Costo total estimado",
            money(result["cost"]),
            "CLP · falsas alarmas + fraudes omitidos"
        )

    with columns[1]:
        saving = result["saving"]
        saving_pct = result["saving_pct"]

        card(
            (
                "Ahorro frente a 0,5"
                if saving >= 0
                else "Sobrecosto frente a 0,5"
            ),
            money(abs(saving)),
            (
                f"{percent(abs(saving_pct) if saving_pct is not None else None)} "
                "respecto al costo base"
            ),
            "positive" if saving >= 0 else "negative"
        )

    with columns[2]:
        card(
            "Fraudes detectados · recall",
            percent(result["recall"]),
            (
                f'{fmt(result["tp"])} detectados de '
                f'{fmt(result["tp"] + result["fn"])} fraudes'
            )
        )

    with columns[3]:
        note = (
            f'{fmt(result["tp"])} fraudes entre '
            f'{fmt(result["alerts"])} alertas'
            if result["alerts"]
            else "No hay alertas: la precisión no está definida"
        )

        card(
            "Alertas correctas · precisión",
            percent(result["precision"]),
            note
        )


def insight(result, threshold, optimal):
    saving = result["saving"]

    text = (
        f"Con el umbral <b>{fmt(threshold, 6)}</b>, "
        f'se detectan <b>{fmt(result["tp"])}</b> fraudes '
        f'y se generan <b>{fmt(result["fp"])}</b> falsas alarmas. '
    )

    if saving >= 0:
        text += (
            f"El costo estimado baja "
            f"<b>{money(saving)} CLP</b> "
            "frente al umbral 0,5."
        )

    else:
        text += (
            f"El costo estimado sube "
            f"<b>{money(-saving)} CLP</b> "
            "frente al umbral 0,5."
        )

    text += (
        " La recomendación de validación para estos costos es "
        f"<b>{fmt(optimal, 6)}</b>."
    )

    tone = "negative" if saving < 0 else ""

    st.markdown(
        f'<div class="insight {tone}">{text}</div>',
        unsafe_allow_html=True
    )


# =============================================================================
# 8. INFORME DESCARGABLE
# =============================================================================

def export_report(
    bundle,
    c_fp,
    c_fn,
    threshold,
    policy,
    manual=None
):
    table = strategy_table(
        bundle,
        c_fp,
        c_fn,
        threshold if manual is None else manual
    )

    result = metrics(
        bundle,
        threshold,
        c_fp,
        c_fn
    )

    rows = ""

    for row in table.round(6).fillna("No definida").values:
        cells = "".join(
            f"<td>{html.escape(str(value))}</td>"
            for value in row
        )

        rows += f"<tr>{cells}</tr>"

    headers = "".join(
        f"<th>{html.escape(column)}</th>"
        for column in table.columns
    )

    return f"""
    <!doctype html>
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <title>FraudShield · Respaldo</title>

        <style>
        body {{
            font: 16px Arial;
            color: #142C43;
            margin: 40px;
            max-width: 1400px;
        }}

        h1 {{
            border-bottom: 4px solid #007F73;
            padding-bottom: 15px;
        }}

        table {{
            border-collapse: collapse;
            font-size: 12px;
            width: 100%;
        }}

        td, th {{
            padding: 10px;
            border: 1px solid #ddd;
            text-align: right;
        }}

        th {{
            background: #EAF0F5;
        }}

        .box {{
            background: #E4F3EF;
            padding: 20px;
            margin: 20px 0;
        }}

        @media print {{
            body {{
                margin: 15px;
            }}
        }}
        </style>
    </head>

    <body>
        <h1>FraudShield · Informe del escenario</h1>

        <p>
            IEEE-CIS Fraud Detection · LightGBM refinado · Caso académico
        </p>

        <p>
            FP: {money(c_fp)} CLP · FN: {money(c_fn)} CLP.
            Costos supuestos, no pérdidas observadas.
        </p>

        <div class="box">
            Decisión: {html.escape(policy)}
            · Umbral: {fmt(threshold, 9)}
            <br>
            Costo: {money(result["cost"])} CLP
            · Ahorro firmado: {money(result["saving"])} CLP
            <br>
            Recall: {percent(result["recall"])}
            · Precisión: {percent(result["precision"])}
        </div>

        <h2>Comparación de estrategias en test</h2>

        <table>
            <thead>
                <tr>{headers}</tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>

        <p>
            El umbral se selecciona en validación del entrenamiento
            con curvas de TunedThresholdClassifierCV.
            Test se usa únicamente para evaluar.
        </p>

        <p>
            El ahorro se compara con el mismo LightGBM sin pesos
            a umbral 0,5. Las métricas no garantizan desempeño futuro.
        </p>

        <p>
            Respaldo estático del escenario.
            Se puede abrir sin conexión o imprimir como PDF.
        </p>
    </body>
    </html>
    """


# =============================================================================
# 9. SECCIÓN: PANORAMA
# =============================================================================

def overview(bundle, c_fp, c_fn, threshold, optimal, theoretical, policy):
    result = metrics(
        bundle,
        threshold,
        c_fp,
        c_fn
    )

    summary_cards(result)

    insight(
        result,
        threshold,
        optimal
    )

    left, right = st.columns(
        [1.65, 1],
        gap="large"
    )

    with left:
        with st.container(border=True):
            st.subheader(
                "¿Dónde se concentran las pérdidas?"
            )

            costs = [
                result["fp"] * c_fp,
                result["fn"] * c_fn
            ]

            fig = go.Figure(
                go.Bar(
                    x=costs,
                    y=[
                        "Falsas alarmas",
                        "Fraudes no detectados"
                    ],
                    orientation="h",
                    marker_color=[AMBER, RED],
                    text=[
                        money(value)
                        for value in costs
                    ],
                    textposition="auto",
                    hovertemplate=(
                        "%{y}<br>$%{x:,.0f} CLP"
                        "<extra></extra>"
                    )
                )
            )

            plot(
                chart_style(
                    fig,
                    height=230,
                    ytitle="Tipo de error"
                ),
                "cost_mix"
            )

            st.caption(
                "Los aciertos tienen costo cero "
                "en la matriz de costos del proyecto."
            )

    with right:
        with st.container(border=True):
            st.subheader("Decisión actual")

            st.metric(
                policy,
                fmt(threshold, 6)
            )

            st.write(
                f'**{fmt(result["alerts"])} operaciones** '
                "pasarían a revisión "
                f'({percent(result["alert_rate"])} del test).'
            )

            st.caption(
                "Una alerta identifica una operación para revisar. "
                "No equivale a un rechazo automático."
            )

    col1, col2, col3 = st.columns(3)

    with col1:
        card(
            "Umbral recomendado",
            fmt(optimal, 6),
            "Elegido con validación dentro de entrenamiento",
            "positive"
        )

    with col2:
        card(
            "Fraudes no detectados",
            fmt(result["fn"]),
            (
                f'{money(result["fn"] * c_fn)} CLP '
                "de costo supuesto"
            ),
            "negative"
        )

    with col3:
        card(
            "Costo de referencia · 0,5",
            money(result["base"]),
            "Mismo modelo sin pesos, mismo test y mismos costos"
        )

    with st.expander("Cómo leer este tablero"):
        st.write(
            "**Recall:** de todos los fraudes reales, "
            "cuántos detectamos. "
            "**Precisión:** de todas las alertas, "
            "cuántas son fraude."
        )

        st.write(
            "Bajar el umbral suele detectar más fraude, "
            "pero también aumenta las falsas alarmas."
        )

        st.write(
            "El ahorro es una diferencia de costos estimados "
            "sobre el conjunto de prueba; no es una proyección "
            "mensual ni un ingreso realizado."
        )


# =============================================================================
# 10. SECCIÓN: SIMULADOR
# =============================================================================

def simulator(bundle, c_fp, c_fn, threshold, optimal, theoretical, policy):
    result = metrics(
        bundle,
        threshold,
        c_fp,
        c_fn
    )

    summary_cards(result)

    left, right = st.columns(
        [1.65, 1],
        gap="large"
    )

    with left:
        with st.container(border=True):
            st.subheader(
                "Costo total según el umbral"
            )

            mode = st.radio(
                "Conjunto del gráfico",
                [
                    "Prueba · evaluación",
                    "Validación · selección"
                ],
                horizontal=True,
                key="curve_split"
            )

            split = (
                "test"
                if mode.startswith("Prueba")
                else "cv"
            )

            fig = cost_chart(
                bundle,
                c_fp,
                c_fn,
                threshold,
                optimal,
                theoretical,
                split
            )

            plot(
                fig,
                "cost_curve"
            )

            if split == "test":
                st.caption(
                    "El mínimo del test no selecciona el umbral. "
                    "La recomendación siempre procede de validación."
                )

            else:
                st.caption(
                    "Curva interpolada de TTCV: costo medio "
                    "por fold × 3. No representa conteos enteros "
                    "observados. Fuera de la grilla, los marcadores "
                    "usan el extremo disponible."
                )

    with right:
        with st.container(border=True):
            st.subheader(
                "Matriz de confusión"
            )

            st.caption(
                f"Test · {policy} · "
                f"umbral {fmt(threshold, 6)}"
            )

            plot(
                confusion_chart(
                    result,
                    c_fp,
                    c_fn
                ),
                "confusion"
            )

            st.caption(
                "FP: revisar una operación legítima. "
                "FN: dejar pasar un fraude."
            )

    insight(
        result,
        threshold,
        optimal
    )

    with st.expander(
        "Detalle del costo y del umbral teórico"
    ):
        st.write(
            f'Costo = {fmt(result["fp"])} × {money(c_fp)} '
            f'+ {fmt(result["fn"])} × {money(c_fn)} '
            f'= **{money(result["cost"])} CLP**.'
        )

        st.write(
            "Umbral teórico = costo FP / "
            "(costo FP + costo FN) "
            f"= **{fmt(theoretical, 9)}**."
        )

        st.caption(
            "El umbral teórico presupone probabilidades "
            "adecuadamente calibradas y costo cero para los aciertos. "
            "Quitar los pesos de clase no garantiza calibración perfecta."
        )


# =============================================================================
# 11. SECCIÓN: ESTRATEGIAS
# =============================================================================

def strategies(bundle, c_fp, c_fn, threshold, optimal, theoretical, policy):
    table = strategy_table(
        bundle,
        c_fp,
        c_fn,
        st.session_state["manual"]
    )

    st.markdown(
        '<div class="step">Comparación económica</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "La misma evaluación para todas las estrategias"
    )

    fig = go.Figure()

    components = [
        (
            "FP",
            AMBER,
            "Costo de falsas alarmas"
        ),
        (
            "FN",
            RED,
            "Costo de fraudes omitidos"
        )
    ]

    for column, color, name in components:
        factor = c_fp if column == "FP" else c_fn

        fig.add_bar(
            y=table["Estrategia"],
            x=table[column] * factor,
            name=name,
            orientation="h",
            marker_color=color,
            hovertemplate=(
                "%{y}<br>$%{x:,.0f} CLP"
                "<extra>" + name + "</extra>"
            )
        )

    fig.update_layout(
        barmode="stack"
    )

    fig.update_yaxes(
        autorange="reversed"
    )

    plot(
        chart_style(
            fig,
            height=390
        ),
        "strategy_bars"
    )

    st.caption(
        "Óptimo CV identifica la recomendación elegida en entrenamiento; "
        "no se elige al ganador mirando esta tabla de test. "
        "«Con pesos» corresponde al LightGBM refinado "
        "reentrenado con todo el train."
    )

    styled_table = table.style.format(
        {
            "Umbral": "{:.6f}",
            "Costo total (CLP)": "{:,.0f}",
            "Ahorro (CLP)": "{:,.0f}",
            "Recall": "{:.2%}",
            "Precisión": "{:.2%}",
            "F1": "{:.3f}"
        },
        na_rep="No definida"
    )

    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Descargar comparación en CSV",
        table.to_csv(index=False).encode("utf-8-sig"),
        "fraudshield_estrategias.csv",
        "text/csv",
        key="download_strategies"
    )

    st.caption(
        "Un ahorro negativo representa un sobrecosto. "
        "Para «Sin detección», la precisión no está definida "
        "porque no existen alertas."
    )


# =============================================================================
# 12. SECCIÓN: SENSIBILIDAD
# =============================================================================

def sensitivity(bundle, c_fp, c_fn, threshold, optimal, theoretical, policy):
    table = sensitivity_table(
        bundle,
        c_fp,
        c_fn
    )

    st.subheader(
        "¿Qué pasa si cambia el costo del fraude?"
    )

    st.write(
        f"Se mantiene el costo FP en **{money(c_fp)} CLP** "
        "y se varía el costo FN alrededor de "
        f"**{money(c_fn)} CLP**."
    )

    if c_fn == 0:
        st.info(
            "Con costo FN cero, los cinco multiplicadores "
            "producen el mismo escenario. Aumenta ese costo "
            "para explorar escenarios diferentes."
        )

    if c_fp == 0:
        st.info(
            "Con costo FP cero, la razón FN/FP no está definida. "
            "Los costos y umbrales siguen siendo calculables."
        )

    left, right = st.columns(
        2,
        gap="large"
    )

    with left:
        with st.container(border=True):
            st.subheader(
                "Efecto sobre el umbral"
            )

            fig = go.Figure()

            for column, name, color in [
                ("Umbral óptimo CV", "Óptimo CV", TEAL),
                ("Umbral teórico", "Teórico", BLUE)
            ]:
                fig.add_scatter(
                    x=table["Costo FN (CLP)"],
                    y=table[column],
                    mode="lines+markers",
                    name=name,
                    line=dict(
                        color=color,
                        width=2.5
                    ),
                    hovertemplate=(
                        "FN: $%{x:,.0f}"
                        "<br>Umbral: %{y:.6f}"
                        "<extra>" + name + "</extra>"
                    )
                )

            fig.update_xaxes(
                title="Costo de un FN (CLP)",
                tickprefix="$",
                tickformat="~s"
            )

            plot(
                chart_style(
                    fig,
                    height=350,
                    ytitle="Umbral"
                ),
                "sensitivity_threshold"
            )

    with right:
        with st.container(border=True):
            st.subheader(
                "Efecto sobre el costo en test"
            )

            fig = go.Figure()

            for column, name, color in [
                ("Costo 0,5 (CLP)", "Tradicional 0,5", MUTED),
                ("Costo teórico (CLP)", "Teórico", BLUE),
                ("Costo óptimo (CLP)", "Óptimo CV", TEAL)
            ]:
                fig.add_scatter(
                    x=table["Costo FN (CLP)"],
                    y=table[column],
                    mode="lines+markers",
                    name=name,
                    line=dict(
                        color=color,
                        width=2.5
                    ),
                    hovertemplate=(
                        "FN: $%{x:,.0f}"
                        "<br>Costo: $%{y:,.0f} CLP"
                        "<extra>" + name + "</extra>"
                    )
                )

            fig.update_xaxes(
                title="Costo de un FN (CLP)",
                tickprefix="$",
                tickformat="~s"
            )

            fig.update_yaxes(
                tickprefix="$",
                tickformat="~s"
            )

            plot(
                chart_style(
                    fig,
                    height=350,
                    ytitle="Costo total (CLP)"
                ),
                "sensitivity_cost"
            )

    formats = {
        column: (
            "{:.6f}"
            if "Umbral" in column
            else "{:,.0f}"
        )
        for column in table.columns
        if column != "Escenario"
    }

    st.dataframe(
        table.style.format(
            formats,
            na_rep="No definida"
        ),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Descargar escenarios en CSV",
        table.to_csv(index=False).encode("utf-8-sig"),
        "fraudshield_sensibilidad.csv",
        "text/csv",
        key="download_sensitivity"
    )

    st.info(
        "Cada recomendación se vuelve a seleccionar sobre la curva "
        "de validación. Los costos mostrados después corresponden "
        "al test. Los cinco escenarios usan ahora la misma grilla "
        "de TTCV; pueden diferir ligeramente de la grilla OOF "
        "del punto 6.7 original."
    )


# =============================================================================
# 13. SECCIÓN: MODELO Y MÉTODO
# =============================================================================

def methodology(bundle, c_fp, c_fn, threshold, optimal, theoretical, policy):
    n_train = (
        bundle.meta["n_train_0"]
        + bundle.meta["n_train_1"]
    )

    n_test = (
        bundle.meta["n_test_0"]
        + bundle.meta["n_test_1"]
    )

    n_fraud = (
        bundle.meta["n_train_1"]
        + bundle.meta["n_test_1"]
    )

    columns = st.columns(4)

    with columns[0]:
        card(
            "Dataset",
            fmt(n_train + n_test),
            "IEEE-CIS Fraud Detection · transacciones"
        )

    with columns[1]:
        card(
            "Fraude real",
            percent(n_fraud / (n_train + n_test)),
            "Clase positiva: isFraud = 1"
        )

    with columns[2]:
        card(
            "Entrenamiento / prueba",
            "80 % / 20 %",
            "Partición estratificada · semilla 42"
        )

    with columns[3]:
        card(
            "Modelo final",
            "LightGBM",
            "Refinado; sin pesos para analizar costos"
        )

    tab1, tab2, tab3 = st.tabs([
        "Modelos y selección",
        "Método y supuestos",
        "Verificación y exposición"
    ])

    with tab1:
        st.subheader(
            "Diez combinaciones iniciales"
        )

        st.caption(
            "Búsquedas sobre 30.000 observaciones de entrenamiento. "
            "Métricas recuperadas de las salidas del notebook, "
            "redondeadas a tres decimales. Esta tabla no corresponde "
            "al reentrenamiento final con 472.432 observaciones."
        )

        if not bundle.modelos.empty:
            fig = go.Figure()

            for name, color in [
                ("SMOTENC", BLUE),
                ("Pesos de clase", TEAL)
            ]:
                part = bundle.modelos[
                    bundle.modelos["Estrategia"] == name
                ]

                fig.add_bar(
                    x=part["Modelo"],
                    y=part["F1 CV"],
                    name=name,
                    marker_color=color,
                    error_y=dict(
                        type="data",
                        array=part["Desviación CV"]
                    ),
                    hovertemplate=(
                        "%{x}<br>F1 CV: %{y:.3f}"
                        "<extra>" + name + "</extra>"
                    )
                )

            fig.update_layout(
                barmode="group"
            )

            plot(
                chart_style(
                    fig,
                    height=340,
                    ytitle="F1 en validación cruzada"
                ),
                "models_cv"
            )

            st.dataframe(
                bundle.modelos,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "La tabla de las diez combinaciones no está exportada. "
                "Consulta el punto 7.1 del notebook."
            )

        st.subheader(
            "Refinamiento de los finalistas"
        )

        if not bundle.refinamiento.empty:
            st.dataframe(
                bundle.refinamiento,
                use_container_width=True,
                hide_index=True
            )

        st.write(
            "**LightGBM refinado** se seleccionó por F1 CV "
            "aproximado de 0,525. La diferencia respecto a XGBoost "
            "es pequeña frente a la variabilidad entre folds; "
            "no prueba superioridad estadística."
        )

        st.write(
            "Para costos se conservan 100 estimadores, 127 hojas, "
            "learning_rate 0,1, min_child_samples 50 y "
            "colsample_bytree 1,0; se elimina class_weight "
            "y se entrena con todo el train."
        )

    with tab2:
        with st.expander(
            "Del entrenamiento a una decisión económica",
            expanded=True
        ):
            st.markdown(
                """
                1. Preprocesamiento dentro del pipeline:
                   imputación, escalado y codificación.
                2. Cinco modelos con SMOTENC y cinco con pesos;
                   búsqueda estratificada de tres folds.
                3. Selección y refinamiento por F1 CV;
                   test no determina el ganador.
                4. LightGBM se reentrena sin SMOTE ni pesos
                   para el análisis de costos.
                5. El umbral se selecciona con validación de entrenamiento;
                   el test estima su resultado económico.
                """
            )

        with st.expander(
            "Cómo cambian los costos sin reentrenar"
        ):
            st.write(
                "Se reutilizan las probabilidades fuera de muestra "
                "(OOF) del punto 6.7 y las mismas tres particiones. "
                "TunedThresholdClassifierCV reconstruye dos curvas: "
                "falsos positivos medios y falsos negativos medios, "
                "con 401 umbrales y su interpolación original."
            )

            st.write(
                "Su combinación lineal permite obtener el objetivo "
                "para cualquier costo FP y FN. La app minimiza "
                "C_FP × FP_medio + C_FN × FN_medio sobre esa grilla."
            )

            st.write(
                "En empates toma el primer umbral, igual que TTCV. "
                "Las predicciones de test se cuentan de forma exacta "
                "con la regla probabilidad >= umbral, "
                "sin redondear el umbral."
            )

            st.caption(
                "La exportación verifica que se reproduce el umbral "
                "base del notebook. El óptimo es el mejor dentro de "
                "la grilla evaluada; no se garantiza un óptimo "
                "continuo global."
            )

        with st.expander(
            "Qué significan los costos"
        ):
            st.write(
                "**FP:** una transacción legítima genera una revisión "
                "innecesaria y posible fricción con el cliente."
            )

            st.write(
                "**FN:** una transacción fraudulenta no es detectada."
            )

            st.write(
                "Los valores base de $5.000 y $100.000 CLP "
                "son supuestos del caso académico. "
                "No son montos convertidos desde TransactionAmt "
                "ni pérdidas reales observadas."
            )

            st.write(
                "Los aciertos tienen costo cero en esta matriz "
                "simplificada."
            )

            st.write(
                "**Costo total = FP × C_FP + FN × C_FN.**"
            )

            st.write(
                "**Ahorro = costo del mismo modelo sin pesos "
                "a umbral 0,5 − costo de la estrategia evaluada.**"
            )

        with st.expander(
            "Calibración, limitaciones y uso responsable"
        ):
            st.write(
                "El notebook reporta Brier 0,0143 sin pesos "
                "y 0,0569 con pesos. Esto respalda la mejor calidad "
                "probabilística relativa del modelo sin pesos, "
                "pero no garantiza calibración perfecta."
            )

            st.write(
                "Se usó una partición aleatoria: falta validar "
                "en transacciones futuras. La consulta repetida "
                "del test puede introducir influencia indirecta."
            )

            st.write(
                "Los costos son constantes por error y no consideran "
                "montos individuales, capacidad operativa ni efectos "
                "a largo plazo. No se evaluó equidad entre grupos "
                "de clientes."
            )

            st.write(
                "La app contiene conteos y distribuciones agregadas. "
                "No publica TransactionID, tarjetas, correos ni filas "
                "del dataset. Una alerta debe apoyar una revisión "
                "humana, no imponer automáticamente un rechazo."
            )

    with tab3:
        base = metrics(
            bundle,
            bundle.meta["umbral_cv_base"],
            bundle.meta["c_fp_base"],
            bundle.meta["c_fn_base"]
        )

        expected = {
            "FP": 6197,
            "FN": 639,
            "Costo óptimo CLP": 94885000,
            "Costo 0,5 CLP": 190845000
        }

        actual = {
            "FP": base["fp"],
            "FN": base["fn"],
            "Costo óptimo CLP": base["cost"],
            "Costo 0,5 CLP": base["base"]
        }

        check = pd.DataFrame([
            {
                "Comprobación": name,
                "Notebook": value,
                "App": actual[name],
                "Coincide": actual[name] == value
            }
            for name, value in expected.items()
        ])

        st.subheader(
            "Escenario original del notebook"
        )

        st.dataframe(
            check,
            use_container_width=True,
            hide_index=True
        )

        if check["Coincide"].all():
            st.success(
                "El escenario base reproduce los resultados del notebook."
            )

        else:
            st.warning(
                "Esta exportación no coincide con el escenario "
                "original del notebook. Revisa la versión de los datos."
            )

        st.subheader(
            "Demostración en tres minutos"
        )

        st.markdown(
            """
            1. **0:00–0:45:** restaurar escenario base;
               explicar ahorro, recall y precisión.
            2. **0:45–1:30:** abrir Simulador;
               comparar 0,5 con el óptimo y mover el umbral manual.
            3. **1:30–2:15:** subir FN a $200.000;
               observar la nueva recomendación.
            4. **2:15–3:00:** abrir Sensibilidad y cerrar
               con costos supuestos y necesidad de validación futura.
            """
        )

        st.caption(
            "Usa «Descargar informe del escenario» en el panel lateral. "
            "El HTML se abre sin internet y se puede imprimir como PDF. "
            "Los gráficos también se descargan desde su barra "
            "de herramientas."
        )


# =============================================================================
# 14. INICIO DE LA APLICACIÓN
# =============================================================================

def main():
    st.set_page_config(
        page_title="FraudShield | Decisiones de riesgo",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(
        CSS,
        unsafe_allow_html=True
    )

    try:
        signature = tuple(
            (
                path.name,
                path.stat().st_mtime_ns,
                path.stat().st_size
            )
            for path in sorted(DATA_DIR.glob("*"))
            if path.is_file()
        )

        bundle = cached_bundle(
            str(DATA_DIR),
            signature
        )

    except (OSError, ValueError, KeyError, TypeError) as error:
        st.title("FraudShield")

        st.error(
            "No se pudieron cargar los datos del dashboard."
        )

        st.write(str(error))

        st.info(
            "Conserva la carpeta dashboard_data completa junto "
            "a app.py. Esta versión requiere también curva_ttcv.csv "
            "y distribucion_test.npz."
        )

        st.stop()

    defaults = {
        "c_fp": float(bundle.meta["c_fp_base"]),
        "c_fn": float(bundle.meta["c_fn_base"]),
        "manual": float(bundle.meta["umbral_cv_base"]),
        "policy": "Óptimo CV"
    }

    for key, value in defaults.items():
        st.session_state.setdefault(
            key,
            value
        )

    def reset():
        for key, value in defaults.items():
            st.session_state[key] = value

    # -------------------------------------------------------------------------
    # Panel lateral: navegación y costos
    # -------------------------------------------------------------------------

    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                Fraud<span>Shield</span>
            </div>
            <div class="brand-sub">
                CENTRO DE DECISIONES DE RIESGO
            </div>
            """,
            unsafe_allow_html=True
        )

        section = st.radio(
            "Secciones",
            SECTIONS,
            key="section"
        )

        st.divider()

        st.markdown(
            "**Escenario de costos**"
        )

        c_fp = st.number_input(
            "Falsa alarma · costo FP (CLP)",
            min_value=0.0,
            max_value=1e9,
            step=1000.0,
            format="%.0f",
            key="c_fp",
            help=(
                "Costo supuesto de revisar "
                "una transacción legítima."
            )
        )

        c_fn = st.number_input(
            "Fraude omitido · costo FN (CLP)",
            min_value=0.0,
            max_value=1e9,
            step=5000.0,
            format="%.0f",
            key="c_fn",
            help=(
                "Pérdida supuesta por un fraude no detectado."
            )
        )

        st.caption(
            "Costos hipotéticos del proyecto, expresados en CLP."
        )

        st.button(
            "Restaurar escenario base",
            on_click=reset,
            use_container_width=True
        )

    try:
        validate_costs(
            c_fp,
            c_fn
        )

    except ValueError as error:
        headline(
            section,
            bundle
        )

        st.warning(str(error))
        st.stop()

    optimal = optimal_threshold(
        bundle,
        c_fp,
        c_fn
    )

    theoretical = c_fp / (c_fp + c_fn)

    # -------------------------------------------------------------------------
    # Panel lateral: decisión y descarga
    # -------------------------------------------------------------------------

    with st.sidebar:
        if c_fp:
            st.caption(
                f"Razón FN/FP: {fmt(c_fn / c_fp, 2)} veces"
            )

        else:
            st.caption(
                "Razón FN/FP: no definida (FP = 0)"
            )

        st.markdown(
            "**Decisión de clasificación**"
        )

        policy = st.selectbox(
            "Umbral aplicado",
            [
                "Óptimo CV",
                "Manual",
                "Teórico",
                "Tradicional 0,5"
            ],
            key="policy"
        )

        manual = st.slider(
            "Explorar umbral manual",
            min_value=0.0,
            max_value=1.0,
            step=0.0001,
            format="%.4f",
            key="manual"
        )

        if policy != "Manual":
            st.caption(
                "Selecciona «Manual» para aplicar este control."
            )

        else:
            st.caption(
                "Las métricas usan el umbral manual."
            )

        thresholds = {
            "Óptimo CV": optimal,
            "Manual": manual,
            "Teórico": theoretical,
            "Tradicional 0,5": 0.5
        }

        threshold = thresholds[policy]

        st.markdown(
            f"**Recomendación CV:** {fmt(optimal, 6)}  \n"
            f"**Aplicado:** {fmt(threshold, 6)}"
        )

        st.divider()

        report = export_report(
            bundle,
            c_fp,
            c_fn,
            threshold,
            policy,
            manual
        )

        st.download_button(
            "Descargar informe del escenario",
            report.encode("utf-8"),
            "fraudshield_informe.html",
            "text/html",
            use_container_width=True
        )

        st.caption(
            "HTML sin conexión · imprimible como PDF"
        )

    # -------------------------------------------------------------------------
    # Contenido principal
    # -------------------------------------------------------------------------

    headline(
        section,
        bundle
    )

    st.caption(
        f"LightGBM refinado · {policy} "
        f"· Umbral {fmt(threshold, 9)} "
        f"· FP {money(c_fp)} / FN {money(c_fn)} CLP"
    )

    pages = {
        "Panorama": overview,
        "Simulador": simulator,
        "Estrategias": strategies,
        "Sensibilidad": sensitivity,
        "Modelo y método": methodology
    }

    pages[section](
        bundle,
        c_fp,
        c_fn,
        threshold,
        optimal,
        theoretical,
        policy
    )

    st.markdown(
        """
        <div class="footer">
            FraudShield · IEEE-CIS Fraud Detection
            · Certamen 1 de Machine Learning
            <br>
            Costos supuestos en CLP.
            Evaluación histórica en test; no constituye
            una estimación garantizada de ahorro futuro.
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
