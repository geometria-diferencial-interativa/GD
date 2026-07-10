import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================

st.set_page_config(
    page_title="Curvas planas e espaciais",
    page_icon="📈",
    layout="wide",
)

EPS = 1e-10


def cumulative_trapezoid_local(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Integral acumulada pelo método dos trapézios, sem depender do SciPy."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape[0] != x.shape[0]:
        raise ValueError("x e y precisam ter o mesmo número de pontos.")
    if len(x) < 2:
        return np.array([], dtype=float)
    dx = np.diff(x)
    if y.ndim == 1:
        areas = 0.5 * (y[:-1] + y[1:]) * dx
    else:
        areas = 0.5 * (y[:-1] + y[1:]) * dx[:, None]
    return np.cumsum(areas, axis=0)


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compatibilidade interna: substitui scipy.integrate.cumulative_trapezoid."""
    return cumulative_trapezoid_local(y, x)


def rk4_on_grid(rhs: Callable[[float, np.ndarray], np.ndarray], grid: np.ndarray, y0: np.ndarray) -> np.ndarray:
    """Resolve y'=f(t,y) em uma malha dada usando Runge–Kutta clássico de ordem 4."""
    grid = np.asarray(grid, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("A malha deve conter pelo menos dois pontos.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("A malha de integração deve ser estritamente crescente.")
    values = np.empty((len(grid), len(y0)), dtype=float)
    values[0] = y0
    for i in range(len(grid) - 1):
        x = float(grid[i])
        h = float(grid[i + 1] - grid[i])
        y = values[i]
        k1 = rhs(x, y)
        k2 = rhs(x + h / 2.0, y + h * k1 / 2.0)
        k3 = rhs(x + h / 2.0, y + h * k2 / 2.0)
        k4 = rhs(x + h, y + h * k3)
        values[i + 1] = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return values


# ============================================================
# MENU LATERAL DA PLATAFORMA
# ============================================================

with st.sidebar:
    st.title("📐 Geometria Diferencial")

    st.page_link("app.py", label="🏠 Início")
    st.page_link("pages/01_Curvas.py", label="1. Curvas")
    st.page_link("pages/02_Superficies_Regulares.py", label="2. Superfícies Regulares")
    st.page_link("pages/03_Plano_Tangente_Normal.py", label="3. Plano Tangente e Vetor Normal")
    st.page_link("pages/04_Formas_Fundamentais.py", label="4. Formas Fundamentais")
    st.page_link("pages/05_Curvaturas.py", label="5. Curvaturas")
    st.page_link(
        "pages/06_Superficies_Minimas_Variacao_Area.py",
        label="6. Superfícies Mínimas e Variação da Área",
    )


# ============================================================
# UTILITÁRIOS MATEMÁTICOS
# ============================================================


def fmt(x: float, digits: int = 5) -> str:
    try:
        value = float(x)
        if not np.isfinite(value):
            return "não definido"
        return f"{value:.{digits}f}"
    except Exception:
        return "não definido"


def vector_latex(v: np.ndarray, digits: int = 5) -> str:
    values = r",\;".join(fmt(float(x), digits) for x in np.asarray(v).reshape(-1))
    return rf"\left({values}\right)"


def safe_unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < EPS:
        return None
    return np.asarray(v, dtype=float) / n


def as_array(value, t: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full_like(t, float(arr), dtype=float)
    return arr


def eval_expr(expr: str, variable_name: str, variable_values: np.ndarray) -> np.ndarray:
    allowed = {
        variable_name: variable_values,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
        "where": np.where,
        "minimum": np.minimum,
        "maximum": np.maximum,
    }
    value = eval(expr, {"__builtins__": {}}, allowed)
    return as_array(value, variable_values)


def numerical_derivatives(values: np.ndarray, parameter: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.gradient(values, parameter, axis=0, edge_order=2)
    second = np.gradient(first, parameter, axis=0, edge_order=2)
    third = np.gradient(second, parameter, axis=0, edge_order=2)
    return first, second, third


def planar_curvature(alpha1: np.ndarray, alpha2: np.ndarray) -> np.ndarray:
    numerator = alpha1[:, 0] * alpha2[:, 1] - alpha1[:, 1] * alpha2[:, 0]
    speed = np.linalg.norm(alpha1, axis=1)
    denominator = speed**3
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > EPS,
    )


def spatial_curvature(alpha1: np.ndarray, alpha2: np.ndarray) -> np.ndarray:
    cross = np.cross(alpha1, alpha2)
    numerator = np.linalg.norm(cross, axis=1)
    denominator = np.linalg.norm(alpha1, axis=1) ** 3
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > EPS,
    )


def spatial_torsion(alpha1: np.ndarray, alpha2: np.ndarray, alpha3: np.ndarray) -> np.ndarray:
    cross = np.cross(alpha1, alpha2)
    numerator = np.einsum("ij,ij->i", cross, alpha3)
    denominator = np.linalg.norm(cross, axis=1) ** 2
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > EPS,
    )


def cumulative_arc_length(alpha1: np.ndarray, parameter: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(alpha1, axis=1)
    return np.concatenate(([0.0], cumulative_trapezoid_local(speed, parameter)))


def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    result = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        result &= np.all(np.isfinite(arr), axis=1) if arr.ndim > 1 else np.isfinite(arr)
    return result


# ============================================================
# ESTRUTURAS DE DADOS PARA CURVAS
# ============================================================


@dataclass
class CurveData:
    parameter: np.ndarray
    alpha: np.ndarray
    alpha1: np.ndarray
    alpha2: np.ndarray
    alpha3: np.ndarray
    latex: str
    is_numeric: bool


# ============================================================
# CURVAS PLANAS PRONTAS
# ============================================================


def planar_curve(
    t: np.ndarray,
    name: str,
    params: Dict[str, float],
    custom: Optional[Dict[str, str]] = None,
) -> CurveData:
    if name == "Reta":
        x0, y0 = params["x0"], params["y0"]
        vx, vy = params["vx"], params["vy"]
        alpha = np.column_stack((x0 + vx * t, y0 + vy * t))
        alpha1 = np.tile(np.array([vx, vy], dtype=float), (len(t), 1))
        alpha2 = np.zeros_like(alpha)
        alpha3 = np.zeros_like(alpha)
        latex = rf"\alpha(t)=({x0:.3g}+{vx:.3g}t,\;{y0:.3g}+{vy:.3g}t)"
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, False)

    if name == "Circunferência":
        r, xc, yc = params["r"], params["xc"], params["yc"]
        alpha = np.column_stack((xc + r * np.cos(t), yc + r * np.sin(t)))
        alpha1 = np.column_stack((-r * np.sin(t), r * np.cos(t)))
        alpha2 = np.column_stack((-r * np.cos(t), -r * np.sin(t)))
        alpha3 = np.column_stack((r * np.sin(t), -r * np.cos(t)))
        latex = rf"\alpha(t)=({xc:.3g}+{r:.3g}\cos t,\;{yc:.3g}+{r:.3g}\sin t)"
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, False)

    if name == "Parábola":
        a = params["a"]
        alpha = np.column_stack((t, a * t**2))
        alpha1 = np.column_stack((np.ones_like(t), 2 * a * t))
        alpha2 = np.column_stack((np.zeros_like(t), np.full_like(t, 2 * a)))
        alpha3 = np.zeros_like(alpha)
        latex = rf"\alpha(t)=(t,\;{a:.3g}t^2)"
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, False)

    if name == "Senoide":
        a, b = params["a"], params["b"]
        alpha = np.column_stack((t, a * np.sin(b * t)))
        alpha1 = np.column_stack((np.ones_like(t), a * b * np.cos(b * t)))
        alpha2 = np.column_stack((np.zeros_like(t), -a * b**2 * np.sin(b * t)))
        alpha3 = np.column_stack((np.zeros_like(t), -a * b**3 * np.cos(b * t)))
        latex = rf"\alpha(t)=(t,\;{a:.3g}\sin({b:.3g}t))"
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, False)

    if name == "Catenária":
        a = params["a"]
        alpha = np.column_stack((t, a * np.cosh(t / a)))
        alpha1 = np.column_stack((np.ones_like(t), np.sinh(t / a)))
        alpha2 = np.column_stack((np.zeros_like(t), np.cosh(t / a) / a))
        alpha3 = np.column_stack((np.zeros_like(t), np.sinh(t / a) / a**2))
        latex = rf"\alpha(t)=\left(t,\;{a:.3g}\cosh\left(\frac{{t}}{{{a:.3g}}}\right)\right)"
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, False)

    if name == "Cúspide semicúbica":
        alpha = np.column_stack((t**2, t**3))
        alpha1 = np.column_stack((2 * t, 3 * t**2))
        alpha2 = np.column_stack((np.full_like(t, 2.0), 6 * t))
        alpha3 = np.column_stack((np.zeros_like(t), np.full_like(t, 6.0)))
        latex = r"\alpha(t)=(t^2,\;t^3)"
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, False)

    if name == "Curva com auto-interseção":
        alpha = np.column_stack((np.sin(t), np.sin(2 * t)))
        alpha1 = np.column_stack((np.cos(t), 2 * np.cos(2 * t)))
        alpha2 = np.column_stack((-np.sin(t), -4 * np.sin(2 * t)))
        alpha3 = np.column_stack((-np.cos(t), -8 * np.cos(2 * t)))
        latex = r"\alpha(t)=(\sin t,\;\sin 2t)"
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, False)

    if name == "Personalizada":
        if custom is None:
            raise ValueError("As expressões da curva personalizada não foram fornecidas.")
        x = eval_expr(custom["x"], "t", t)
        y = eval_expr(custom["y"], "t", t)
        alpha = np.column_stack((x, y))
        alpha1, alpha2, alpha3 = numerical_derivatives(alpha, t)
        latex = rf"\alpha(t)=\left({custom['x']},\;{custom['y']}\right)"
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, True)

    raise ValueError("Curva plana desconhecida.")


# ============================================================
# CURVAS ESPACIAIS PRONTAS
# ============================================================


def spatial_curve(
    t: np.ndarray,
    name: str,
    params: Dict[str, float],
    custom: Optional[Dict[str, str]] = None,
) -> CurveData:
    if name == "Reta espacial":
        p = np.array([params["x0"], params["y0"], params["z0"]], dtype=float)
        v = np.array([params["vx"], params["vy"], params["vz"]], dtype=float)
        alpha = p + t[:, None] * v
        alpha1 = np.tile(v, (len(t), 1))
        alpha2 = np.zeros_like(alpha)
        alpha3 = np.zeros_like(alpha)
        latex = (
            rf"\alpha(t)=({p[0]:.3g},{p[1]:.3g},{p[2]:.3g})"
            rf"+t({v[0]:.3g},{v[1]:.3g},{v[2]:.3g})"
        )
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, False)

    if name == "Hélice circular":
        a, b = params["a"], params["b"]
        alpha = np.column_stack((a * np.cos(t), a * np.sin(t), b * t))
        alpha1 = np.column_stack((-a * np.sin(t), a * np.cos(t), np.full_like(t, b)))
        alpha2 = np.column_stack((-a * np.cos(t), -a * np.sin(t), np.zeros_like(t)))
        alpha3 = np.column_stack((a * np.sin(t), -a * np.cos(t), np.zeros_like(t)))
        latex = rf"\alpha(t)=({a:.3g}\cos t,\;{a:.3g}\sin t,\;{b:.3g}t)"
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, False)

    if name == "Parábola espacial":
        c = params["c"]
        alpha = np.column_stack((t, t**2, c * t))
        alpha1 = np.column_stack((np.ones_like(t), 2 * t, np.full_like(t, c)))
        alpha2 = np.column_stack((np.zeros_like(t), np.full_like(t, 2.0), np.zeros_like(t)))
        alpha3 = np.zeros_like(alpha)
        latex = rf"\alpha(t)=(t,\;t^2,\;{c:.3g}t)"
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, False)

    if name == "Cúbica torcida":
        alpha = np.column_stack((t, t**2, t**3))
        alpha1 = np.column_stack((np.ones_like(t), 2 * t, 3 * t**2))
        alpha2 = np.column_stack((np.zeros_like(t), np.full_like(t, 2.0), 6 * t))
        alpha3 = np.column_stack((np.zeros_like(t), np.zeros_like(t), np.full_like(t, 6.0)))
        latex = r"\alpha(t)=(t,\;t^2,\;t^3)"
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, False)

    if name == "Curva toroidal":
        R, r, m, n = params["R"], params["r"], params["m"], params["n"]
        A = R + r * np.cos(n * t)
        alpha = np.column_stack((A * np.cos(m * t), A * np.sin(m * t), r * np.sin(n * t)))
        alpha1, alpha2, alpha3 = numerical_derivatives(alpha, t)
        latex = (
            r"\alpha(t)=\big((R+r\cos(nt))\cos(mt),\;"
            r"(R+r\cos(nt))\sin(mt),\;r\sin(nt)\big)"
        )
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, True)

    if name == "Curva de Viviani":
        a = params["a"]
        alpha = np.column_stack(
            (
                a * (1 + np.cos(t)),
                a * np.sin(t),
                2 * a * np.sin(t / 2),
            )
        )
        alpha1 = np.column_stack(
            (
                -a * np.sin(t),
                a * np.cos(t),
                a * np.cos(t / 2),
            )
        )
        alpha2 = np.column_stack(
            (
                -a * np.cos(t),
                -a * np.sin(t),
                -0.5 * a * np.sin(t / 2),
            )
        )
        alpha3 = np.column_stack(
            (
                a * np.sin(t),
                -a * np.cos(t),
                -0.25 * a * np.cos(t / 2),
            )
        )
        latex = rf"\alpha(t)=\left({a:.3g}(1+\cos t),\;{a:.3g}\sin t,\;{2*a:.3g}\sin\frac{{t}}{{2}}\right)"
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, False)

    if name == "Personalizada":
        if custom is None:
            raise ValueError("As expressões da curva personalizada não foram fornecidas.")
        x = eval_expr(custom["x"], "t", t)
        y = eval_expr(custom["y"], "t", t)
        z = eval_expr(custom["z"], "t", t)
        alpha = np.column_stack((x, y, z))
        alpha1, alpha2, alpha3 = numerical_derivatives(alpha, t)
        latex = rf"\alpha(t)=\left({custom['x']},\;{custom['y']},\;{custom['z']}\right)"
        return CurveData(t, alpha, alpha1, alpha2, alpha3, latex, True)

    raise ValueError("Curva espacial desconhecida.")


# ============================================================
# GRÁFICOS E ELEMENTOS GEOMÉTRICOS
# ============================================================


def add_2d_vector(fig: go.Figure, p: np.ndarray, v: np.ndarray, name: str, scale: float = 1.0) -> None:
    if v is None or np.linalg.norm(v) < EPS:
        return
    q = p + scale * v
    fig.add_trace(
        go.Scatter(
            x=[p[0], q[0]],
            y=[p[1], q[1]],
            mode="lines+markers",
            name=name,
            marker=dict(size=[0, 8], symbol=["circle", "triangle-up"]),
            line=dict(width=4),
        )
    )


def add_3d_vector(fig: go.Figure, p: np.ndarray, v: np.ndarray, name: str, scale: float = 1.0) -> None:
    if v is None or np.linalg.norm(v) < EPS:
        return
    q = p + scale * v
    fig.add_trace(
        go.Scatter3d(
            x=[p[0], q[0]],
            y=[p[1], q[1]],
            z=[p[2], q[2]],
            mode="lines",
            name=name,
            line=dict(width=7),
        )
    )
    fig.add_trace(
        go.Cone(
            x=[q[0]],
            y=[q[1]],
            z=[q[2]],
            u=[v[0]],
            v=[v[1]],
            w=[v[2]],
            anchor="tip",
            sizemode="absolute",
            sizeref=max(0.08, 0.18 * scale),
            showscale=False,
            name=name,
            showlegend=False,
        )
    )


def add_2d_line(fig: go.Figure, p: np.ndarray, direction: np.ndarray, name: str, length: float) -> None:
    d = safe_unit(direction)
    if d is None:
        return
    q1 = p - length * d
    q2 = p + length * d
    fig.add_trace(
        go.Scatter(
            x=[q1[0], q2[0]],
            y=[q1[1], q2[1]],
            mode="lines",
            name=name,
            line=dict(width=2, dash="dash"),
        )
    )


def add_3d_line(fig: go.Figure, p: np.ndarray, direction: np.ndarray, name: str, length: float) -> None:
    d = safe_unit(direction)
    if d is None:
        return
    q1 = p - length * d
    q2 = p + length * d
    fig.add_trace(
        go.Scatter3d(
            x=[q1[0], q2[0]],
            y=[q1[1], q2[1]],
            z=[q1[2], q2[2]],
            mode="lines",
            name=name,
            line=dict(width=4, dash="dash"),
        )
    )


def make_planar_plot(
    alpha: np.ndarray,
    p: np.ndarray,
    velocity: np.ndarray,
    T: Optional[np.ndarray],
    N: Optional[np.ndarray],
    vector_scale: float,
    show_velocity: bool,
    show_tangent_vector: bool,
    show_normal_vector: bool,
    show_tangent_line: bool,
    show_normal_line: bool,
    show_circle: bool,
    center: Optional[np.ndarray],
    radius: Optional[float],
    equal_axes: bool,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=alpha[:, 0],
            y=alpha[:, 1],
            mode="lines",
            name="Curva",
            line=dict(width=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p[0]],
            y=[p[1]],
            mode="markers",
            name="Ponto α(t₀)",
            marker=dict(size=10),
        )
    )

    extent = max(np.ptp(alpha[:, 0]), np.ptp(alpha[:, 1]), 1.0)
    line_length = 0.6 * extent

    if show_velocity:
        add_2d_vector(fig, p, velocity, "α'(t₀): velocidade", vector_scale)
    if show_tangent_vector and T is not None:
        add_2d_vector(fig, p, T, "T(t₀): tangente unitário", vector_scale)
    if show_normal_vector and N is not None:
        add_2d_vector(fig, p, N, "N(t₀): normal", vector_scale)
    if show_tangent_line and T is not None:
        add_2d_line(fig, p, T, "Reta tangente", line_length)
    if show_normal_line and N is not None:
        add_2d_line(fig, p, N, "Reta normal", line_length)

    if show_circle and center is not None and radius is not None and np.isfinite(radius):
        theta = np.linspace(0, 2 * np.pi, 300)
        circle = center + radius * np.column_stack((np.cos(theta), np.sin(theta)))
        fig.add_trace(
            go.Scatter(
                x=circle[:, 0],
                y=circle[:, 1],
                mode="lines",
                name="Círculo osculador",
                line=dict(width=3, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[center[0]],
                y=[center[1]],
                mode="markers",
                name="Centro de curvatura",
                marker=dict(size=8, symbol="x"),
            )
        )

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=35, b=0),
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(
            font=dict(size=13),
            x=0.02,
            y=0.02,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="rgba(100,100,100,0.4)",
            borderwidth=1,
        ),
    )
    if equal_axes:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def make_spatial_plot(
    alpha: np.ndarray,
    p: np.ndarray,
    T: Optional[np.ndarray],
    N: Optional[np.ndarray],
    B: Optional[np.ndarray],
    vector_scale: float,
    show_frame: bool,
    show_tangent_line: bool,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=alpha[:, 0],
            y=alpha[:, 1],
            z=alpha[:, 2],
            mode="lines",
            name="Curva",
            line=dict(width=6),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[p[0]],
            y=[p[1]],
            z=[p[2]],
            mode="markers",
            name="Ponto α(t₀)",
            marker=dict(size=6),
        )
    )

    extent = max(np.ptp(alpha[:, 0]), np.ptp(alpha[:, 1]), np.ptp(alpha[:, 2]), 1.0)
    if show_frame:
        add_3d_vector(fig, p, T, "T(t₀): tangente", vector_scale)
        add_3d_vector(fig, p, N, "N(t₀): normal principal", vector_scale)
        add_3d_vector(fig, p, B, "B(t₀): binormal", vector_scale)
    if show_tangent_line and T is not None:
        add_3d_line(fig, p, T, "Reta tangente", 0.6 * extent)

    fig.update_layout(
        height=670,
        margin=dict(l=0, r=0, t=35, b=0),
        legend=dict(
            font=dict(size=13),
            itemsizing="constant",
            x=0.02,
            y=0.02,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="rgba(100,100,100,0.4)",
            borderwidth=1,
        ),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
    )
    return fig


def make_scalar_plot(x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=y_label, line=dict(width=4)))
    fig.update_layout(
        height=360,
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=0, r=0, t=45, b=0),
        showlegend=False,
    )
    return fig


# ============================================================
# CONCEITOS FUNDAMENTAIS
# ============================================================


def render_fundamental_concepts() -> None:
    with st.expander("📘 Conceitos fundamentais comuns", expanded=False):
        st.markdown(
            r"""
Uma **curva parametrizada diferenciável** em $\mathbb{R}^n$ é uma aplicação
"""
        )
        st.latex(r"\alpha:I\subset\mathbb{R}\longrightarrow\mathbb{R}^n")
        st.markdown("cujas funções coordenadas são diferenciáveis.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Traço da curva**")
            st.latex(r"\alpha(I)=\{\alpha(t):t\in I\}")
            st.markdown(
                "O traço é o conjunto geométrico percorrido. Parametrizações diferentes podem ter o mesmo traço."
            )

            st.markdown("**Vetor velocidade e rapidez**")
            st.latex(r"\alpha'(t)=(x_1'(t),\ldots,x_n'(t))")
            st.latex(r"v(t)=\|\alpha'(t)\|")

            st.markdown("**Curva regular**")
            st.latex(r"\alpha'(t)\neq 0\quad\text{para todo }t\in I")
            st.markdown("Em um ponto regular, a direção tangente está bem definida.")

        with col2:
            st.markdown("**Reta tangente em $t_0$**")
            st.latex(r"r(u)=\alpha(t_0)+u\alpha'(t_0)")

            st.markdown("**Mudança de parâmetro**")
            st.latex(r"\beta(s)=\alpha(\varphi(s))")
            st.markdown(
                r"Se $\varphi'(s)>0$, a orientação é preservada; se $\varphi'(s)<0$, a orientação é invertida."
            )

            st.markdown("**Comprimento de arco**")
            st.latex(r"L(\alpha)=\int_a^b\|\alpha'(t)\|\,dt")
            st.latex(r"s(t)=\int_{t_0}^{t}\|\alpha'(u)\|\,du")

        st.info(
            "Uma curva simples não possui auto-interseções, exceto pela coincidência possível das extremidades em uma curva fechada."
        )


# ============================================================
# ANÁLISE DE CURVAS PLANAS
# ============================================================


def render_planar_analysis() -> None:
    st.header(r"Curvas planas em $\mathbb{R}^2$")
    st.markdown(
        r"Nesta seção, analisamos uma curva $\alpha(t)=(x(t),y(t))$ e os objetos geométricos associados a um ponto escolhido."
    )

    with st.sidebar:
        st.subheader("Curva plana")
        name = st.selectbox(
            "Escolha uma curva",
            [
                "Reta",
                "Circunferência",
                "Parábola",
                "Senoide",
                "Catenária",
                "Cúspide semicúbica",
                "Curva com auto-interseção",
                "Personalizada",
            ],
            key="planar_name",
        )

        params: Dict[str, float] = {}
        custom: Optional[Dict[str, str]] = None

        if name == "Reta":
            params["x0"] = st.number_input("x₀", value=0.0, key="pl_x0")
            params["y0"] = st.number_input("y₀", value=0.0, key="pl_y0")
            params["vx"] = st.number_input("vₓ", value=1.0, key="pl_vx")
            params["vy"] = st.number_input("vᵧ", value=1.0, key="pl_vy")
        elif name == "Circunferência":
            params["r"] = st.slider("Raio r", 0.2, 5.0, 1.5, 0.1, key="pl_r")
            params["xc"] = st.number_input("Centro x_c", value=0.0, key="pl_xc")
            params["yc"] = st.number_input("Centro y_c", value=0.0, key="pl_yc")
        elif name in {"Parábola", "Senoide", "Catenária"}:
            params["a"] = st.slider("Parâmetro a", 0.2, 3.0, 1.0, 0.1, key=f"pl_a_{name}")
            if name == "Senoide":
                params["b"] = st.slider("Parâmetro b", 0.2, 4.0, 1.0, 0.1, key="pl_b")
        elif name == "Personalizada":
            st.caption("Funções: sin, cos, tan, sinh, cosh, exp, log, sqrt, abs, pi.")
            custom = {
                "x": st.text_input("x(t)", "cos(t)", key="pl_custom_x"),
                "y": st.text_input("y(t)", "sin(t)", key="pl_custom_y"),
            }

        default_domains = {
            "Reta": (-3.0, 3.0),
            "Circunferência": (0.0, 2 * np.pi),
            "Parábola": (-2.5, 2.5),
            "Senoide": (-2 * np.pi, 2 * np.pi),
            "Catenária": (-2.5, 2.5),
            "Cúspide semicúbica": (-2.0, 2.0),
            "Curva com auto-interseção": (0.0, 2 * np.pi),
            "Personalizada": (-3.0, 3.0),
        }
        default_min, default_max = default_domains[name]

        st.subheader("Domínio e ponto")
        tmin = st.number_input("t mínimo", value=float(default_min), key=f"pl_tmin_{name}")
        tmax = st.number_input("t máximo", value=float(default_max), key=f"pl_tmax_{name}")
        resolution = st.slider("Resolução", 200, 1500, 700, 100, key="pl_resolution")

        if tmax > tmin:
            t0 = st.slider(
                "Ponto t₀",
                min_value=float(tmin),
                max_value=float(tmax),
                value=float((tmin + tmax) / 2),
                key=f"pl_t0_{name}",
            )
        else:
            t0 = float(tmin)

        st.subheader("Elementos visuais")
        vector_scale = st.slider("Escala dos vetores", 0.1, 3.0, 1.0, 0.1, key="pl_scale")
        show_velocity = st.checkbox("Vetor velocidade", value=False, key="pl_show_velocity")
        show_tangent_vector = st.checkbox("Vetor tangente unitário", value=True, key="pl_show_T")
        show_normal_vector = st.checkbox("Vetor normal", value=True, key="pl_show_N")
        show_tangent_line = st.checkbox("Reta tangente", value=True, key="pl_show_tangent")
        show_normal_line = st.checkbox("Reta normal", value=False, key="pl_show_normal_line")
        show_circle = st.checkbox("Círculo osculador", value=True, key="pl_show_circle")
        equal_axes = st.checkbox("Mesma escala nos eixos", value=True, key="pl_equal_axes")

    if tmax <= tmin:
        st.error(r"O domínio deve satisfazer $t_{\min}<t_{\max}$.")
        return

    try:
        t = np.linspace(tmin, tmax, resolution)
        data = planar_curve(t, name, params, custom)
    except Exception as exc:
        st.error(f"Não foi possível avaliar a curva: {exc}")
        return

    mask = finite_mask(data.alpha, data.alpha1, data.alpha2)
    if not np.all(mask):
        st.warning("Alguns valores não são finitos. O domínio ou a expressão personalizada deve ser ajustado.")
    if np.count_nonzero(mask) < 5:
        st.error("Não há pontos válidos suficientes para construir o gráfico.")
        return

    i0 = nearest_index(t, t0)
    p = data.alpha[i0]
    v = data.alpha1[i0]
    a2 = data.alpha2[i0]
    speed = float(np.linalg.norm(v))
    T = safe_unit(v)

    N = None
    if T is not None:
        N = np.array([-T[1], T[0]], dtype=float)

    kappa_values = planar_curvature(data.alpha1, data.alpha2)
    kappa = float(kappa_values[i0]) if np.isfinite(kappa_values[i0]) else np.nan

    radius = None
    center = None
    if N is not None and np.isfinite(kappa) and abs(kappa) > 1e-9:
        radius = 1.0 / abs(kappa)
        center = p + (1.0 / kappa) * N

    st.subheader("Curva escolhida")
    st.latex(rf"\alpha:[{fmt(tmin,2)},{fmt(tmax,2)}]\longrightarrow\mathbb{{R}}^2")
    st.latex(data.latex)

    if data.is_numeric:
        st.caption("As derivadas desta curva são aproximadas numericamente.")

    left, right = st.columns([1.45, 1.0])
    with left:
        fig = make_planar_plot(
            data.alpha[mask],
            p,
            v,
            T,
            N,
            vector_scale,
            show_velocity,
            show_tangent_vector,
            show_normal_vector,
            show_tangent_line,
            show_normal_line,
            show_circle,
            center,
            radius,
            equal_axes,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Valores em $t_0$")
        st.latex(rf"t_0={fmt(t[i0],5)}")
        st.latex(rf"\alpha(t_0)={vector_latex(p)}")
        st.latex(rf"\alpha'(t_0)={vector_latex(v)}")
        st.latex(rf"\|\alpha'(t_0)\|={fmt(speed)}")

        if T is None:
            st.error(r"A curva não é regular nesse ponto: $\alpha'(t_0)=0$.")
        else:
            st.latex(rf"T(t_0)={vector_latex(T)}")
            st.latex(rf"N(t_0)={vector_latex(N)}")
            st.latex(rf"\kappa(t_0)={fmt(kappa)}")

            if radius is None or center is None:
                st.info(r"O raio e o centro de curvatura não estão definidos quando $\kappa(t_0)=0$.")
            else:
                st.latex(rf"\rho(t_0)=\frac{{1}}{{|\kappa(t_0)|}}={fmt(radius)}")
                st.latex(rf"C(t_0)={vector_latex(center)}")

    with st.expander("Como os cálculos são realizados", expanded=True):
        st.latex(r"T(t)=\frac{\alpha'(t)}{\|\alpha'(t)\|}")
        st.latex(r"N(t)=\frac{1}{\|\alpha'(t)\|}\big(-y'(t),x'(t)\big)")
        st.latex(
            r"\kappa(t)=\frac{x'(t)y''(t)-y'(t)x''(t)}{\big(x'(t)^2+y'(t)^2\big)^{3/2}}"
        )
        st.latex(r"C(t_0)=\alpha(t_0)+\frac{1}{\kappa(t_0)}N(t_0)")
        st.markdown(
            "O círculo osculador é o círculo que melhor aproxima a curva até a segunda ordem no ponto escolhido."
        )

    with st.expander("Comprimento de arco e reparametrização", expanded=False):
        arc = cumulative_arc_length(data.alpha1, t)
        total_length = arc[-1]
        st.latex(r"s(t)=\int_{t_{\min}}^t\|\alpha'(u)\|\,du")
        st.latex(rf"L(\alpha)\approx {fmt(total_length)}")
        st.plotly_chart(
            make_scalar_plot(t, arc, "t", "s(t)", "Comprimento de arco acumulado"),
            use_container_width=True,
        )
        st.markdown(
            "Uma reparametrização altera a maneira como o traço é percorrido, mas não necessariamente altera o traço geométrico."
        )


# ============================================================
# TEOREMA FUNDAMENTAL DAS CURVAS PLANAS
# ============================================================


def planar_kappa_function(s: np.ndarray, name: str, params: Dict[str, float], custom_expr: str) -> np.ndarray:
    if name == "Curvatura nula":
        return np.zeros_like(s)
    if name == "Curvatura constante":
        return np.full_like(s, params["c"])
    if name == "Curvatura linear":
        return params["a"] * s
    if name == "Curvatura senoidal":
        return params["a"] * np.sin(params["b"] * s)
    if name == "Curvatura racional":
        return params["a"] / (1 + s**2)
    if name == "Personalizada":
        return eval_expr(custom_expr, "s", s)
    raise ValueError("Função curvatura desconhecida.")


def reconstruct_planar_curve(
    s: np.ndarray,
    kappa: np.ndarray,
    p0: np.ndarray,
    theta0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = theta0 + np.concatenate(([0.0], cumulative_trapezoid_local(kappa, s)))
    T = np.column_stack((np.cos(theta), np.sin(theta)))
    x = p0[0] + np.concatenate(([0.0], cumulative_trapezoid_local(T[:, 0], s)))
    y = p0[1] + np.concatenate(([0.0], cumulative_trapezoid_local(T[:, 1], s)))
    alpha = np.column_stack((x, y))
    N = np.column_stack((-T[:, 1], T[:, 0]))
    return alpha, theta, T, N


def render_planar_reconstruction() -> None:
    st.header("Teorema Fundamental das Curvas Planas")
    st.markdown(
        r"""
O estudante escolhe a função curvatura $\kappa(s)$. Pelo Teorema Fundamental das
Curvas Planas, depois de fixarmos um ponto inicial $p_0$ e uma direção tangente inicial,
existe uma única curva parametrizada pelo comprimento de arco que possui essa curvatura,
a menos de movimentos rígidos do plano.
"""
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.latex(r"\theta'(s)=\kappa(s)")
    with c2:
        st.latex(r"T(s)=(\cos\theta(s),\sin\theta(s))")
    with c3:
        st.latex(r"\alpha'(s)=T(s)")

    with st.expander("Como a reconstrução é realizada", expanded=False):
        st.markdown(
            r"""
Primeiro integramos a curvatura para encontrar o ângulo da direção tangente:

$$
\theta(s)=\theta_0+\int_{s_0}^{s}\kappa(u)\,du.
$$

Em seguida, construímos o vetor tangente unitário

$$
T(s)=(\cos\theta(s),\sin\theta(s)).
$$

Por fim, integramos $\alpha'(s)=T(s)$:

$$
\alpha(s)=p_0+\int_{s_0}^{s}T(u)\,du.
$$

Como $\|T(s)\|=1$, o parâmetro $s$ é o comprimento de arco.
"""
        )

    with st.sidebar:
        st.subheader("Curvatura prescrita")
        name = st.selectbox(
            r"Escolha $\kappa(s)$",
            [
                "Curvatura nula",
                "Curvatura constante",
                "Curvatura linear",
                "Curvatura senoidal",
                "Curvatura racional",
                "Personalizada",
            ],
            key="tf_planar_kappa_name",
        )
        params: Dict[str, float] = {}
        custom_expr = "0.5 + 0.2*sin(s)"
        if name == "Curvatura constante":
            params["c"] = st.slider("Constante c", -3.0, 3.0, 1.0, 0.1, key="tfp_c")
        elif name in {"Curvatura linear", "Curvatura senoidal", "Curvatura racional"}:
            params["a"] = st.slider("Parâmetro a", -3.0, 3.0, 0.5, 0.1, key=f"tfp_a_{name}")
            if name == "Curvatura senoidal":
                params["b"] = st.slider("Parâmetro b", 0.1, 5.0, 1.0, 0.1, key="tfp_b")
        elif name == "Personalizada":
            st.caption("Use s, sin, cos, exp, sqrt, pi, entre outras funções disponíveis.")
            custom_expr = st.text_input("κ(s)", custom_expr, key="tfp_custom")

        st.subheader("Intervalo e dados iniciais")
        smin = st.number_input("s mínimo", value=0.0, key="tfp_smin")
        smax = st.number_input("s máximo", value=12.0, key="tfp_smax")
        resolution = st.slider("Resolução", 300, 2000, 900, 100, key="tfp_resolution")
        x0 = st.number_input("x₀", value=0.0, key="tfp_x0")
        y0 = st.number_input("y₀", value=0.0, key="tfp_y0")
        theta0_deg = st.slider("Ângulo inicial θ₀ (graus)", -180, 180, 0, 5, key="tfp_theta0")

        if smax > smin:
            smove = st.slider("Ponto móvel s₀", float(smin), float(smax), float((smin + smax) / 2), key="tfp_smove")
        else:
            smove = float(smin)

        st.subheader("Objetos geométricos")
        vector_scale = st.slider("Escala dos vetores", 0.1, 3.0, 1.0, 0.1, key="tfp_scale")
        show_tangent = st.checkbox("Vetor e reta tangente", True, key="tfp_show_tangent")
        show_normal = st.checkbox("Vetor e reta normal", True, key="tfp_show_normal")
        show_circle = st.checkbox("Círculo osculador", True, key="tfp_show_circle")
        compare_rigid = st.checkbox("Comparar outra condição inicial", value=False, key="tfp_compare")
        if compare_rigid:
            x1 = st.number_input("Novo x₀", value=2.0, key="tfp_x1")
            y1 = st.number_input("Novo y₀", value=1.0, key="tfp_y1")
            theta1_deg = st.slider("Novo θ₀ (graus)", -180, 180, 45, 5, key="tfp_theta1")

    if smax <= smin:
        st.error(r"O domínio deve satisfazer $s_{\min}<s_{\max}$.")
        return

    s = np.linspace(smin, smax, resolution)
    try:
        kappa = planar_kappa_function(s, name, params, custom_expr)
        if kappa.shape != s.shape:
            kappa = np.broadcast_to(kappa, s.shape).astype(float)
        if not np.all(np.isfinite(kappa)):
            raise ValueError("A função curvatura produziu valores não finitos.")
        alpha, theta, T, N = reconstruct_planar_curve(
            s,
            kappa,
            np.array([x0, y0], dtype=float),
            math.radians(theta0_deg),
        )
    except Exception as exc:
        st.error(rf"Não foi possível reconstruir a curva: {exc}")
        return

    i0 = nearest_index(s, smove)
    p = alpha[i0]
    kap0 = float(kappa[i0])
    center = None
    radius = None
    if abs(kap0) > 1e-8:
        radius = 1.0 / abs(kap0)
        center = p + (1.0 / kap0) * N[i0]

    top1, top2 = st.columns(2)
    with top1:
        st.plotly_chart(make_scalar_plot(s, kappa, "s", "κ(s)", "Curvatura escolhida"), use_container_width=True)
    with top2:
        st.plotly_chart(make_scalar_plot(s, theta, "s", "θ(s)", "Direção tangente reconstruída"), use_container_width=True)

    fig = make_planar_plot(
        alpha=alpha,
        p=p,
        velocity=T[i0],
        T=T[i0],
        N=N[i0],
        vector_scale=vector_scale,
        show_velocity=False,
        show_tangent_vector=show_tangent,
        show_normal_vector=show_normal,
        show_tangent_line=show_tangent,
        show_normal_line=show_normal,
        show_circle=show_circle,
        center=center,
        radius=radius,
        equal_axes=True,
    )
    fig.data[0].name = "Curva reconstruída"
    fig.data[1].name = "Ponto α(s₀)"

    if compare_rigid:
        alpha2, _, _, _ = reconstruct_planar_curve(
            s,
            kappa,
            np.array([x1, y1], dtype=float),
            math.radians(theta1_deg),
        )
        fig.add_trace(go.Scatter(
            x=alpha2[:, 0], y=alpha2[:, 1], mode="lines",
            name="Mesma κ, outros dados iniciais", line=dict(width=4, dash="dash")
        ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Objetos no ponto selecionado")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("s₀", fmt(s[i0]))
    m2.metric("κ(s₀)", fmt(kap0))
    m3.metric("Raio de curvatura", fmt(radius) if radius is not None else "não definido")
    m4.metric("‖T(s₀)‖", fmt(np.linalg.norm(T[i0]), 8))

    d1, d2 = st.columns(2)
    with d1:
        st.latex(rf"\alpha(s_0)={vector_latex(p)}")
        st.latex(rf"T(s_0)={vector_latex(T[i0])}")
        st.latex(rf"N(s_0)={vector_latex(N[i0])}")
    with d2:
        st.latex(rf"\kappa(s_0)={fmt(kap0)}")
        if center is not None and radius is not None:
            st.latex(rf"\rho(s_0)=\frac{{1}}{{|\kappa(s_0)|}}={fmt(radius)}")
            st.latex(rf"C(s_0)=\alpha(s_0)+\frac{{1}}{{\kappa(s_0)}}N(s_0)={vector_latex(center)}")
        else:
            st.info("Como κ(s₀)=0, o círculo osculador não possui raio finito nesse ponto.")

    st.success(
        "A curva exibida foi obtida numericamente a partir da curvatura escolhida. "
        "Alterar o ponto ou a direção inicial apenas desloca ou gira a curva, sem alterar sua curvatura."
    )


# ============================================================
# ANÁLISE DE CURVAS ESPACIAIS
# ============================================================


def render_spatial_analysis() -> None:
    st.header(r"Curvas espaciais em $\mathbb{R}^3$")
    st.markdown(
        "Para uma curva espacial regular, o triedro de Frenet é formado pelos vetores tangente $T$, normal principal $N$ e binormal $B$."
    )

    with st.sidebar:
        st.subheader("Curva espacial")
        name = st.selectbox(
            "Escolha uma curva",
            [
                "Reta espacial",
                "Hélice circular",
                "Parábola espacial",
                "Cúbica torcida",
                "Curva toroidal",
                "Curva de Viviani",
                "Personalizada",
            ],
            key="spatial_name",
        )

        params: Dict[str, float] = {}
        custom: Optional[Dict[str, str]] = None

        if name == "Reta espacial":
            params["x0"] = st.number_input("x₀", value=0.0, key="sp_x0")
            params["y0"] = st.number_input("y₀", value=0.0, key="sp_y0")
            params["z0"] = st.number_input("z₀", value=0.0, key="sp_z0")
            params["vx"] = st.number_input("vₓ", value=1.0, key="sp_vx")
            params["vy"] = st.number_input("vᵧ", value=1.0, key="sp_vy")
            params["vz"] = st.number_input("v_z", value=1.0, key="sp_vz")
        elif name == "Hélice circular":
            params["a"] = st.slider("Raio a", 0.2, 4.0, 1.0, 0.1, key="sp_helix_a")
            params["b"] = st.slider("Passo b", -2.0, 2.0, 0.4, 0.1, key="sp_helix_b")
        elif name == "Parábola espacial":
            params["c"] = st.slider("Parâmetro c", -3.0, 3.0, 1.0, 0.1, key="sp_parab_c")
        elif name == "Curva toroidal":
            params["R"] = st.slider("Raio maior R", 1.0, 5.0, 2.5, 0.1, key="sp_R")
            params["r"] = st.slider("Raio menor r", 0.2, 2.0, 0.8, 0.1, key="sp_r")
            params["m"] = float(st.slider("Frequência m", 1, 7, 2, 1, key="sp_m"))
            params["n"] = float(st.slider("Frequência n", 1, 9, 3, 1, key="sp_n"))
        elif name == "Curva de Viviani":
            params["a"] = st.slider("Parâmetro a", 0.2, 4.0, 1.0, 0.1, key="sp_viviani_a")
        elif name == "Personalizada":
            st.caption("Funções: sin, cos, tan, sinh, cosh, exp, log, sqrt, abs, pi.")
            custom = {
                "x": st.text_input("x(t)", "cos(t)", key="sp_custom_x"),
                "y": st.text_input("y(t)", "sin(t)", key="sp_custom_y"),
                "z": st.text_input("z(t)", "0.3*t", key="sp_custom_z"),
            }

        default_domains = {
            "Reta espacial": (-3.0, 3.0),
            "Hélice circular": (-2 * np.pi, 2 * np.pi),
            "Parábola espacial": (-2.5, 2.5),
            "Cúbica torcida": (-2.0, 2.0),
            "Curva toroidal": (0.0, 2 * np.pi),
            "Curva de Viviani": (0.0, 4 * np.pi),
            "Personalizada": (-3.0, 3.0),
        }
        default_min, default_max = default_domains[name]

        st.subheader("Domínio e ponto")
        tmin = st.number_input("t mínimo", value=float(default_min), key=f"sp_tmin_{name}")
        tmax = st.number_input("t máximo", value=float(default_max), key=f"sp_tmax_{name}")
        resolution = st.slider("Resolução", 300, 1800, 800, 100, key="sp_resolution")

        if tmax > tmin:
            t0 = st.slider("Ponto t₀", float(tmin), float(tmax), float((tmin + tmax) / 2), key=f"sp_t0_{name}")
        else:
            t0 = float(tmin)

        st.subheader("Elementos visuais")
        vector_scale = st.slider("Escala do triedro", 0.1, 3.0, 1.0, 0.1, key="sp_scale")
        show_frame = st.checkbox("Mostrar triedro de Frenet", value=True, key="sp_show_frame")
        show_tangent_line = st.checkbox("Mostrar reta tangente", value=False, key="sp_show_tangent")

    if tmax <= tmin:
        st.error(r"O domínio deve satisfazer $t_{\min}<t_{\max}$.")
        return

    try:
        t = np.linspace(tmin, tmax, resolution)
        data = spatial_curve(t, name, params, custom)
    except Exception as exc:
        st.error(f"Não foi possível avaliar a curva: {exc}")
        return

    mask = finite_mask(data.alpha, data.alpha1, data.alpha2, data.alpha3)
    if np.count_nonzero(mask) < 5:
        st.error("Não há pontos válidos suficientes para construir o gráfico.")
        return

    i0 = nearest_index(t, t0)
    p = data.alpha[i0]
    v1 = data.alpha1[i0]
    v2 = data.alpha2[i0]
    v3 = data.alpha3[i0]

    T = safe_unit(v1)
    cross = np.cross(v1, v2)
    B = safe_unit(cross)
    N = np.cross(B, T) if B is not None and T is not None else None
    if N is not None:
        N = safe_unit(N)

    kappa_values = spatial_curvature(data.alpha1, data.alpha2)
    tau_values = spatial_torsion(data.alpha1, data.alpha2, data.alpha3)
    kappa = float(kappa_values[i0]) if np.isfinite(kappa_values[i0]) else np.nan
    tau = float(tau_values[i0]) if np.isfinite(tau_values[i0]) else np.nan

    st.subheader("Curva escolhida")
    st.latex(rf"\alpha:[{fmt(tmin,2)},{fmt(tmax,2)}]\longrightarrow\mathbb{{R}}^3")
    st.latex(data.latex)

    if data.is_numeric:
        st.caption("Algumas derivadas desta curva são aproximadas numericamente.")

    left, right = st.columns([1.5, 1.0])
    with left:
        st.plotly_chart(
            make_spatial_plot(data.alpha[mask], p, T, N, B, vector_scale, show_frame, show_tangent_line),
            use_container_width=True,
        )

    with right:
        st.subheader("Valores em $t_0$")
        st.latex(rf"t_0={fmt(t[i0])}")
        st.latex(rf"\alpha(t_0)={vector_latex(p)}")
        st.latex(rf"\alpha'(t_0)={vector_latex(v1)}")
        st.latex(rf"\alpha''(t_0)={vector_latex(v2)}")
        st.latex(rf"\alpha'''(t_0)={vector_latex(v3)}")

        if T is None:
            st.error("A curva não é regular nesse ponto.")
        else:
            st.latex(rf"T(t_0)={vector_latex(T)}")
            if B is None or N is None:
                st.info(
                    r"O triedro de Frenet clássico não está definido porque $\alpha'(t_0)\times\alpha''(t_0)=0$."
                )
            else:
                st.latex(rf"N(t_0)={vector_latex(N)}")
                st.latex(rf"B(t_0)={vector_latex(B)}")
            st.latex(rf"\kappa(t_0)={fmt(kappa)}")
            st.latex(rf"\tau(t_0)={fmt(tau)}")

    with st.expander("Fórmulas do triedro de Frenet", expanded=True):
        st.latex(r"T=\frac{\alpha'}{\|\alpha'\|}")
        st.latex(r"B=\frac{\alpha'\times\alpha''}{\|\alpha'\times\alpha''\|}")
        st.latex(r"N=B\times T")
        st.latex(r"\kappa=\frac{\|\alpha'\times\alpha''\|}{\|\alpha'\|^3}")
        st.latex(
            r"\tau=\frac{\langle\alpha'\times\alpha'',\alpha'''\rangle}{\|\alpha'\times\alpha''\|^2}"
        )
        st.warning(
            r"Em geral, o vetor normal principal não é obtido apenas normalizando $\alpha''$. Isso só funciona em situações especiais, como parametrizações pelo comprimento de arco."
        )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_scalar_plot(t, kappa_values, "t", "κ(t)", "Curvatura"), use_container_width=True)
    with col2:
        st.plotly_chart(make_scalar_plot(t, tau_values, "t", "τ(t)", "Torção"), use_container_width=True)


# ============================================================
# TEOREMA FUNDAMENTAL DAS CURVAS NO ESPAÇO
# ============================================================


def invariant_function(
    s: np.ndarray,
    name: str,
    params: Dict[str, float],
    custom_expr: str,
    kind: str,
) -> np.ndarray:
    if name == "Constante":
        return np.full_like(s, params["c"])
    if name == "Nula":
        return np.zeros_like(s)
    if name == "Linear":
        return params["a"] + params["b"] * s
    if name == "Senoidal":
        return params["a"] + params["b"] * np.sin(params["c"] * s)
    if name == "Racional":
        return params["a"] / (1 + params["b"] * s**2)
    if name == "Personalizada":
        return eval_expr(custom_expr, "s", s)
    raise ValueError(f"Função {kind} desconhecida.")


def orthonormal_frame(v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T0 = safe_unit(v1)
    if T0 is None:
        raise ValueError("O vetor inicial para T não pode ser nulo.")
    v2_orth = v2 - np.dot(v2, T0) * T0
    N0 = safe_unit(v2_orth)
    if N0 is None:
        raise ValueError("Os vetores iniciais usados para T e N não podem ser paralelos.")
    B0 = safe_unit(np.cross(T0, N0))
    if B0 is None:
        raise ValueError("Não foi possível construir o vetor binormal inicial.")
    N0 = np.cross(B0, T0)
    return T0, N0, B0


def solve_frenet_system(
    s: np.ndarray,
    kappa: np.ndarray,
    tau: np.ndarray,
    p0: np.ndarray,
    T0: np.ndarray,
    N0: np.ndarray,
    B0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def kappa_interp(x: float) -> float:
        return float(np.interp(x, s, kappa))

    def tau_interp(x: float) -> float:
        return float(np.interp(x, s, tau))

    def rhs(x: float, y: np.ndarray) -> np.ndarray:
        T = y[3:6]
        N = y[6:9]
        B = y[9:12]
        kap = kappa_interp(x)
        tor = tau_interp(x)
        d_alpha = T
        dT = kap * N
        dN = -kap * T + tor * B
        dB = -tor * N
        return np.concatenate((d_alpha, dT, dN, dB))

    y0 = np.concatenate((p0, T0, N0, B0))
    values = rk4_on_grid(rhs, s, y0)
    alpha = values[:, 0:3]
    T_raw = values[:, 3:6]
    N_raw = values[:, 6:9]

    T = np.zeros_like(T_raw)
    N = np.zeros_like(N_raw)
    B = np.zeros_like(T_raw)
    for i in range(len(s)):
        Ti, Ni, Bi = orthonormal_frame(T_raw[i], N_raw[i])
        T[i], N[i], B[i] = Ti, Ni, Bi

    return alpha, T, N, B


def render_spatial_reconstruction() -> None:
    st.header("Teorema Fundamental das Curvas no Espaço")
    st.markdown(
        r"""
O estudante escolhe a curvatura $\kappa(s)>0$ e a torção $\tau(s)$. Pelo Teorema
Fundamental das Curvas no Espaço, depois de fixados o ponto e o triedro inicial, existe
uma única curva parametrizada pelo comprimento de arco que possui esses invariantes,
a menos de uma isometria de $\mathbb{R}^3$.
"""
    )

    st.latex(r"\alpha'=T,\qquad T'=\kappa N,\qquad N'=-\kappa T+\tau B,\qquad B'=-\tau N")

    with st.expander("Como a curva espacial é encontrada", expanded=False):
        st.markdown(
            r"""
O site resolve numericamente o sistema de Frenet–Serret. A curvatura mede quanto a
curva se afasta de sua reta tangente, enquanto a torção mede quanto ela se afasta do
plano osculador. Se $\tau\equiv0$, a curva reconstruída é plana; se $\kappa$ e $\tau$
são constantes positivas, obtemos uma hélice circular, a menos de movimento rígido.
"""
        )

    with st.sidebar:
        st.subheader("Curvatura κ(s)")
        k_name = st.selectbox("Tipo de κ", ["Constante", "Linear", "Senoidal", "Racional", "Personalizada"], key="tfs_k_name")
        k_params: Dict[str, float] = {}
        k_custom = "1 + 0.2*sin(s)"
        if k_name == "Constante":
            k_params["c"] = st.slider("κ constante", 0.05, 3.0, 1.0, 0.05, key="tfs_k_c")
        elif k_name == "Linear":
            k_params["a"] = st.slider("κ: termo constante", 0.05, 2.0, 0.8, 0.05, key="tfs_k_a")
            k_params["b"] = st.slider("κ: coeficiente linear", -0.2, 0.2, 0.02, 0.01, key="tfs_k_b")
        elif k_name == "Senoidal":
            k_params["a"] = st.slider("κ: nível médio", 0.1, 2.0, 1.0, 0.05, key="tfs_k_sa")
            k_params["b"] = st.slider("κ: amplitude", 0.0, 0.9, 0.25, 0.05, key="tfs_k_sb")
            k_params["c"] = st.slider("κ: frequência", 0.1, 4.0, 1.0, 0.1, key="tfs_k_sc")
        elif k_name == "Racional":
            k_params["a"] = st.slider("κ: numerador", 0.1, 3.0, 1.0, 0.1, key="tfs_k_ra")
            k_params["b"] = st.slider("κ: parâmetro", 0.05, 2.0, 0.2, 0.05, key="tfs_k_rb")
        else:
            st.caption("A expressão deve permanecer positiva em todo o intervalo.")
            k_custom = st.text_input("κ(s)", k_custom, key="tfs_k_custom")

        st.subheader("Torção τ(s)")
        t_name = st.selectbox("Tipo de τ", ["Nula", "Constante", "Linear", "Senoidal", "Racional", "Personalizada"], key="tfs_t_name")
        t_params: Dict[str, float] = {}
        t_custom = "0.3*cos(s)"
        if t_name == "Constante":
            t_params["c"] = st.slider("τ constante", -2.0, 2.0, 0.4, 0.05, key="tfs_t_c")
        elif t_name == "Linear":
            t_params["a"] = st.slider("τ: termo constante", -1.0, 1.0, 0.0, 0.05, key="tfs_t_a")
            t_params["b"] = st.slider("τ: coeficiente linear", -0.2, 0.2, 0.03, 0.01, key="tfs_t_b")
        elif t_name == "Senoidal":
            t_params["a"] = st.slider("τ: nível médio", -1.0, 1.0, 0.0, 0.05, key="tfs_t_sa")
            t_params["b"] = st.slider("τ: amplitude", 0.0, 1.5, 0.4, 0.05, key="tfs_t_sb")
            t_params["c"] = st.slider("τ: frequência", 0.1, 4.0, 1.0, 0.1, key="tfs_t_sc")
        elif t_name == "Racional":
            t_params["a"] = st.slider("τ: numerador", -2.0, 2.0, 0.5, 0.05, key="tfs_t_ra")
            t_params["b"] = st.slider("τ: parâmetro", 0.05, 2.0, 0.2, 0.05, key="tfs_t_rb")
        elif t_name == "Personalizada":
            t_custom = st.text_input("τ(s)", t_custom, key="tfs_t_custom")

        st.subheader("Intervalo e dados iniciais")
        smin = st.number_input("s mínimo", value=0.0, key="tfs_smin")
        smax = st.number_input("s máximo", value=15.0, key="tfs_smax")
        resolution = st.slider("Resolução", 300, 1600, 700, 100, key="tfs_resolution")
        p0 = np.array([
            st.number_input("x₀", value=0.0, key="tfs_x0"),
            st.number_input("y₀", value=0.0, key="tfs_y0"),
            st.number_input("z₀", value=0.0, key="tfs_z0"),
        ], dtype=float)

        st.subheader("Triedro inicial")
        frame_option = st.selectbox("Dados iniciais", ["Triedro canônico", "Personalizado"], key="tfs_frame_option")
        if frame_option == "Triedro canônico":
            v1 = np.array([1.0, 0.0, 0.0])
            v2 = np.array([0.0, 1.0, 0.0])
        else:
            st.caption("Os vetores serão ortonormalizados automaticamente.")
            v1 = np.array([
                st.number_input("T inicial: x", value=1.0, key="tfs_Tx"),
                st.number_input("T inicial: y", value=0.0, key="tfs_Ty"),
                st.number_input("T inicial: z", value=0.0, key="tfs_Tz"),
            ])
            v2 = np.array([
                st.number_input("N auxiliar: x", value=0.0, key="tfs_Nx"),
                st.number_input("N auxiliar: y", value=1.0, key="tfs_Ny"),
                st.number_input("N auxiliar: z", value=0.0, key="tfs_Nz"),
            ])

        if smax > smin:
            smove = st.slider("Ponto móvel s₀", float(smin), float(smax), float((smin + smax) / 2), key="tfs_smove")
        else:
            smove = float(smin)

        st.subheader("Objetos geométricos")
        vector_scale = st.slider("Escala do triedro", 0.1, 3.0, 1.0, 0.1, key="tfs_scale")
        show_tangent = st.checkbox("Reta tangente", True, key="tfs_show_tangent")
        show_osculating = st.checkbox("Plano osculador", False, key="tfs_show_osc")
        show_normal_plane = st.checkbox("Plano normal", False, key="tfs_show_normal_plane")
        show_rectifying = st.checkbox("Plano retificante", False, key="tfs_show_rect")

    if smax <= smin:
        st.error(r"O domínio deve satisfazer $s_{\min}<s_{\max}$.")
        return

    s = np.linspace(smin, smax, resolution)
    try:
        kappa = invariant_function(s, k_name, k_params, k_custom, "curvatura")
        tau = invariant_function(s, t_name, t_params, t_custom, "torção")
        kappa = np.broadcast_to(kappa, s.shape).astype(float)
        tau = np.broadcast_to(tau, s.shape).astype(float)
        if not np.all(np.isfinite(kappa)) or not np.all(np.isfinite(tau)):
            raise ValueError("As funções produziram valores não finitos.")
        if np.any(kappa <= 0):
            raise ValueError("O Teorema Fundamental clássico requer κ(s)>0 em todo o intervalo.")
        T0, N0, B0 = orthonormal_frame(v1, v2)
        alpha, T, N, B = solve_frenet_system(s, kappa, tau, p0, T0, N0, B0)
    except Exception as exc:
        st.error(f"Não foi possível reconstruir a curva: {exc}")
        return

    i0 = nearest_index(s, smove)
    p = alpha[i0]

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_scalar_plot(s, kappa, "s", "κ(s)", "Curvatura escolhida"), use_container_width=True)
    with col2:
        st.plotly_chart(make_scalar_plot(s, tau, "s", "τ(s)", "Torção escolhida"), use_container_width=True)

    fig = make_spatial_plot(alpha, p, T[i0], N[i0], B[i0], vector_scale, True, show_tangent)
    fig.data[0].name = "Curva reconstruída"
    fig.data[1].name = "Ponto α(s₀)"

    extent = max(np.ptp(alpha[:, 0]), np.ptp(alpha[:, 1]), np.ptp(alpha[:, 2]), 1.0)
    plane_size = 0.22 * extent
    grid = np.linspace(-plane_size, plane_size, 8)
    U, V = np.meshgrid(grid, grid)

    def add_plane(e1: np.ndarray, e2: np.ndarray, name: str, opacity: float = 0.25) -> None:
        points = p[None, None, :] + U[:, :, None] * e1[None, None, :] + V[:, :, None] * e2[None, None, :]
        fig.add_trace(go.Surface(
            x=points[:, :, 0], y=points[:, :, 1], z=points[:, :, 2],
            name=name, showscale=False, opacity=opacity, showlegend=True,
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
        ))

    if show_osculating:
        add_plane(T[i0], N[i0], "Plano osculador")
    if show_normal_plane:
        add_plane(N[i0], B[i0], "Plano normal")
    if show_rectifying:
        add_plane(T[i0], B[i0], "Plano retificante")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Objetos no ponto selecionado")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("s₀", fmt(s[i0]))
    m2.metric("κ(s₀)", fmt(kappa[i0]))
    m3.metric("τ(s₀)", fmt(tau[i0]))
    m4.metric("Raio 1/κ", fmt(1.0 / kappa[i0]))

    d1, d2 = st.columns(2)
    with d1:
        st.latex(rf"\alpha(s_0)={vector_latex(p)}")
        st.latex(rf"T(s_0)={vector_latex(T[i0])}")
        st.latex(rf"N(s_0)={vector_latex(N[i0])}")
        st.latex(rf"B(s_0)={vector_latex(B[i0])}")
    with d2:
        st.latex(rf"\|T(s_0)\|={fmt(np.linalg.norm(T[i0]), 8)}")
        st.latex(rf"\langle T,N\rangle={fmt(np.dot(T[i0], N[i0]), 8)}")
        st.latex(rf"\langle T,B\rangle={fmt(np.dot(T[i0], B[i0]), 8)}")
        st.latex(rf"\langle N,B\rangle={fmt(np.dot(N[i0], B[i0]), 8)}")

    if np.max(np.abs(tau)) < 1e-8:
        st.info("Como τ(s)=0 em todo o intervalo, a curva reconstruída é plana.")
    elif np.max(kappa) - np.min(kappa) < 1e-8 and np.max(tau) - np.min(tau) < 1e-8:
        st.info("Como κ e τ são constantes, a curva reconstruída é uma hélice circular, a menos de movimento rígido.")

    st.success(
        "A curva espacial exibida foi encontrada numericamente resolvendo o sistema de Frenet–Serret "
        "para as funções κ(s) e τ(s) escolhidas."
    )


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

st.title("Curvas planas e espaciais")
st.markdown(
    """
Neste módulo, estudamos curvas parametrizadas, seus vetores associados, curvatura, torção e os Teoremas Fundamentais das Curvas.
"""
)
st.latex(r"\alpha:I\subset\mathbb{R}\longrightarrow\mathbb{R}^n")

render_fundamental_concepts()

with st.sidebar:
    st.divider()
    st.header("Ambiente de estudo")
    environment = st.radio(
        "Escolha o espaço",
        ["Curvas planas em ℝ²", "Curvas espaciais em ℝ³"],
        key="environment",
    )
    mode = st.radio(
        "Escolha o modo",
        ["Analisar uma curva conhecida", "Reconstruir pelo Teorema Fundamental"],
        key="mode",
    )

if environment == "Curvas planas em ℝ²":
    if mode == "Analisar uma curva conhecida":
        render_planar_analysis()
    else:
        render_planar_reconstruction()
else:
    if mode == "Analisar uma curva conhecida":
        render_spatial_analysis()
    else:
        render_spatial_reconstruction()
