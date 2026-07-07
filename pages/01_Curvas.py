import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(
    page_title="Curvas em R³",
    page_icon="📈",
    layout="wide"
)

with st.sidebar:

    st.title("📐 Geometria Diferencial")

    st.page_link("app.py", label="🏠 Início")
    st.page_link("pages/01_Curvas.py", label="1. Curvas em R³")

    st.markdown("2. Superfícies Regulares")
    st.markdown("3. Plano Tangente e Normal")
    st.markdown("4. Primeira Forma Fundamental")
    st.markdown("5. Segunda Forma Fundamental")
    st.markdown("6. Curvaturas")
    st.markdown("7. Superfícies Mínimas")
    st.markdown("8. Variação da Área")

def fmt(x, digits=5):
    try:
        x = float(x)
        if not np.isfinite(x):
            return "não definido"
        return f"{x:.{digits}f}"
    except Exception:
        return "não definido"


def norm(v):
    return np.linalg.norm(v, axis=-1)


def safe_unit(v):
    n = np.linalg.norm(v)
    if n < 1e-10:
        return np.zeros_like(v)
    return v / n


def curve(t, name, a=1.0, b=0.5):
    if name == "Reta":
        return np.column_stack((t, t, t))

    if name == "Circunferência":
        return np.column_stack((a * np.cos(t), a * np.sin(t), 0 * t))

    if name == "Hélice circular":
        return np.column_stack((a * np.cos(t), a * np.sin(t), b * t))

    if name == "Parábola espacial":
        return np.column_stack((t, t**2, t**3))

    if name == "Curva toroidal":
        return np.column_stack(
            (
                (2 + np.cos(3 * t)) * np.cos(t),
                (2 + np.cos(3 * t)) * np.sin(t),
                np.sin(3 * t),
            )
        )

    return np.column_stack((t, t, t))


def derivatives(alpha, dt):
    alpha1 = np.gradient(alpha, dt, axis=0)
    alpha2 = np.gradient(alpha1, dt, axis=0)
    alpha3 = np.gradient(alpha2, dt, axis=0)
    return alpha1, alpha2, alpha3


def curvature(alpha1, alpha2):
    cross = np.cross(alpha1, alpha2)
    numerator = norm(cross)
    denominator = norm(alpha1) ** 3
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-10
    )


def torsion(alpha1, alpha2, alpha3):
    cross = np.cross(alpha1, alpha2)
    numerator = np.einsum("ij,ij->i", cross, alpha3)
    denominator = norm(cross) ** 2
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-10
    )


def make_plot(alpha, p, T, N, B, show_frame=True):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=alpha[:, 0],
            y=alpha[:, 1],
            z=alpha[:, 2],
            mode="lines",
            name="curva",
            line=dict(width=6),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[p[0]],
            y=[p[1]],
            z=[p[2]],
            mode="markers",
            name="ponto escolhido",
            marker=dict(size=6),
        )
    )

    if show_frame:
        scale = 0.8

        vectors = [
            ("T: tangente", T),
            ("N: normal", N),
            ("B: binormal", B),
        ]

        for label, v in vectors:
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0], p[0] + scale * v[0]],
                    y=[p[1], p[1] + scale * v[1]],
                    z=[p[2], p[2] + scale * v[2]],
                    mode="lines+markers",
                    name=label,
                    line=dict(width=5),
                )
            )

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
    )

    return fig


st.title("Módulo 1 — Curvas em R³")

st.markdown(
    r"""
Neste módulo estudamos curvas parametrizadas

\[
\alpha:I\subset\mathbb{R}\longrightarrow \mathbb{R}^3,
\qquad
\alpha(t)=(x(t),y(t),z(t)).
\]

A curva é regular quando

\[
\alpha'(t)\neq 0.
\]

A partir das derivadas de \(\alpha\), calculamos vetor tangente, curvatura,
torção e o triedro de Frenet.
"""
)

with st.sidebar:
    st.header("Curva")

    name = st.selectbox(
        "Escolha uma curva",
        [
            "Reta",
            "Circunferência",
            "Hélice circular",
            "Parábola espacial",
            "Curva toroidal",
        ],
    )

    a = st.slider("Parâmetro a", 0.2, 3.0, 1.0, 0.1)
    b = st.slider("Parâmetro b", -2.0, 2.0, 0.5, 0.1)

    st.header("Domínio")

    tmin = st.number_input("t mínimo", value=-6.0)
    tmax = st.number_input("t máximo", value=6.0)

    n = st.slider("Resolução", 100, 1000, 400, 50)

    st.header("Ponto de leitura")

    t0 = st.slider("Escolha o ponto t₀", float(tmin), float(tmax), float((tmin + tmax) / 2))

    show_frame = st.checkbox("Mostrar triedro de Frenet", value=True)


if tmax <= tmin:
    st.error("O domínio precisa satisfazer t mínimo < t máximo.")
    st.stop()


t = np.linspace(tmin, tmax, n)
dt = t[1] - t[0]

alpha = curve(t, name, a, b)
alpha1, alpha2, alpha3 = derivatives(alpha, dt)

kappa = curvature(alpha1, alpha2)
tau = torsion(alpha1, alpha2, alpha3)

i0 = int(np.argmin(np.abs(t - t0)))

p = alpha[i0]
v1 = alpha1[i0]
v2 = alpha2[i0]

T = safe_unit(v1)

k0 = kappa[i0]
if np.isfinite(k0) and k0 > 1e-8:
    N = safe_unit(v2)
else:
    N = np.zeros(3)

B = safe_unit(np.cross(T, N))

left, right = st.columns([2.2, 1])

with left:
    st.plotly_chart(
        make_plot(alpha, p, T, N, B, show_frame),
        use_container_width=True
    )

with right:
    st.subheader("Valores no ponto escolhido")

    st.write(rf"\(t_0={fmt(t[i0])}\)")

    st.write(
        rf"\(\alpha(t_0)=({fmt(p[0])}, {fmt(p[1])}, {fmt(p[2])})\)"
    )

    st.write(
        rf"\(\alpha'(t_0)=({fmt(v1[0])}, {fmt(v1[1])}, {fmt(v1[2])})\)"
    )

    st.write(rf"\(|\alpha'(t_0)|={fmt(np.linalg.norm(v1))}\)")

    st.metric("Curvatura κ(t₀)", fmt(kappa[i0]))
    st.metric("Torção τ(t₀)", fmt(tau[i0]))

    df = pd.DataFrame(
        {
            "Vetor": ["T", "N", "B"],
            "x": [fmt(T[0]), fmt(N[0]), fmt(B[0])],
            "y": [fmt(T[1]), fmt(N[1]), fmt(B[1])],
            "z": [fmt(T[2]), fmt(N[2]), fmt(B[2])],
        }
    )

    st.dataframe(df, hide_index=True, use_container_width=True)


st.subheader("Fórmulas usadas")

st.latex(r"T(t)=\frac{\alpha'(t)}{|\alpha'(t)|}")

st.latex(
    r"\kappa(t)=\frac{|\alpha'(t)\times \alpha''(t)|}{|\alpha'(t)|^3}"
)

st.latex(
    r"\tau(t)=\frac{\langle \alpha'(t)\times \alpha''(t),\alpha'''(t)\rangle}{|\alpha'(t)\times \alpha''(t)|^2}"
)

st.subheader("Interpretação geométrica")

st.write(
    """
    O vetor tangente indica a direção instantânea da curva.
    A curvatura mede o quanto a curva muda de direção.
    A torção mede o quanto a curva sai de um plano.
    """
)

st.info(
    """
    Na hélice circular, a curvatura e a torção aparecem como quantidades constantes,
    o que está de acordo com o exemplo clássico estudado no capítulo de curvas.
    """
)
