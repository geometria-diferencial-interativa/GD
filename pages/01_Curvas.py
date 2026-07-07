import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fractions import Fraction


st.set_page_config(
    page_title="Curvas em R³",
    page_icon="📈",
    layout="wide",
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


def frac(x, max_denominator=1000):
    try:
        x = float(x)
        if not np.isfinite(x):
            return r"\text{não definido}"
        return str(Fraction(x).limit_denominator(max_denominator))
    except Exception:
        return r"\text{não definido}"


def norm(v):
    return np.linalg.norm(v, axis=-1)


def safe_unit(v):
    n = np.linalg.norm(v)
    if n < 1e-10:
        return np.zeros_like(v)
    return v / n


def eval_expr(expr, t):
    allowed = {
        "t": t,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
    }
    return eval(expr, {"__builtins__": {}}, allowed)


def curve(t, name, a=1.0, b=0.5, custom=None):
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

    if name == "Personalizada":
        x = eval_expr(custom["x"], t)
        y = eval_expr(custom["y"], t)
        z = eval_expr(custom["z"], t)
        return np.column_stack((x, y, z))

    return np.column_stack((t, t, t))


def curve_latex(name, custom=None):
    if name == "Reta":
        return r"\alpha(t)=(t,t,t)"
    if name == "Circunferência":
        return r"\alpha(t)=(a\cos t,\;a\sin t,\;0)"
    if name == "Hélice circular":
        return r"\alpha(t)=(a\cos t,\;a\sin t,\;bt)"
    if name == "Parábola espacial":
        return r"\alpha(t)=(t,\;t^2,\;t^3)"
    if name == "Curva toroidal":
        return r"\alpha(t)=((2+\cos(3t))\cos t,\;(2+\cos(3t))\sin t,\;\sin(3t))"
    if name == "Personalizada":
        return rf"\alpha(t)=\left({custom['x']},\;{custom['y']},\;{custom['z']}\right)"
    return r"\alpha(t)=(x(t),y(t),z(t))"


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
        where=denominator > 1e-10,
    )


def torsion(alpha1, alpha2, alpha3):
    cross = np.cross(alpha1, alpha2)
    numerator = np.einsum("ij,ij->i", cross, alpha3)
    denominator = norm(cross) ** 2
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-10,
    )


def add_arrow(fig, p, v, name, scale=1.0):
    if np.linalg.norm(v) < 1e-10:
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
            sizemode="absolute",
            sizeref=0.25,
            anchor="tip",
            showscale=False,
            name=name,
        )
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
        add_arrow(fig, p, T, "T: tangente", scale=0.9)
        add_arrow(fig, p, N, "N: normal", scale=0.9)
        add_arrow(fig, p, B, "B: binormal", scale=0.9)

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

st.write("Neste módulo estudamos curvas parametrizadas:")
st.latex(r"\alpha:I\subset\mathbb{R}\longrightarrow \mathbb{R}^3")
st.latex(r"\alpha(t)=(x(t),y(t),z(t))")
st.write("A curva é regular quando:")
st.latex(r"\alpha'(t)\neq 0")

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
            "Personalizada",
        ],
    )

    a = st.slider("Parâmetro a", 0.2, 3.0, 1.0, 0.1)
    b = st.slider("Parâmetro b", -2.0, 2.0, 0.5, 0.1)

    custom = {}

    if name == "Personalizada":
        st.markdown("Digite as coordenadas usando a variável `t`.")
        st.caption("Funções disponíveis: sin, cos, tan, exp, log, sqrt, pi.")
        custom["x"] = st.text_input("x(t)", "cos(t)")
        custom["y"] = st.text_input("y(t)", "sin(t)")
        custom["z"] = st.text_input("z(t)", "t")

    st.header("Domínio")

    tmin = st.number_input("t mínimo", value=-6.0)
    tmax = st.number_input("t máximo", value=6.0)

    n = st.slider("Resolução", 100, 1000, 400, 50)

    st.header("Ponto de leitura")

    t0 = st.slider(
        "Escolha o ponto t₀",
        float(tmin),
        float(tmax),
        float((tmin + tmax) / 2),
    )

    show_frame = st.checkbox("Mostrar triedro de Frenet", value=True)


if tmax <= tmin:
    st.error("O domínio precisa satisfazer t mínimo < t máximo.")
    st.stop()


try:
    t = np.linspace(tmin, tmax, n)
    dt = t[1] - t[0]

    alpha = curve(t, name, a, b, custom)
    alpha1, alpha2, alpha3 = derivatives(alpha, dt)

    kappa = curvature(alpha1, alpha2)
    tau = torsion(alpha1, alpha2, alpha3)

    i0 = int(np.argmin(np.abs(t - t0)))

    p = alpha[i0]
    v1 = alpha1[i0]
    v2 = alpha2[i0]

    T = safe_unit(v1)

    if np.isfinite(kappa[i0]) and kappa[i0] > 1e-8:
        N = safe_unit(v2)
    else:
        N = np.zeros(3)

    B = safe_unit(np.cross(T, N))

    st.subheader("Curva escolhida")
    st.markdown("A curva visualizada no gráfico é a aplicação parametrizada:")

    st.latex(
        rf"\alpha:I\subset\mathbb{{R}}\longrightarrow \mathbb{{R}}^3,"
        rf"\qquad I=[{fmt(tmin,2)},\,{fmt(tmax,2)}]"
    )

    st.latex(curve_latex(name, custom))

    st.markdown("Portanto, para cada valor de \(t\in I\), obtemos um ponto")
    st.latex(r"\alpha(t)=(x(t),y(t),z(t))\in\mathbb{R}^3.")

    if name == "Reta":
        st.markdown("Nesta parametrização, \(t\) controla simultaneamente as três coordenadas.")
    elif name == "Circunferência":
        st.markdown(f"Nesta parametrização, \(a={fmt(a,2)}\) é o raio da circunferência.")
    elif name == "Hélice circular":
        st.markdown(
            f"Nesta parametrização, \(a={fmt(a,2)}\) é o raio da hélice "
            f"e \(b={fmt(b,2)}\) controla a inclinação vertical."
        )
    elif name == "Parábola espacial":
        st.markdown("Nesta parametrização, as coordenadas crescem como \(t\), \(t^2\) e \(t^3\).")
    elif name == "Curva toroidal":
        st.markdown("Nesta parametrização, a curva se enrola sobre uma região toroidal.")
    elif name == "Personalizada":
        st.markdown("Esta curva foi definida pelo estudante a partir das três funções coordenadas.")

    left, right = st.columns([2.2, 1])

    with left:
        st.plotly_chart(
            make_plot(alpha, p, T, N, B, show_frame),
            use_container_width=True,
        )

    with right:
        st.subheader("Valores no ponto escolhido")

        st.markdown("**Parâmetro escolhido:**")
        st.latex(rf"t_0={fmt(t[i0])}")

        st.markdown("**Ponto da curva:**")
        st.latex(rf"\alpha(t_0)=\left({fmt(p[0])},\,{fmt(p[1])},\,{fmt(p[2])}\right)")

        st.markdown("**Vetor velocidade:**")
        st.latex(rf"\alpha'(t_0)=\left({fmt(v1[0])},\,{fmt(v1[1])},\,{fmt(v1[2])}\right)")

        st.markdown("**Norma da velocidade:**")
        st.latex(rf"|\alpha'(t_0)|={fmt(np.linalg.norm(v1))}")

        st.markdown("**Curvatura no ponto:**")
        st.latex(rf"\kappa(t_0)\approx {fmt(kappa[i0])}")
        st.latex(rf"\kappa(t_0)\approx {frac(kappa[i0])}")

        st.markdown("**Torção no ponto:**")
        st.latex(rf"\tau(t_0)\approx {fmt(tau[i0])}")
        st.latex(rf"\tau(t_0)\approx {frac(tau[i0])}")

        df = pd.DataFrame(
            {
                "Vetor": ["T", "N", "B"],
                "Coordenada x": [fmt(T[0]), fmt(N[0]), fmt(B[0])],
                "Coordenada y": [fmt(T[1]), fmt(N[1]), fmt(B[1])],
                "Coordenada z": [fmt(T[2]), fmt(N[2]), fmt(B[2])],
            }
        )

        st.subheader("Triedro de Frenet")
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Fórmulas usadas")

    st.latex(r"T(t)=\frac{\alpha'(t)}{|\alpha'(t)|}")
    st.latex(r"\kappa(t)=\frac{|\alpha'(t)\times \alpha''(t)|}{|\alpha'(t)|^3}")
    st.latex(
        r"\tau(t)=\frac{\langle \alpha'(t)\times \alpha''(t),\alpha'''(t)\rangle}"
        r"{|\alpha'(t)\times \alpha''(t)|^2}"
    )

except Exception as e:
    st.error("Não foi possível gerar a curva. Verifique as expressões e o domínio escolhidos.")
    st.exception(e)
