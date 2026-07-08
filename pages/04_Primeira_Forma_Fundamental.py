import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(
    page_title="Primeira Forma Fundamental",
    page_icon="📏",
    layout="wide",
)


# ============================================================
# MENU LATERAL
# ============================================================

with st.sidebar:
    st.title("📐 Geometria Diferencial")

    st.page_link("app.py", label="🏠 Início")
    st.page_link("pages/01_Curvas.py", label="1. Curvas em R³")
    st.page_link("pages/02_Superficies_Regulares.py", label="2. Superfícies Regulares")
    st.page_link("pages/03_Plano_Tangente_Normal.py", label="3. Plano Tangente e Vetor Normal")
    st.page_link("pages/04_Primeira_Forma_Fundamental.py", label="4. Primeira Forma Fundamental")

    st.markdown("5. Segunda Forma Fundamental")
    st.markdown("6. Curvaturas")
    st.markdown("7. Superfícies Mínimas")
    st.markdown("8. Variação da Área")


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def fmt(x, digits=5):
    try:
        x = float(x)
        if not np.isfinite(x):
            return "não definido"
        return f"{x:.{digits}f}"
    except Exception:
        return "não definido"


def eval_expr(expr, U, V):
    allowed = {
        "u": U,
        "v": V,
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


def make_grid(umin, umax, vmin, vmax, n):
    u = np.linspace(umin, umax, n)
    v = np.linspace(vmin, vmax, n)
    return np.meshgrid(u, v, indexing="ij")


def surface(U, V, name, a=1.0, R=1.0, r=0.35, custom=None):
    if name == "Plano":
        return np.stack((U, V, 0 * U), axis=-1)

    if name == "Esfera":
        return np.stack(
            (
                R * np.cos(U) * np.sin(V),
                R * np.sin(U) * np.sin(V),
                R * np.cos(V),
            ),
            axis=-1,
        )

    if name == "Cilindro":
        return np.stack((R * np.cos(U), R * np.sin(U), V), axis=-1)

    if name == "Catenoide":
        return np.stack(
            (
                a * np.cosh(V / a) * np.cos(U),
                a * np.cosh(V / a) * np.sin(U),
                V,
            ),
            axis=-1,
        )

    if name == "Helicoide":
        return np.stack((U * np.cos(V), U * np.sin(V), a * V), axis=-1)

    if name == "Toro":
        return np.stack(
            (
                (R + r * np.cos(V)) * np.cos(U),
                (R + r * np.cos(V)) * np.sin(U),
                r * np.sin(V),
            ),
            axis=-1,
        )

    if name == "Personalizada":
        return np.stack(
            (
                eval_expr(custom["x"], U, V),
                eval_expr(custom["y"], U, V),
                eval_expr(custom["z"], U, V),
            ),
            axis=-1,
        )

    return np.stack((U, V, 0 * U), axis=-1)


def surface_latex(name, custom=None):
    if name == "Plano":
        return r"X(u,v)=(u,v,0)"
    if name == "Esfera":
        return r"X(u,v)=(R\cos u\sin v,\;R\sin u\sin v,\;R\cos v)"
    if name == "Cilindro":
        return r"X(u,v)=(R\cos u,\;R\sin u,\;v)"
    if name == "Catenoide":
        return r"X(u,v)=(a\cosh(v/a)\cos u,\;a\cosh(v/a)\sin u,\;v)"
    if name == "Helicoide":
        return r"X(u,v)=(u\cos v,\;u\sin v,\;av)"
    if name == "Toro":
        return r"X(u,v)=((R+r\cos v)\cos u,\;(R+r\cos v)\sin u,\;r\sin v)"
    if name == "Personalizada":
        return rf"X(u,v)=\left({custom['x']},\;{custom['y']},\;{custom['z']}\right)"
    return r"X(u,v)=(x(u,v),y(u,v),z(u,v))"


def default_domain(name):
    if name in ["Esfera", "Cilindro", "Toro", "Catenoide"]:
        return 0.0, float(2 * np.pi), -2.0, 2.0
    if name == "Helicoide":
        return -2.0, 2.0, 0.0, float(4 * np.pi)
    return -2.0, 2.0, -2.0, 2.0


def numerical_derivatives(X, du, dv):
    Xu = np.gradient(X, du, axis=0)
    Xv = np.gradient(X, dv, axis=1)
    return Xu, Xv


def closest_index(grid, value, axis):
    line = grid[:, 0] if axis == 0 else grid[0, :]
    return int(np.argmin(np.abs(line - value)))


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-10:
        return np.zeros_like(v)
    return v / n


def add_arrow(fig, p, v, label, scale=0.7):
    n = np.linalg.norm(v)
    if n < 1e-10:
        return

    v = v / n
    q = p + scale * v

    fig.add_trace(
        go.Scatter3d(
            x=[p[0], q[0]],
            y=[p[1], q[1]],
            z=[p[2], q[2]],
            mode="lines",
            name=label,
            showlegend=True,
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
            sizeref=0.18,
            anchor="tip",
            showscale=False,
            showlegend=False,
        )
    )


def make_plot(X, i0, j0, Xu, Xv, W, show_vectors=True, show_coord=True):
    fig = go.Figure()
    p = X[i0, j0]

    fig.add_trace(
        go.Surface(
            x=X[..., 0],
            y=X[..., 1],
            z=X[..., 2],
            opacity=0.65,
            colorscale="Viridis",
            showscale=False,
            name="Superfície",
            showlegend=True,
        )
    )

    # Região de área local em torno de X(u0,v0)
    ni, nj = X.shape[0], X.shape[1]
    i1 = max(i0 - 3, 0)
    i2 = min(i0 + 4, ni)
    j1 = max(j0 - 3, 0)
    j2 = min(j0 + 4, nj)

    patch = X[i1:i2, j1:j2, :]

    fig.add_trace(
        go.Surface(
            x=patch[..., 0],
            y=patch[..., 1],
            z=patch[..., 2],
            opacity=0.95,
            colorscale=[
                [0, "rgba(255,160,0,0.95)"],
                [1, "rgba(255,160,0,0.95)"],
            ],
            showscale=False,
            name="Elemento de área local",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[p[0]],
            y=[p[1]],
            z=[p[2]],
            mode="markers",
            name="Ponto X(u₀,v₀)",
            marker=dict(size=7),
        )
    )

    if show_coord:
        fig.add_trace(
            go.Scatter3d(
                x=X[:, j0, 0],
                y=X[:, j0, 1],
                z=X[:, j0, 2],
                mode="lines",
                name="Curva u ↦ X(u,v₀)",
                line=dict(width=6),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=X[i0, :, 0],
                y=X[i0, :, 1],
                z=X[i0, :, 2],
                mode="lines",
                name="Curva v ↦ X(u₀,v)",
                line=dict(width=6),
            )
        )

    if show_vectors:
        add_arrow(fig, p, Xu[i0, j0], "Xᵤ(u₀,v₀)")
        add_arrow(fig, p, Xv[i0, j0], "Xᵥ(u₀,v₀)")
        add_arrow(fig, p, W, "w=aXᵤ+bXᵥ")

    fig.update_layout(
        height=680,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            font=dict(size=15),
            itemsizing="constant",
            x=0.02,
            y=0.02,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.75)",
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

# ============================================================
# TÍTULO E INTRODUÇÃO
# ============================================================

st.title("Módulo 4 — Primeira Forma Fundamental")

st.write(
    "A primeira forma fundamental mede comprimentos, ângulos e áreas sobre uma superfície."
)

st.latex(r"X:U\subset\mathbb{R}^2\longrightarrow \mathbb{R}^3")

st.write("Se")

st.latex(r"X(u,v)=(x(u,v),y(u,v),z(u,v)),")

st.write("então os coeficientes da primeira forma fundamental são:")

st.latex(r"E=\langle X_u,X_u\rangle")
st.latex(r"F=\langle X_u,X_v\rangle")
st.latex(r"G=\langle X_v,X_v\rangle")


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:
    st.header("Superfície")

    name = st.selectbox(
        "Escolha uma superfície",
        [
            "Plano",
            "Esfera",
            "Cilindro",
            "Catenoide",
            "Helicoide",
            "Toro",
            "Personalizada",
        ],
    )

    a = 1.0
    R = 1.0
    r = 0.35

    if name in ["Catenoide", "Helicoide"]:
        a = st.slider("Parâmetro a", 0.2, 3.0, 1.0, 0.1)

    if name in ["Esfera", "Cilindro", "Toro"]:
        R = st.slider("Raio principal R", 0.2, 3.0, 1.0, 0.1)

    if name == "Toro":
        r = st.slider("Raio menor r", 0.1, 1.2, 0.35, 0.05)

    custom = {}

    if name == "Personalizada":
        st.markdown("Digite as coordenadas usando as variáveis `u` e `v`.")
        st.caption("Funções disponíveis: sin, cos, tan, exp, log, sqrt, pi.")
        custom["x"] = st.text_input("x(u,v)", "u")
        custom["y"] = st.text_input("y(u,v)", "v")
        custom["z"] = st.text_input("z(u,v)", "sin(u)*cos(v)")

    st.header("Domínio")

    du0, du1, dv0, dv1 = default_domain(name)

    umin = st.number_input("u mínimo", value=float(du0))
    umax = st.number_input("u máximo", value=float(du1))
    vmin = st.number_input("v mínimo", value=float(dv0))
    vmax = st.number_input("v máximo", value=float(dv1))

    n = st.slider("Resolução", 40, 160, 80, 10)

    st.header("Ponto no Domínio")

    u0 = st.slider("Escolha u₀", float(umin), float(umax), float((umin + umax) / 2))
    v0 = st.slider("Escolha v₀", float(vmin), float(vmax), float((vmin + vmax) / 2))

    st.header("Vetor tangente")

    st.caption("Escolha um vetor no domínio: w = a∂u + b∂v.")

    coef_a = st.slider("Coeficiente a", -3.0, 3.0, 1.0, 0.1)
    coef_b = st.slider("Coeficiente b", -3.0, 3.0, 0.0, 0.1)

    st.header("Visualização")

    show_vectors = st.checkbox("Mostrar vetores tangentes", value=True)
    show_coord = st.checkbox("Mostrar curvas coordenadas", value=True)


# ============================================================
# CÁLCULOS
# ============================================================

if umax <= umin or vmax <= vmin:
    st.error("O domínio precisa satisfazer u mínimo < u máximo e v mínimo < v máximo.")
    st.stop()


try:
    U, V = make_grid(umin, umax, vmin, vmax, n)

    du = (umax - umin) / (n - 1)
    dv = (vmax - vmin) / (n - 1)

    X = surface(U, V, name, a=a, R=R, r=r, custom=custom)

    Xu, Xv = numerical_derivatives(X, du, dv)

    E = np.sum(Xu * Xu, axis=-1)
    F = np.sum(Xu * Xv, axis=-1)
    G = np.sum(Xv * Xv, axis=-1)

    area_density = np.sqrt(np.maximum(E * G - F**2, 0))
    total_area = np.sum(area_density) * du * dv

    i0 = closest_index(U, u0, axis=0)
    j0 = closest_index(V, v0, axis=1)

    p = X[i0, j0]
    xu = Xu[i0, j0]
    xv = Xv[i0, j0]

    E0 = E[i0, j0]
    F0 = F[i0, j0]
    G0 = G[i0, j0]

    W = coef_a * xu + coef_b * xv

    I_w = E0 * coef_a**2 + 2 * F0 * coef_a * coef_b + G0 * coef_b**2
    norm_w = np.sqrt(max(I_w, 0))

    cos_angle = F0 / np.sqrt(E0 * G0) if E0 > 1e-10 and G0 > 1e-10 else np.nan
    cos_angle = np.clip(cos_angle, -1, 1) if np.isfinite(cos_angle) else np.nan
    angle = np.arccos(cos_angle) if np.isfinite(cos_angle) else np.nan

    st.subheader("Superfície escolhida")

    st.markdown("A superfície visualizada é:")
    st.latex(
        rf"X:U\subset\mathbb{{R}}^2\longrightarrow \mathbb{{R}}^3,"
        rf"\qquad U=[{fmt(umin,2)},\,{fmt(umax,2)}]\times[{fmt(vmin,2)},\,{fmt(vmax,2)}]"
    )

    st.latex(surface_latex(name, custom))

    left, right = st.columns([2.2, 1])

    with left:
        st.plotly_chart(
            make_plot(X, i0, j0, Xu, Xv, W, show_vectors, show_coord),
            use_container_width=True,
        )

    with right:
        st.subheader("Valores no ponto escolhido")

        st.markdown("**Ponto no domínio:**")
        st.latex(rf"(u_0,v_0)=({fmt(U[i0,j0])},\,{fmt(V[i0,j0])})")

        st.markdown("**Ponto na superfície:**")
        st.latex(rf"X(u_0,v_0)=\left({fmt(p[0])},\,{fmt(p[1])},\,{fmt(p[2])}\right)")

        st.markdown("**Coeficientes da primeira forma:**")
        st.latex(rf"E(u_0,v_0)={fmt(E0)}")
        st.latex(rf"F(u_0,v_0)={fmt(F0)}")
        st.latex(rf"G(u_0,v_0)={fmt(G0)}")

        st.markdown("**Matriz métrica:**")
        st.latex(
            rf"g_{{ij}}="
            rf"\begin{{pmatrix}}"
            rf"{fmt(E0)} & {fmt(F0)}\\"
            rf"{fmt(F0)} & {fmt(G0)}"
            rf"\end{{pmatrix}}"
        )

        st.markdown("**Elemento de área local:**")
        st.latex(r"dA=\sqrt{EG-F^2}\,du\,dv")
        st.latex(rf"\sqrt{{EG-F^2}}={fmt(area_density[i0,j0])}")
        st.latex(rf"dA\approx {fmt(area_density[i0,j0] * du * dv)}")

        st.markdown("**Área aproximada no domínio:**")
        st.latex(rf"A\approx {fmt(total_area)}")

    st.subheader("Comprimento de vetores tangentes")

    st.write("Para um vetor tangente")

    st.latex(r"w=aX_u+bX_v,")

    st.write("a primeira forma fundamental calcula")

    st.latex(r"I(w)=\langle w,w\rangle=Ea^2+2Fab+Gb^2.")

    st.write("No ponto escolhido, com os valores selecionados na barra lateral:")

    st.latex(
        rf"w={fmt(coef_a,2)}X_u+{fmt(coef_b,2)}X_v"
    )

    st.latex(
        rf"I(w)={fmt(E0)}({fmt(coef_a,2)})^2"
        rf"+2({fmt(F0)})({fmt(coef_a,2)})({fmt(coef_b,2)})"
        rf"+{fmt(G0)}({fmt(coef_b,2)})^2"
        rf"={fmt(I_w)}"
    )

    st.latex(rf"\|w\|=\sqrt{{I(w)}}={fmt(norm_w)}")

    st.subheader("Ângulo entre as curvas coordenadas")

    st.write("O ângulo entre as curvas coordenadas é obtido por:")

    st.latex(
        r"\cos\theta="
        r"\frac{\langle X_u,X_v\rangle}{\|X_u\|\|X_v\|}"
        r"=\frac{F}{\sqrt{EG}}"
    )

    st.latex(rf"\cos\theta={fmt(cos_angle)}")

    if np.isfinite(angle):
        st.latex(rf"\theta\approx {fmt(angle)}\ \text{{rad}}")
        st.latex(rf"\theta\approx {fmt(np.degrees(angle))}^\circ")
    else:
        st.latex(r"\theta=\text{não definido}")

    st.subheader("Área de uma superfície parametrizada")

    st.write("A área aproximada é calculada numericamente por:")

    st.latex(
        r"A(X)\approx \sum_{i,j}\sqrt{E(u_i,v_j)G(u_i,v_j)-F(u_i,v_j)^2}\,\Delta u\,\Delta v"
    )

    st.latex(rf"A(X)\approx {fmt(total_area)}")

    st.subheader("Tabela numérica")

    df = pd.DataFrame(
        {
            "Quantidade": [
                "E",
                "F",
                "G",
                "EG-F²",
                "sqrt(EG-F²)",
                "I(w)",
                "||w||",
                "cos(theta)",
                "theta em graus",
            ],
            "Valor no ponto": [
                fmt(E0),
                fmt(F0),
                fmt(G0),
                fmt(E0 * G0 - F0**2),
                fmt(area_density[i0, j0]),
                fmt(I_w),
                fmt(norm_w),
                fmt(cos_angle),
                fmt(np.degrees(angle)) if np.isfinite(angle) else "não definido",
            ],
        }
    )

    st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Interpretação geométrica")

    st.write(
        "A primeira forma fundamental permite medir na superfície usando apenas os vetores tangentes "
        "da parametrização. Os coeficientes E, F e G determinam comprimentos, ângulos e áreas."
    )

    st.latex(r"E=\langle X_u,X_u\rangle")
    st.latex(r"F=\langle X_u,X_v\rangle")
    st.latex(r"G=\langle X_v,X_v\rangle")

    st.write(
        "Quando F é zero, as curvas coordenadas são ortogonais no ponto escolhido. "
        "Quando F é diferente de zero, o ângulo entre elas não é reto."
    )

except Exception as e:
    st.error("Não foi possível gerar a superfície. Verifique as expressões e o domínio escolhidos.")
    st.exception(e)
