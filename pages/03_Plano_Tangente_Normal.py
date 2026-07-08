import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(
    page_title="Plano Tangente e Vetor Normal",
    page_icon="📐",
    layout="wide",
)


with st.sidebar:
    st.title("📐 Geometria Diferencial")
    st.page_link("app.py", label="🏠 Início")
    st.page_link("pages/01_Curvas.py", label="1. Curvas em R³")
    st.page_link("pages/02_Superficies_Regulares.py", label="2. Superfícies Regulares")
    st.page_link("pages/03_Plano_Tangente_Normal.py", label="3. Plano Tangente e Normal")
    st.page_link("pages/04_Formas_Fundamentais.py", label="4. Formas Fundamentais")
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


def eval_expr(expr, U, V):
    allowed = {
        "u": U, "v": V,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "abs": np.abs, "pi": np.pi,
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


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-10:
        return np.zeros_like(v)
    return v / n


def closest_index(grid, value, axis):
    line = grid[:, 0] if axis == 0 else grid[0, :]
    return int(np.argmin(np.abs(line - value)))


def add_arrow(fig, p, v, label, scale=0.6):
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


def tangent_plane(p, Xu, Xv, size=1.2, m=25):
    Xu = unit(Xu)
    Xv = unit(Xv)

    s = np.linspace(-size, size, m)
    t = np.linspace(-size, size, m)
    S, T = np.meshgrid(s, t)

    plane = (
        p[None, None, :]
        + S[..., None] * Xu[None, None, :]
        + T[..., None] * Xv[None, None, :]
    )

    return plane
    
def make_plot(
    X,
    i0,
    j0,
    Xu,
    Xv,
    N,
    plane_size,
    show_surface=True,
    show_coord=True,
    show_vectors=True,
    show_plane=True,
    show_normal_line=True,
):
    fig = go.Figure()
    p = X[i0, j0]

    if show_surface:
        fig.add_trace(
            go.Surface(
                x=X[..., 0],
                y=X[..., 1],
                z=X[..., 2],
                opacity=0.72,
                colorscale="Viridis",
                showscale=False,
                name="Superfície",
                showlegend=True,
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[p[0]],
            y=[p[1]],
            z=[p[2]],
            mode="markers",
            name="Ponto p=X(u₀,v₀)",
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

    if show_plane:
        P = tangent_plane(
            p,
            Xu[i0, j0],
            Xv[i0, j0],
            size=plane_size,
        )

        fig.add_trace(
            go.Surface(
                x=P[..., 0],
                y=P[..., 1],
                z=P[..., 2],
                opacity=0.75,
                colorscale=[
                    [0, "rgba(255,180,0,0.75)"],
                    [1, "rgba(255,180,0,0.75)"],
                ],
                showscale=False,
                name="Plano tangente",
                showlegend=True,
            )
        )

    if show_vectors:
        add_arrow(fig, p, Xu[i0, j0], "Xᵤ(u₀,v₀)")
        add_arrow(fig, p, Xv[i0, j0], "Xᵥ(u₀,v₀)")
        add_arrow(fig, p, N[i0, j0], "N(u₀,v₀)")

    if show_normal_line:
        nvec = N[i0, j0]
        q1 = p - plane_size * nvec
        q2 = p + plane_size * nvec

        fig.add_trace(
            go.Scatter3d(
                x=[q1[0], q2[0]],
                y=[q1[1], q2[1]],
                z=[q1[2], q2[2]],
                mode="lines",
                name="Reta normal",
                line=dict(width=5, dash="dash"),
            )
        )

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

st.title("3 - Plano Tangente e Vetor Normal")

st.write("Neste módulo estudamos a geometria local de uma superfície regular em torno de um ponto.")

st.latex(r"X:U\subset\mathbb{R}^2\longrightarrow \mathbb{R}^3")
st.latex(r"p=X(u_0,v_0)")

st.write("As curvas coordenadas são:")
st.latex(r"u\longmapsto X(u,v_0)")
st.latex(r"v\longmapsto X(u_0,v)")

st.write("Seus vetores tangentes são:")
st.latex(r"X_u(u_0,v_0)")
st.latex(r"X_v(u_0,v_0)")


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

    st.caption("Escolha os parâmetros u₀ e v₀ para determinar o ponto X(u₀,v₀).")

    u0 = st.slider("Escolha u₀", float(umin), float(umax), float((umin + umax) / 2))
    v0 = st.slider("Escolha v₀", float(vmin), float(vmax), float((vmin + vmax) / 2))

    st.header("Elementos geométricos")

    show_surface = st.checkbox("Mostrar superfície", value=True)
    show_coord = st.checkbox("Mostrar curvas coordenadas", value=True)
    show_vectors = st.checkbox("Mostrar Xᵤ, Xᵥ e N", value=True)
    show_plane = st.checkbox("Mostrar plano tangente", value=True)
    show_normal_line = st.checkbox("Mostrar reta normal", value=True)

    plane_size = st.slider("Tamanho do plano tangente", 0.3, 3.0, 1.4, 0.1)

if umax <= umin or vmax <= vmin:
    st.error("O domínio precisa satisfazer u mínimo < u máximo e v mínimo < v máximo.")
    st.stop()


try:
    U, V = make_grid(umin, umax, vmin, vmax, n)

    du = (umax - umin) / (n - 1)
    dv = (vmax - vmin) / (n - 1)

    X = surface(U, V, name, a=a, R=R, r=r, custom=custom)

    Xu, Xv = numerical_derivatives(X, du, dv)

    cross = np.cross(Xu, Xv)
    cross_norm = np.linalg.norm(cross, axis=-1, keepdims=True)

    N = np.divide(
        cross,
        cross_norm,
        out=np.zeros_like(cross),
        where=cross_norm > 1e-10,
    )

    i0 = closest_index(U, u0, axis=0)
    j0 = closest_index(V, v0, axis=1)

    p = X[i0, j0]
    xu = Xu[i0, j0]
    xv = Xv[i0, j0]
    cp = np.cross(xu, xv)
    cp_norm = np.linalg.norm(cp)
    nvec = unit(cp)

    st.subheader("Superfície escolhida")

    st.markdown("A superfície visualizada é a parametrização:")
    st.latex(
        rf"X:U\subset\mathbb{{R}}^2\longrightarrow \mathbb{{R}}^3,"
        rf"\qquad U=[{fmt(umin,2)},\,{fmt(umax,2)}]\times[{fmt(vmin,2)},\,{fmt(vmax,2)}]"
    )

    st.latex(surface_latex(name, custom))

    st.markdown("O ponto escolhido na superfície é:")
    st.latex(r"p=X(u_0,v_0)")

    left, right = st.columns([2.2, 1])

    with left:
        st.plotly_chart(
            make_plot(
                X,
                i0,
                j0,
                Xu,
                Xv,
                N,
                plane_size,
                show_surface,
                show_coord,
                show_vectors,
                show_plane,
                show_normal_line,
            ),
            use_container_width=True,
        )

    with right:
        st.subheader("Valores no ponto escolhido")

        st.markdown("**Ponto no domínio:**")
        st.latex(rf"(u_0,v_0)=({fmt(U[i0,j0])},\,{fmt(V[i0,j0])})")

        st.markdown("**Ponto na superfície:**")
        st.latex(rf"p=X(u_0,v_0)=\left({fmt(p[0])},\,{fmt(p[1])},\,{fmt(p[2])}\right)")

        st.markdown("**Vetores tangentes:**")
        st.latex(rf"X_u(u_0,v_0)=\left({fmt(xu[0])},\,{fmt(xu[1])},\,{fmt(xu[2])}\right)")
        st.latex(rf"X_v(u_0,v_0)=\left({fmt(xv[0])},\,{fmt(xv[1])},\,{fmt(xv[2])}\right)")

        st.markdown("**Produto vetorial:**")
        st.latex(rf"X_u\times X_v=\left({fmt(cp[0])},\,{fmt(cp[1])},\,{fmt(cp[2])}\right)")

        st.markdown("**Norma do produto vetorial:**")
        st.latex(rf"\left|X_u\times X_v\right|={fmt(cp_norm)}")

        st.markdown("**Normal unitária:**")
        st.latex(rf"N(u_0,v_0)=\left({fmt(nvec[0])},\,{fmt(nvec[1])},\,{fmt(nvec[2])}\right)")

        df = pd.DataFrame(
            {
                "Objeto": ["X_u", "X_v", "X_u × X_v", "N"],
                "x": [fmt(xu[0]), fmt(xv[0]), fmt(cp[0]), fmt(nvec[0])],
                "y": [fmt(xu[1]), fmt(xv[1]), fmt(cp[1]), fmt(nvec[1])],
                "z": [fmt(xu[2]), fmt(xv[2]), fmt(cp[2]), fmt(nvec[2])],
            }
        )

        st.subheader("Tabela numérica")
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Plano tangente")

    st.write("O plano tangente é gerado pelos vetores tangentes:")
    st.latex(r"T_pS=\operatorname{span}\{X_u(u_0,v_0),X_v(u_0,v_0)\}")

    st.write("Sua equação paramétrica é:")
    st.latex(r"P(s,t)=p+sX_u(u_0,v_0)+tX_v(u_0,v_0)")

    st.write("Com os valores numéricos do ponto escolhido:")
    st.latex(
        rf"P(s,t)=\left({fmt(p[0])},\,{fmt(p[1])},\,{fmt(p[2])}\right)"
        rf"+s\left({fmt(xu[0])},\,{fmt(xu[1])},\,{fmt(xu[2])}\right)"
        rf"+t\left({fmt(xv[0])},\,{fmt(xv[1])},\,{fmt(xv[2])}\right)"
    )

    st.subheader("Vetor normal")

    st.write("Como a superfície é regular, temos:")
    st.latex(r"X_u(u_0,v_0)\times X_v(u_0,v_0)\neq 0")

    st.write("O vetor normal unitário é:")
    st.latex(
        r"N(u_0,v_0)=\frac{X_u(u_0,v_0)\times X_v(u_0,v_0)}"
        r"{\left\|X_u(u_0,v_0)\times X_v(u_0,v_0)\right\|}"
    )

    st.subheader("Interpretação geométrica")

    st.write("As curvas coordenadas passam pelo ponto escolhido e têm vetores tangentes:")
    st.latex(r"X_u(u_0,v_0)\quad\text{e}\quad X_v(u_0,v_0)")

    st.write(
        "O plano tangente é o plano que melhor aproxima a superfície perto do ponto escolhido. "
        "O vetor normal é perpendicular ao plano tangente e é obtido pelo produto vetorial dos vetores tangentes."
    )

except Exception as e:
    st.error("Não foi possível gerar a superfície. Verifique as expressões e o domínio escolhidos.")
    st.exception(e)
