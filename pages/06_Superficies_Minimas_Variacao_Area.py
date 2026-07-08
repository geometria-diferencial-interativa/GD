import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(
    page_title="Superfícies Mínimas e Variação da Área",
    page_icon="🫧",
    layout="wide",
)


with st.sidebar:
    st.title("📐 Geometria Diferencial")

    st.page_link("app.py", label="🏠 Início")
    st.page_link("pages/01_Curvas.py", label="1. Curvas em R³")
    st.page_link("pages/02_Superficies_Regulares.py", label="2. Superfícies Regulares")
    st.page_link("pages/03_Plano_Tangente_Normal.py", label="3. Plano Tangente e Normal")
    st.page_link("pages/04_Formas_Fundamentais.py", label="4. Formas Fundamentais")
    st.page_link("pages/05_Curvaturas.py", label="5. Curvaturas")
    st.page_link(
        "pages/06_Superficies_Minimas_Variacao_Area.py",
        label="6. Superfícies Mínimas e Variação da Área",
    )


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


def default_domain(name):
    if name in ["Catenoide", "Helicoide"]:
        return 0.0, float(2 * np.pi), -1.5, 1.5
    if name == "Enneper":
        return -1.6, 1.6, -1.6, 1.6
    if name == "Scherk":
        return -1.2, 1.2, -1.2, 1.2
    if name == "Plano":
        return -2.0, 2.0, -2.0, 2.0
    if name == "Esfera":
        return 0.0, float(2 * np.pi), 0.1, float(np.pi - 0.1)
    if name == "Cilindro":
        return 0.0, float(2 * np.pi), -2.0, 2.0
    return -2.0, 2.0, -2.0, 2.0


def surface(U, V, name, a=1.0, R=1.0, custom=None):
    if name == "Plano":
        return np.stack((U, V, 0 * U), axis=-1)

    if name == "Catenoide":
        X = a * np.cosh(V / a) * np.cos(U)
        Y = a * np.cosh(V / a) * np.sin(U)
        Z = V
        return np.stack((X, Y, Z), axis=-1)

    if name == "Helicoide":
        X = V * np.cos(U)
        Y = V * np.sin(U)
        Z = a * U
        return np.stack((X, Y, Z), axis=-1)

    if name == "Enneper":
        X = U - (U**3) / 3 + U * V**2
        Y = V - (V**3) / 3 + V * U**2
        Z = U**2 - V**2
        return np.stack((X, Y, Z), axis=-1)

    if name == "Scherk":
        X = U
        Y = V
        Z = np.log(np.cos(V) / np.cos(U))
        return np.stack((X, Y, Z), axis=-1)

    if name == "Esfera":
        X = R * np.cos(U) * np.sin(V)
        Y = R * np.sin(U) * np.sin(V)
        Z = R * np.cos(V)
        return np.stack((X, Y, Z), axis=-1)

    if name == "Cilindro":
        X = R * np.cos(U)
        Y = R * np.sin(U)
        Z = V
        return np.stack((X, Y, Z), axis=-1)

    if name == "Personalizada":
        X = eval_expr(custom["x"], U, V)
        Y = eval_expr(custom["y"], U, V)
        Z = eval_expr(custom["z"], U, V)
        return np.stack((X, Y, Z), axis=-1)

    return np.stack((U, V, 0 * U), axis=-1)


def surface_latex(name, custom=None):
    if name == "Plano":
        return r"X(u,v)=(u,v,0)"
    if name == "Catenoide":
        return r"X(u,v)=(a\cosh(v/a)\cos u,\;a\cosh(v/a)\sin u,\;v)"
    if name == "Helicoide":
        return r"X(u,v)=(v\cos u,\;v\sin u,\;au)"
    if name == "Enneper":
        return r"X(u,v)=\left(u-\frac{u^3}{3}+uv^2,\;v-\frac{v^3}{3}+vu^2,\;u^2-v^2\right)"
    if name == "Scherk":
        return r"X(u,v)=\left(u,\;v,\;\log\frac{\cos v}{\cos u}\right)"
    if name == "Esfera":
        return r"X(u,v)=(R\cos u\sin v,\;R\sin u\sin v,\;R\cos v)"
    if name == "Cilindro":
        return r"X(u,v)=(R\cos u,\;R\sin u,\;v)"
    if name == "Personalizada":
        return rf"X(u,v)=\left({custom['x']},\;{custom['y']},\;{custom['z']}\right)"
    return r"X(u,v)=(x(u,v),y(u,v),z(u,v))"


def derivatives_surface(X, du, dv):
    Xu = np.gradient(X, du, axis=0)
    Xv = np.gradient(X, dv, axis=1)
    Xuu = np.gradient(Xu, du, axis=0)
    Xuv = np.gradient(Xu, dv, axis=1)
    Xvv = np.gradient(Xv, dv, axis=1)
    return Xu, Xv, Xuu, Xuv, Xvv


def compute_geometry(X, du, dv):
    Xu, Xv, Xuu, Xuv, Xvv = derivatives_surface(X, du, dv)

    cross = np.cross(Xu, Xv)
    cross_norm = np.linalg.norm(cross, axis=-1, keepdims=True)

    N = np.divide(
        cross,
        cross_norm,
        out=np.zeros_like(cross),
        where=cross_norm > 1e-10,
    )

    E = np.sum(Xu * Xu, axis=-1)
    F = np.sum(Xu * Xv, axis=-1)
    G = np.sum(Xv * Xv, axis=-1)

    e = np.sum(Xuu * N, axis=-1)
    f = np.sum(Xuv * N, axis=-1)
    g = np.sum(Xvv * N, axis=-1)

    denom = E * G - F**2

    H = np.divide(
        e * G - 2 * f * F + g * E,
        2 * denom,
        out=np.full_like(denom, np.nan),
        where=np.abs(denom) > 1e-10,
    )

    K = np.divide(
        e * g - f**2,
        denom,
        out=np.full_like(denom, np.nan),
        where=np.abs(denom) > 1e-10,
    )

    area_density = cross_norm[..., 0]

    return {
        "Xu": Xu,
        "Xv": Xv,
        "N": N,
        "E": E,
        "F": F,
        "G": G,
        "e": e,
        "f": f,
        "g": g,
        "H": H,
        "K": K,
        "area_density": area_density,
    }


def total_area(area_density, du, dv):
    return float(np.nansum(area_density) * du * dv)


def closest_index(grid, value, axis):
    line = grid[:, 0] if axis == 0 else grid[0, :]
    return int(np.argmin(np.abs(line - value)))


def make_plot(X, C, title, colorbar_title, point=None, N=None, show_normal=False):
    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=X[..., 0],
            y=X[..., 1],
            z=X[..., 2],
            surfacecolor=C,
            colorscale="RdBu",
            colorbar=dict(title=colorbar_title),
            opacity=0.78,
            showscale=True,
            name=title,
        )
    )

    if point is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[point[0]],
                y=[point[1]],
                z=[point[2]],
                mode="markers",
                name="Ponto escolhido",
                marker=dict(size=7),
            )
        )

    if show_normal and point is not None and N is not None:
        n = np.linalg.norm(N)
        if n > 1e-10:
            v = N / n
            q = point + 0.7 * v
            fig.add_trace(
                go.Scatter3d(
                    x=[point[0], q[0]],
                    y=[point[1], q[1]],
                    z=[point[2], q[2]],
                    mode="lines",
                    name="Normal N",
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

    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=35, b=0),
        title=dict(text=title, x=0.5),
        legend=dict(
            font=dict(size=13),
            x=0.02,
            y=0.02,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.82)",
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


st.title("6 — Superfícies Mínimas e Variação da Área")

st.write(
    "Este módulo conecta a condição geométrica \(H=0\) com a ideia variacional de área. "
    "A superfície inicial é visualizada no instante \(t=0\), e a superfície variada é dada por "
    "\(X_t=X+t\,hN\)."
)

st.info(
    """
    Uma superfície mínima é uma superfície cuja curvatura média é nula em todos os pontos.
    Do ponto de vista variacional, isso significa que a superfície é ponto crítico do funcional área
    para variações normais.
    """
)


with st.sidebar:
    st.header("Superfície inicial")

    name = st.selectbox(
        "Escolha uma superfície",
        [
            "Plano",
            "Catenoide",
            "Helicoide",
            "Enneper",
            "Scherk",
            "Esfera",
            "Cilindro",
            "Personalizada",
        ],
    )

    a_surface = 1.0
    R = 1.0

    if name in ["Catenoide", "Helicoide"]:
        a_surface = st.slider("Parâmetro a", 0.2, 3.0, 1.0, 0.1)

    if name in ["Esfera", "Cilindro"]:
        R = st.slider("Raio R", 0.2, 3.0, 1.0, 0.1)

    custom = {}

    if name == "Personalizada":
        st.markdown("Digite as coordenadas usando `u` e `v`.")
        custom["x"] = st.text_input("x(u,v)", "u")
        custom["y"] = st.text_input("y(u,v)", "v")
        custom["z"] = st.text_input("z(u,v)", "sin(u)*cos(v)")

    st.header("Domínio")

    du0, du1, dv0, dv1 = default_domain(name)

    umin = st.number_input("u mínimo", value=float(du0))
    umax = st.number_input("u máximo", value=float(du1))
    vmin = st.number_input("v mínimo", value=float(dv0))
    vmax = st.number_input("v máximo", value=float(dv1))

    n = st.slider("Resolução", 45, 150, 85, 5)

    st.header("Ponto no domínio")

    u0 = st.slider("Escolha u₀", float(umin), float(umax), float((umin + umax) / 2))
    v0 = st.slider("Escolha v₀", float(vmin), float(vmax), float((vmin + vmax) / 2))

    st.header("Variação normal")

    st.markdown("A superfície variada é:")
    st.latex(r"X_t(u,v)=X(u,v)+t\,h(u,v)N(u,v)")

    h_expr = st.text_input("Função h(u,v)", "sin(u)*cos(v)")
    t_var = st.slider("Parâmetro t", -1.0, 1.0, 0.0, 0.05)

    st.header("Visualização")

    color_by = st.selectbox(
        "Colorir por",
        [
            "Curvatura média H",
            "Curvatura gaussiana K",
            "Função h(u,v)",
            "Coordenada z",
        ],
    )

    show_normal = st.checkbox("Mostrar normal no ponto", value=True)


if umax <= umin or vmax <= vmin:
    st.error("O domínio deve satisfazer u mínimo < u máximo e v mínimo < v máximo.")
    st.stop()


try:
    U, V = make_grid(umin, umax, vmin, vmax, n)

    du = (umax - umin) / (n - 1)
    dv = (vmax - vmin) / (n - 1)

    X = surface(U, V, name, a=a_surface, R=R, custom=custom)
    geom = compute_geometry(X, du, dv)

    h = eval_expr(h_expr, U, V)
    Xt = X + t_var * h[..., None] * geom["N"]
    geom_t = compute_geometry(Xt, du, dv)

    i0 = closest_index(U, u0, axis=0)
    j0 = closest_index(V, v0, axis=1)

    p0 = X[i0, j0]
    pt = Xt[i0, j0]

    area0 = total_area(geom["area_density"], du, dv)
    area_t = total_area(geom_t["area_density"], du, dv)

    if abs(t_var) > 1e-10:
        area_rate = (area_t - area0) / t_var
    else:
        area_rate = np.nan

    H_abs_mean = float(np.nanmean(np.abs(geom["H"])))

    if color_by == "Curvatura média H":
        C0 = geom["H"]
        Ct = geom_t["H"]
        colorbar = "H"
    elif color_by == "Curvatura gaussiana K":
        C0 = geom["K"]
        Ct = geom_t["K"]
        colorbar = "K"
    elif color_by == "Função h(u,v)":
        C0 = h
        Ct = h
        colorbar = "h"
    else:
        C0 = X[..., 2]
        Ct = Xt[..., 2]
        colorbar = "z"

    st.subheader("Superfície escolhida")

    st.latex(rf"X:U\subset\mathbb{{R}}^2\longrightarrow\mathbb{{R}}^3")
    st.latex(rf"U=[{fmt(umin,2)},{fmt(umax,2)}]\times[{fmt(vmin,2)},{fmt(vmax,2)}]")
    st.latex(surface_latex(name, custom))

    st.header("Definições pontuais de curvatura")

    st.write("As curvaturas são calculadas ponto a ponto na superfície.")

    st.latex(r"E=\langle X_u,X_u\rangle,\quad F=\langle X_u,X_v\rangle,\quad G=\langle X_v,X_v\rangle")
    st.latex(r"e=\langle X_{uu},N\rangle,\quad f=\langle X_{uv},N\rangle,\quad g=\langle X_{vv},N\rangle")
    st.latex(r"K=\frac{eg-f^2}{EG-F^2}")
    st.latex(r"H=\frac{eG-2fF+gE}{2(EG-F^2)}")

    st.write(
        "Uma superfície é chamada mínima quando sua curvatura média é nula em todos os pontos:"
    )

    st.latex(r"H(p)=0,\qquad \text{para todo }p\in S.")

    colA, colB, colC, colD = st.columns(4)

    colA.metric("Área de X", fmt(area0))
    colB.metric("Área de Xₜ", fmt(area_t), fmt(area_t - area0))
    colC.metric("Média de |H| em X", fmt(H_abs_mean))
    colD.metric("H(P) em X", fmt(geom["H"][i0, j0]))

    st.header("Visualização: superfície inicial e superfície variada")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            make_plot(
                X,
                C0,
                "Instante t=0: superfície inicial X",
                colorbar,
                point=p0,
                N=geom["N"][i0, j0],
                show_normal=show_normal,
            ),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            make_plot(
                Xt,
                Ct,
                f"Instante t={fmt(t_var,2)}: superfície variada Xₜ",
                colorbar,
                point=pt,
                N=geom_t["N"][i0, j0],
                show_normal=show_normal,
            ),
            use_container_width=True,
        )

    st.header("Variação normal da área")

    st.write("A variação normal escolhida é:")

    st.latex(r"X_t(u,v)=X(u,v)+t\,h(u,v)N(u,v)")
    st.latex(rf"h(u,v)={h_expr}")
    st.latex(rf"t={fmt(t_var)}")

    st.write("No ponto escolhido:")

    st.latex(rf"(u_0,v_0)=({fmt(U[i0,j0])},{fmt(V[i0,j0])})")
    st.latex(rf"h(u_0,v_0)={fmt(h[i0,j0])}")
    st.latex(rf"X(u_0,v_0)=\left({fmt(p0[0])},{fmt(p0[1])},{fmt(p0[2])}\right)")
    st.latex(rf"X_t(u_0,v_0)=\left({fmt(pt[0])},{fmt(pt[1])},{fmt(pt[2])}\right)")

    st.header("Relação entre superfícies mínimas e variação da área")

    st.write(
        "A primeira variação da área, para uma variação normal, é dada por uma fórmula do tipo:"
    )

    st.latex(r"\frac{d}{dt}A(X_t)\bigg|_{t=0}=-2\int_U H\,h\,dA")

    st.write(
        "Assim, se \(H=0\) em todos os pontos, então a derivada da área em \(t=0\) é nula "
        "para toda função de variação \(h\). Portanto, superfícies mínimas são pontos críticos "
        "do funcional área."
    )

    st.latex(r"H\equiv 0\quad\Longrightarrow\quad \frac{d}{dt}A(X_t)\bigg|_{t=0}=0")

    st.subheader("Valores numéricos da área")

    df_area = pd.DataFrame(
        {
            "Quantidade": [
                "Área de X",
                "Área de X_t",
                "Diferença A(X_t)-A(X)",
                "Quociente aproximado [A(X_t)-A(X)]/t",
                "H(P) na superfície inicial",
                "H_t(P) na superfície variada",
                "K(P) na superfície inicial",
                "K_t(P) na superfície variada",
            ],
            "Valor": [
                fmt(area0),
                fmt(area_t),
                fmt(area_t - area0),
                fmt(area_rate),
                fmt(geom["H"][i0, j0]),
                fmt(geom_t["H"][i0, j0]),
                fmt(geom["K"][i0, j0]),
                fmt(geom_t["K"][i0, j0]),
            ],
        }
    )

    st.dataframe(df_area, hide_index=True, use_container_width=True)

    st.success(
        """
        Experimente escolher catenoide ou helicoide e variar t perto de zero.
        Como essas superfícies são mínimas, a área deve variar de forma mais estável
        em primeira ordem. Depois compare com a esfera ou o cilindro, onde H não é nulo.
        """
    )

except Exception as e:
    st.error("Não foi possível gerar a superfície ou a variação. Verifique domínio, parâmetros e h(u,v).")
    st.exception(e)
