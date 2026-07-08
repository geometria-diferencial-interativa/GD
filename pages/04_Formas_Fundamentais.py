import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(
    page_title="Formas Fundamentais",
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
    st.page_link("pages/05_Curvaturas.py", label="5. Curvaturas")
    st.page_link("pages/06_Superficies_Minimas_Variacao_Area.py",label="6. Superfícies Mínimas e Variação da Área",)



def fmt(x, digits=5):
    try:
        x = float(x)
        if not np.isfinite(x):
            return "não definido"
        return f"{x:.{digits}f}"
    except Exception:
        return "não definido"


def safe_unit(v):
    n = np.linalg.norm(v)
    if n < 1e-10:
        return np.zeros_like(v)
    return v / n


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

    if name == "Parabolóide hiperbólico":
        return np.stack((U, V, V**2 - U**2), axis=-1)

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
    if name == "Parabolóide hiperbólico":
        return r"X(u,v)=(u,v,v^2-u^2)"
    if name == "Personalizada":
        return rf"X(u,v)=\left({custom['x']},\;{custom['y']},\;{custom['z']}\right)"
    return r"X(u,v)=(x(u,v),y(u,v),z(u,v))"


def default_domain(name):
    if name == "Esfera":
        return 0.0, float(2 * np.pi), 0.05, float(np.pi - 0.05)
    if name == "Cilindro":
        return 0.0, float(2 * np.pi), -2.0, 2.0
    if name == "Catenoide":
        return 0.0, float(2 * np.pi), -1.5, 1.5
    if name == "Helicoide":
        return -2.0, 2.0, 0.0, float(4 * np.pi)
    if name == "Toro":
        return 0.0, float(2 * np.pi), 0.0, float(2 * np.pi)
    if name == "Parabolóide hiperbólico":
        return -1.5, 1.5, -1.5, 1.5
    return -2.0, 2.0, -2.0, 2.0


def derivatives_surface(X, du, dv):
    Xu = np.gradient(X, du, axis=0)
    Xv = np.gradient(X, dv, axis=1)

    Xuu = np.gradient(Xu, du, axis=0)
    Xuv = np.gradient(Xu, dv, axis=1)
    Xvv = np.gradient(Xv, dv, axis=1)

    return Xu, Xv, Xuu, Xuv, Xvv


def unit_normal(Xu, Xv):
    cross = np.cross(Xu, Xv)
    length = np.linalg.norm(cross, axis=-1, keepdims=True)

    N = np.divide(
        cross,
        length,
        out=np.zeros_like(cross),
        where=length > 1e-10,
    )

    return N, length[..., 0]


def closest_index(grid, value, axis):
    line = grid[:, 0] if axis == 0 else grid[0, :]
    return int(np.argmin(np.abs(line - value)))


def add_arrow(fig, p, vector, label, scale=0.75):
    n = np.linalg.norm(vector)

    if n < 1e-10:
        return

    direction = vector / n
    q = p + scale * direction

    fig.add_trace(
        go.Scatter3d(
            x=[p[0], q[0]],
            y=[p[1], q[1]],
            z=[p[2], q[2]],
            mode="lines",
            name=label,
            line=dict(width=7),
        )
    )

    fig.add_trace(
        go.Cone(
            x=[q[0]],
            y=[q[1]],
            z=[q[2]],
            u=[direction[0]],
            v=[direction[1]],
            w=[direction[2]],
            sizemode="absolute",
            sizeref=0.18,
            anchor="tip",
            showscale=False,
            showlegend=False,
        )
    )


def add_area_patch(fig, X, i0, j0, patch_size=3):
    ni, nj = X.shape[0], X.shape[1]

    i1 = max(i0 - patch_size, 0)
    i2 = min(i0 + patch_size + 1, ni)

    j1 = max(j0 - patch_size, 0)
    j2 = min(j0 + patch_size + 1, nj)

    patch = X[i1:i2, j1:j2, :]

    fig.add_trace(
        go.Surface(
            x=patch[..., 0],
            y=patch[..., 1],
            z=patch[..., 2],
            opacity=1,
            colorscale=[[0, "orange"], [1, "orange"]],
            showscale=False,
            name="Elemento de área",
            showlegend=True,
        )
    )


def add_tangent_plane(fig, p, Xu0, Xv0, scale=0.9):
    e1 = safe_unit(Xu0)
    e2 = safe_unit(Xv0 - np.dot(Xv0, e1) * e1)

    s = np.linspace(-scale, scale, 12)
    t = np.linspace(-scale, scale, 12)
    S, T = np.meshgrid(s, t)

    P = p[None, None, :] + S[..., None] * e1 + T[..., None] * e2

    fig.add_trace(
        go.Surface(
            x=P[..., 0],
            y=P[..., 1],
            z=P[..., 2],
            opacity=0.35,
            colorscale=[[0, "lightblue"], [1, "lightblue"]],
            showscale=False,
            name="Plano tangente",
            showlegend=True,
        )
    )


def add_normal_plane(fig, p, W, N0, scale=1.1):
    w = safe_unit(W)
    n = safe_unit(N0)

    s = np.linspace(-scale, scale, 12)
    t = np.linspace(-scale, scale, 12)
    S, T = np.meshgrid(s, t)

    P = p[None, None, :] + S[..., None] * w + T[..., None] * n

    fig.add_trace(
        go.Surface(
            x=P[..., 0],
            y=P[..., 1],
            z=P[..., 2],
            opacity=0.40,
            colorscale=[[0, "rgba(255,120,120,0.55)"], [1, "rgba(255,120,120,0.55)"]],
            showscale=False,
            name="Plano normal gerado por w e N",
            showlegend=True,
        )
    )


def add_normal_section(fig, X, p, W, N0):
    w = safe_unit(W)
    n = safe_unit(N0)

    plane_normal = safe_unit(np.cross(w, n))

    if np.linalg.norm(plane_normal) < 1e-10:
        return

    values = np.abs(np.tensordot(X - p, plane_normal, axes=([2], [0])))

    diagonal = np.linalg.norm(
        np.nanmax(X.reshape(-1, 3), axis=0) - np.nanmin(X.reshape(-1, 3), axis=0)
    )

    threshold = 0.015 * diagonal

    mask = values < threshold

    pts = X[mask]

    if pts.shape[0] < 3:
        return

    dist = np.linalg.norm(pts - p, axis=1)
    order = np.argsort(dist)
    pts = pts[order[:300]]

    coord = np.dot(pts - p, w)
    order2 = np.argsort(coord)
    pts = pts[order2]

    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            name="Aproximação da seção normal",
            marker=dict(size=4),
        )
    )


def make_plot(
    X,
    i0,
    j0,
    Xu,
    Xv,
    N,
    W,
    show_coord,
    show_vectors,
    show_normal,
    show_plane,
    show_area,
    show_normal_plane,
    show_normal_section,
):
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

    if show_area:
        add_area_patch(fig, X, i0, j0)

    if show_plane:
        add_tangent_plane(fig, p, Xu[i0, j0], Xv[i0, j0])

    if show_normal_plane:
        add_normal_plane(fig, p, W, N[i0, j0])

    if show_normal_section:
        add_normal_section(fig, X, p, W, N[i0, j0])

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
        add_arrow(fig, p, Xu[i0, j0], "Xᵤ")
        add_arrow(fig, p, Xv[i0, j0], "Xᵥ")
        add_arrow(fig, p, W, "w(θ)")

    if show_normal:
        add_arrow(fig, p, N[i0, j0], "N")

    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            font=dict(size=14),
            itemsizing="constant",
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


st.title("4 — Formas Fundamentais")

st.write(
    "Nesta página estudamos, no mesmo ponto da superfície, a Primeira Forma Fundamental, "
    "a Segunda Forma Fundamental e a curvatura normal em uma direção tangente variável."
)

st.info(
    """
    A Primeira Forma Fundamental mede comprimentos, ângulos e áreas.
    A Segunda Forma Fundamental mede como a superfície se curva no espaço.
    A curvatura normal observa a curvatura em uma direção tangente específica.
    """
)


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
            "Parabolóide hiperbólico",
            "Personalizada",
        ],
    )

    a_surface = 1.0
    R = 1.0
    r = 0.35

    if name in ["Catenoide", "Helicoide"]:
        a_surface = st.slider("Parâmetro da superfície a", 0.2, 3.0, 1.0, 0.1)

    if name in ["Esfera", "Cilindro", "Toro"]:
        R = st.slider("Raio principal R", 0.2, 3.0, 1.0, 0.1)

    if name == "Toro":
        r = st.slider("Raio menor r", 0.1, 1.2, 0.35, 0.05)

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

    n = st.slider("Resolução", 45, 170, 90, 5)

    st.header("Ponto no Domínio")

    u0 = st.slider("Escolha u₀", float(umin), float(umax), float((umin + umax) / 2))
    v0 = st.slider("Escolha v₀", float(vmin), float(vmax), float((vmin + vmax) / 2))

    st.header("Direção tangente")

    theta = st.slider(
        "Ângulo θ da direção w",
        0.0,
        float(2 * np.pi),
        float(np.pi / 4),
        0.01,
    )

    st.caption("O vetor w(θ) gira no plano tangente.")

    st.header("Visualização")

    show_coord = st.checkbox("Mostrar curvas coordenadas", value=True)
    show_vectors = st.checkbox("Mostrar vetores Xᵤ, Xᵥ e w", value=True)
    show_normal = st.checkbox("Mostrar vetor normal N", value=True)
    show_plane = st.checkbox("Mostrar plano tangente", value=True)
    show_area = st.checkbox("Mostrar elemento de área", value=True)
    show_normal_plane = st.checkbox("Mostrar plano normal gerado por w e N", value=True)
    show_normal_section = st.checkbox("Mostrar aproximação da seção normal", value=True)


if umax <= umin or vmax <= vmin:
    st.error("O domínio deve satisfazer u mínimo < u máximo e v mínimo < v máximo.")
    st.stop()


try:
    U, V = make_grid(umin, umax, vmin, vmax, n)

    du = (umax - umin) / (n - 1)
    dv = (vmax - vmin) / (n - 1)

    X = surface(U, V, name, a=a_surface, R=R, r=r, custom=custom)

    Xu, Xv, Xuu, Xuv, Xvv = derivatives_surface(X, du, dv)
    N, area_density = unit_normal(Xu, Xv)

    i0 = closest_index(U, u0, axis=0)
    j0 = closest_index(V, v0, axis=1)

    p = X[i0, j0]

    Xu0 = Xu[i0, j0]
    Xv0 = Xv[i0, j0]
    Xuu0 = Xuu[i0, j0]
    Xuv0 = Xuv[i0, j0]
    Xvv0 = Xvv[i0, j0]
    N0 = N[i0, j0]

    E = np.dot(Xu0, Xu0)
    F = np.dot(Xu0, Xv0)
    G = np.dot(Xv0, Xv0)

    e = np.dot(Xuu0, N0)
    f = np.dot(Xuv0, N0)
    g = np.dot(Xvv0, N0)

    # Base ortonormal do plano tangente
    T1 = safe_unit(Xu0)
    T2 = safe_unit(Xv0 - np.dot(Xv0, T1) * T1)

    W = np.cos(theta) * T1 + np.sin(theta) * T2
    W = safe_unit(W)

    # Escreve W = a Xu + b Xv, aproximadamente
    A = np.column_stack((Xu0, Xv0))
    coeffs, *_ = np.linalg.lstsq(A, W, rcond=None)
    coef_a = coeffs[0]
    coef_b = coeffs[1]

    Iww = E * coef_a**2 + 2 * F * coef_a * coef_b + G * coef_b**2
    IIww = e * coef_a**2 + 2 * f * coef_a * coef_b + g * coef_b**2

    kn = IIww / Iww if abs(Iww) > 1e-10 else np.nan

    st.subheader("Superfície escolhida")

    st.latex(rf"X:U\subset\mathbb{{R}}^2\longrightarrow\mathbb{{R}}^3")
    st.latex(rf"U=[{fmt(umin,2)},{fmt(umax,2)}]\times[{fmt(vmin,2)},{fmt(vmax,2)}]")
    st.latex(surface_latex(name, custom))

    st.subheader("Visualização geométrica")

    st.write(
        "O gráfico mostra a superfície, o ponto escolhido, os vetores tangentes, "
        "o vetor normal, o elemento de área, o plano tangente, o plano normal e "
        "uma aproximação da seção normal."
    )

    st.plotly_chart(
        make_plot(
            X,
            i0,
            j0,
            Xu,
            Xv,
            N,
            W,
            show_coord,
            show_vectors,
            show_normal,
            show_plane,
            show_area,
            show_normal_plane,
            show_normal_section,
        ),
        use_container_width=True,
    )

    st.header("1. Ponto escolhido")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Ponto no domínio:**")
        st.latex(rf"(u_0,v_0)=({fmt(U[i0,j0])},{fmt(V[i0,j0])})")

    with col2:
        st.markdown("**Ponto na superfície:**")
        st.latex(rf"p=X(u_0,v_0)=\left({fmt(p[0])},{fmt(p[1])},{fmt(p[2])}\right)")

    st.header("2. Primeira Forma Fundamental")

    st.write(
        "A Primeira Forma Fundamental mede grandezas métricas na superfície: "
        "comprimentos, ângulos e áreas."
    )

    st.latex(r"E=\langle X_u,X_u\rangle")
    st.latex(r"F=\langle X_u,X_v\rangle")
    st.latex(r"G=\langle X_v,X_v\rangle")

    c1, c2, c3 = st.columns(3)
    c1.metric("E", fmt(E))
    c2.metric("F", fmt(F))
    c3.metric("G", fmt(G))

    st.latex(
        rf"I="
        rf"\begin{{pmatrix}}"
        rf"{fmt(E)} & {fmt(F)}\\"
        rf"{fmt(F)} & {fmt(G)}"
        rf"\end{{pmatrix}}"
    )

    st.latex(r"I=E\,du^2+2F\,du\,dv+G\,dv^2")

    st.subheader("Elemento de área")

    st.write("A área infinitesimal é dada por:")

    st.latex(r"dA=\|X_u\times X_v\|\,du\,dv")
    st.latex(r"dA=\sqrt{EG-F^2}\,du\,dv")
    st.latex(rf"\sqrt{{EG-F^2}}={fmt(area_density[i0,j0])}")

    st.header("3. Segunda Forma Fundamental")

    st.write(
        "A Segunda Forma Fundamental mede como a superfície se curva em relação "
        "ao vetor normal."
    )

    st.latex(r"N=\frac{X_u\times X_v}{\|X_u\times X_v\|}")
    st.latex(rf"N(u_0,v_0)=\left({fmt(N0[0])},{fmt(N0[1])},{fmt(N0[2])}\right)")

    st.latex(r"e=\langle X_{uu},N\rangle")
    st.latex(r"f=\langle X_{uv},N\rangle")
    st.latex(r"g=\langle X_{vv},N\rangle")

    c1, c2, c3 = st.columns(3)
    c1.metric("e", fmt(e))
    c2.metric("f", fmt(f))
    c3.metric("g", fmt(g))

    st.latex(
        rf"II="
        rf"\begin{{pmatrix}}"
        rf"{fmt(e)} & {fmt(f)}\\"
        rf"{fmt(f)} & {fmt(g)}"
        rf"\end{{pmatrix}}"
    )

    st.latex(r"II=e\,du^2+2f\,du\,dv+g\,dv^2")

    st.header("4. Direção tangente móvel")

    st.write(
        "Em vez de escolher separadamente os coeficientes de \(X_u\) e \(X_v\), "
        "agora escolhemos um ângulo \\(\\theta\\). Esse ângulo faz a direção "
        "\\(w(\\theta)\\) girar no plano tangente."
    )

    st.latex(r"w(\theta)=\cos(\theta)T_1+\sin(\theta)T_2")

    st.write("Aqui, \(T_1\) e \(T_2\) formam uma base ortonormal do plano tangente.")

    st.latex(rf"\theta={fmt(theta)}\ \text{{rad}}")
    st.latex(rf"\theta\approx {fmt(np.degrees(theta))}^\circ")

    st.latex(rf"w(\theta)=\left({fmt(W[0])},{fmt(W[1])},{fmt(W[2])}\right)")

    st.write("Na base \(\{X_u,X_v\}\), numericamente:")

    st.latex(rf"w\approx {fmt(coef_a)}X_u+{fmt(coef_b)}X_v")

    st.header("5. Curvatura normal na direção escolhida")

    st.write(
        "A curvatura normal compara a Segunda Forma Fundamental com a Primeira "
        "Forma Fundamental na direção tangente escolhida."
    )

    st.latex(r"I(w,w)=Ea^2+2Fab+Gb^2")
    st.latex(r"II(w,w)=ea^2+2fab+gb^2")

    st.latex(rf"I(w,w)={fmt(Iww)}")
    st.latex(rf"II(w,w)={fmt(IIww)}")

    st.latex(r"k_n(w)=\frac{II(w,w)}{I(w,w)}")
    st.latex(rf"k_n(w)={fmt(kn)}")

    st.header("6. Plano normal e seção normal")

    st.write(
        "O plano normal associado à direção escolhida é o plano gerado por "
        "\(w\) e \(N\)."
    )

    st.latex(r"\Pi_{\text{normal}}=\operatorname{span}\{w,N\}")

    st.write(
        "A interseção desse plano com a superfície é chamada seção normal. "
        "No gráfico, ela é mostrada de forma aproximada pelos pontos da malha "
        "que ficam próximos desse plano."
    )

    st.success(
        """
        Experimente mover o ponto (u₀,v₀) e variar o ângulo θ.

        Observe que o vetor w gira no plano tangente e que a curvatura normal
        muda conforme a direção escolhida.
        """
    )

    st.header("Resumo geométrico")

    df = pd.DataFrame(
        {
            "Objeto": [
                "Primeira Forma Fundamental",
                "Segunda Forma Fundamental",
                "Direção tangente",
                "Curvatura normal",
                "Seção normal",
            ],
            "Significado": [
                "mede comprimentos, ângulos e áreas",
                "mede como a superfície se curva no espaço",
                "direção escolhida no plano tangente",
                "curvatura da superfície na direção escolhida",
                "interseção da superfície com o plano gerado por w e N",
            ],
        }
    )

    st.dataframe(df, hide_index=True, use_container_width=True)

except Exception as e:
    st.error("Não foi possível gerar a superfície. Verifique as expressões e o domínio escolhidos.")
    st.exception(e)
