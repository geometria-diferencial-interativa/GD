import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(
    page_title="Curvaturas",
    page_icon="🌀",
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
    st.page_link("pages/03_Plano_Tangente_Normal.py", label="3. Plano Tangente e Normal")
    st.page_link("pages/04_Formas_Fundamentais.py", label="4. Formas Fundamentais")
    st.page_link("pages/05_Curvaturas.py", label="5. Curvaturas")

    st.markdown("6. Superfícies Mínimas")
    st.markdown("7. Variação da Área")


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


def closest_index(grid, value, axis):
    line = grid[:, 0] if axis == 0 else grid[0, :]
    return int(np.argmin(np.abs(line - value)))


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

    K = np.divide(
        e * g - f**2,
        denom,
        out=np.full_like(denom, np.nan),
        where=np.abs(denom) > 1e-10,
    )

    H = np.divide(
        e * G - 2 * f * F + g * E,
        2 * denom,
        out=np.full_like(denom, np.nan),
        where=np.abs(denom) > 1e-10,
    )

    discr = H**2 - K
    discr = np.where(discr < 0, np.nan, discr)

    k1 = H + np.sqrt(discr)
    k2 = H - np.sqrt(discr)

    return {
        "Xu": Xu,
        "Xv": Xv,
        "Xuu": Xuu,
        "Xuv": Xuv,
        "Xvv": Xvv,
        "N": N,
        "E": E,
        "F": F,
        "G": G,
        "e": e,
        "f": f,
        "g": g,
        "K": K,
        "H": H,
        "k1": k1,
        "k2": k2,
        "area_density": cross_norm[..., 0],
    }


def classify_point(K, H, k1, k2, tol=1e-4):
    if not np.isfinite(K) or not np.isfinite(H):
        return "não definido"

    if abs(k1) < tol and abs(k2) < tol:
        return "planar"

    if K > tol:
        if abs(k1 - k2) < tol:
            return "umbílico elíptico"
        return "elíptico"

    if K < -tol:
        return "hiperbólico"

    if abs(K) <= tol:
        return "parabólico"

    return "não definido"


def classification_array(K, k1, k2, tol=1e-4):
    C = np.zeros_like(K)

    C[np.isnan(K)] = np.nan
    C[K > tol] = 1
    C[K < -tol] = -1
    C[np.abs(K) <= tol] = 0

    planar = (np.abs(k1) < tol) & (np.abs(k2) < tol)
    C[planar] = 2

    return C


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


def principal_directions(Xu0, Xv0, E0, F0, G0, e0, f0, g0):
    I = np.array([[E0, F0], [F0, G0]], dtype=float)
    II = np.array([[e0, f0], [f0, g0]], dtype=float)

    if abs(np.linalg.det(I)) < 1e-10:
        return None, None

    S = np.linalg.solve(I, II)
    vals, vecs = np.linalg.eig(S)

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    directions = []

    for j in range(2):
        a = float(np.real(vecs[0, j]))
        b = float(np.real(vecs[1, j]))
        w = a * Xu0 + b * Xv0
        directions.append(safe_unit(w))

    return vals, directions


def make_plot(
    X,
    geom,
    i0,
    j0,
    color_by,
    show_coord=True,
    show_normal=True,
    show_principal=True,
):
    fig = go.Figure()
    p = X[i0, j0]

    if color_by == "Curvatura gaussiana K":
        C = geom["K"]
        title = "K"
        colorscale = "RdBu"

    elif color_by == "Curvatura média H":
        C = geom["H"]
        title = "H"
        colorscale = "RdBu"

    elif color_by == "Curvatura principal k₁":
        C = geom["k1"]
        title = "k₁"
        colorscale = "RdBu"

    elif color_by == "Curvatura principal k₂":
        C = geom["k2"]
        title = "k₂"
        colorscale = "RdBu"

    else:
        C = classification_array(geom["K"], geom["k1"], geom["k2"])
        title = "classificação"
        colorscale = [
            [0.00, "royalblue"],
            [0.25, "royalblue"],
            [0.50, "lightgray"],
            [0.75, "orange"],
            [1.00, "purple"],
        ]

    fig.add_trace(
        go.Surface(
            x=X[..., 0],
            y=X[..., 1],
            z=X[..., 2],
            surfacecolor=C,
            colorscale=colorscale,
            colorbar=dict(title=title),
            opacity=0.72,
            showscale=True,
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

    if show_normal:
        add_arrow(fig, p, geom["N"][i0, j0], "N")

    if show_principal:
        E0 = geom["E"][i0, j0]
        F0 = geom["F"][i0, j0]
        G0 = geom["G"][i0, j0]
        e0 = geom["e"][i0, j0]
        f0 = geom["f"][i0, j0]
        g0 = geom["g"][i0, j0]

        vals, directions = principal_directions(
            geom["Xu"][i0, j0],
            geom["Xv"][i0, j0],
            E0,
            F0,
            G0,
            e0,
            f0,
            g0,
        )

        if directions is not None:
            add_arrow(fig, p, directions[0], "Direção principal k₁", scale=0.9)
            add_arrow(fig, p, directions[1], "Direção principal k₂", scale=0.9)

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


# ============================================================
# TÍTULO
# ============================================================

st.title("5 — Curvaturas")

st.write(
    "Neste módulo estudamos as principais medidas de curvatura de uma superfície: "
    "curvaturas principais, curvatura gaussiana, curvatura média e classificação dos pontos."
)

st.info(
    """
    A ideia central é observar como a superfície se curva em cada ponto.
    As curvaturas principais são os valores extremos da curvatura normal.
    A curvatura gaussiana é o produto desses valores.
    A curvatura média é a média aritmética deles.
    """
)


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

    st.header("Visualização")

    color_by = st.selectbox(
        "Colorir a superfície por",
        [
            "Curvatura gaussiana K",
            "Curvatura média H",
            "Curvatura principal k₁",
            "Curvatura principal k₂",
            "Classificação dos pontos",
        ],
    )

    show_coord = st.checkbox("Mostrar curvas coordenadas", value=True)
    show_normal = st.checkbox("Mostrar vetor normal", value=True)
    show_principal = st.checkbox("Mostrar direções principais", value=True)


# ============================================================
# CÁLCULOS
# ============================================================

if umax <= umin or vmax <= vmin:
    st.error("O domínio deve satisfazer u mínimo < u máximo e v mínimo < v máximo.")
    st.stop()


try:
    U, V = make_grid(umin, umax, vmin, vmax, n)

    du = (umax - umin) / (n - 1)
    dv = (vmax - vmin) / (n - 1)

    X = surface(U, V, name, a=a_surface, R=R, r=r, custom=custom)

    geom = compute_geometry(X, du, dv)

    i0 = closest_index(U, u0, axis=0)
    j0 = closest_index(V, v0, axis=1)

    p = X[i0, j0]

    E0 = geom["E"][i0, j0]
    F0 = geom["F"][i0, j0]
    G0 = geom["G"][i0, j0]

    e0 = geom["e"][i0, j0]
    f0 = geom["f"][i0, j0]
    g0 = geom["g"][i0, j0]

    K0 = geom["K"][i0, j0]
    H0 = geom["H"][i0, j0]
    k10 = geom["k1"][i0, j0]
    k20 = geom["k2"][i0, j0]

    classification = classify_point(K0, H0, k10, k20)

    st.subheader("Superfície escolhida")

    st.latex(rf"X:U\subset\mathbb{{R}}^2\longrightarrow\mathbb{{R}}^3")
    st.latex(rf"U=[{fmt(umin,2)},{fmt(umax,2)}]\times[{fmt(vmin,2)},{fmt(vmax,2)}]")
    st.latex(surface_latex(name, custom))

    st.subheader("Visualização geométrica")

    st.write(
        "A coloração representa a quantidade escolhida na barra lateral. "
        "Assim, \(K\), \(H\), \(k_1\) e \(k_2\) são visualizados como funções sobre a superfície."
    )

    st.plotly_chart(
        make_plot(
            X,
            geom,
            i0,
            j0,
            color_by,
            show_coord,
            show_normal,
            show_principal,
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
        st.latex(rf"X(u_0,v_0)=\left({fmt(p[0])},{fmt(p[1])},{fmt(p[2])}\right)")

    st.header("2. Fórmulas fundamentais")

    st.write("A partir das formas fundamentais, calculamos as curvaturas.")

    st.latex(r"I=\begin{pmatrix}E&F\\F&G\end{pmatrix}")
    st.latex(r"II=\begin{pmatrix}e&f\\f&g\end{pmatrix}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Primeira forma")
        st.latex(rf"E={fmt(E0)}")
        st.latex(rf"F={fmt(F0)}")
        st.latex(rf"G={fmt(G0)}")

    with col2:
        st.subheader("Segunda forma")
        st.latex(rf"e={fmt(e0)}")
        st.latex(rf"f={fmt(f0)}")
        st.latex(rf"g={fmt(g0)}")

    st.header("3. Curvatura gaussiana")

    st.write(
        "A curvatura gaussiana é o produto das curvaturas principais. "
        "Também pode ser calculada diretamente pelos coeficientes das formas fundamentais."
    )

    st.latex(r"K=k_1k_2")
    st.latex(r"K=\frac{eg-f^2}{EG-F^2}")
    st.latex(rf"K(u_0,v_0)={fmt(K0)}")

    st.header("4. Curvatura média")

    st.write(
        "A curvatura média é a média aritmética das curvaturas principais."
    )

    st.latex(r"H=\frac{k_1+k_2}{2}")
    st.latex(r"H=\frac{eG-2fF+gE}{2(EG-F^2)}")
    st.latex(rf"H(u_0,v_0)={fmt(H0)}")

    st.header("5. Curvaturas principais")

    st.write(
        "As curvaturas principais são os valores extremos da curvatura normal no ponto."
    )

    st.latex(r"k_1=H+\sqrt{H^2-K}")
    st.latex(r"k_2=H-\sqrt{H^2-K}")

    col1, col2 = st.columns(2)

    col1.metric("k₁", fmt(k10))
    col2.metric("k₂", fmt(k20))

    st.header("6. Classificação do ponto")

    st.write(
        "A classificação depende principalmente do sinal da curvatura gaussiana."
    )

    st.latex(r"K>0\Rightarrow \text{ponto elíptico}")
    st.latex(r"K<0\Rightarrow \text{ponto hiperbólico}")
    st.latex(r"K=0\Rightarrow \text{ponto parabólico ou planar}")

    st.success(f"Classificação no ponto escolhido: **{classification}**")

    st.header("Tabela numérica")

    df = pd.DataFrame(
        {
            "Quantidade": [
                "E",
                "F",
                "G",
                "e",
                "f",
                "g",
                "K",
                "H",
                "k₁",
                "k₂",
                "Classificação",
            ],
            "Valor no ponto": [
                fmt(E0),
                fmt(F0),
                fmt(G0),
                fmt(e0),
                fmt(f0),
                fmt(g0),
                fmt(K0),
                fmt(H0),
                fmt(k10),
                fmt(k20),
                classification,
            ],
        }
    )

    st.dataframe(df, hide_index=True, use_container_width=True)

    st.header("Interpretação geométrica")

    st.write(
        """
        Se \(K>0\), a superfície se curva no mesmo sentido nas duas direções principais,
        como ocorre localmente em uma esfera.

        Se \(K<0\), a superfície se curva em sentidos opostos nas direções principais,
        como ocorre em uma sela.

        Se \(H=0\), a superfície tem curvatura média nula no ponto. Essa é a condição
        que aparecerá no próximo módulo, dedicado às superfícies mínimas.
        """
    )

except Exception as e:
    st.error("Não foi possível gerar a superfície. Verifique as expressões e o domínio escolhidos.")
    st.exception(e)
