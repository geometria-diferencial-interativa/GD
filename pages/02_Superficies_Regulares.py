import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(
    page_title="Superfícies Regulares",
    page_icon="🌐",
    layout="wide",
)


with st.sidebar:
    st.title("📐 Geometria Diferencial")

    st.page_link("app.py", label="🏠 Início")
    st.page_link("pages/01_Curvas.py", label="1. Curvas em R³")
    st.page_link(
        "pages/02_Superficies_Regulares.py",
        label="2. Superfícies Regulares"
    )

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
    U, V = np.meshgrid(u, v, indexing="ij")
    return U, V


def surface(U, V, name, a=1.0, R=1.0, r=0.35, custom=None):
    if name == "Plano":
        X = U
        Y = V
        Z = 0 * U
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

    if name == "Catenoide":
        X = a * np.cosh(V / a) * np.cos(U)
        Y = a * np.cosh(V / a) * np.sin(U)
        Z = V
        return np.stack((X, Y, Z), axis=-1)

    if name == "Helicoide":
        X = U * np.cos(V)
        Y = U * np.sin(V)
        Z = a * V
        return np.stack((X, Y, Z), axis=-1)

    if name == "Toro":
        X = (R + r * np.cos(V)) * np.cos(U)
        Y = (R + r * np.cos(V)) * np.sin(U)
        Z = r * np.sin(V)
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
    if name == "Plano":
        return -2.0, 2.0, -2.0, 2.0
    return -2.0, 2.0, -2.0, 2.0


def derivatives_surface(X, du, dv):
    Xu = np.gradient(X, du, axis=0)
    Xv = np.gradient(X, dv, axis=1)
    return Xu, Xv


def unit_normal(Xu, Xv):
    cross = np.cross(Xu, Xv)
    norm = np.linalg.norm(cross, axis=-1, keepdims=True)
    return np.divide(cross, norm, out=np.zeros_like(cross), where=norm > 1e-10), norm[..., 0]


def closest_index(grid, value, axis):
    if axis == 0:
        line = grid[:, 0]
    else:
        line = grid[0, :]
    return int(np.argmin(np.abs(line - value)))


def add_arrow(fig, p, v, label, scale=0.5):
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
            name=label,
        )
    )


def make_plot(X, U, V, i0, j0, Xu, Xv, N, show_vectors=True, show_coord=True):
    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=X[..., 0],
            y=X[..., 1],
            z=X[..., 2],
            opacity=0.75,
            colorscale="Viridis",
            showscale=False,
            name="superfície",
        )
    )

    p = X[i0, j0]

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

    if show_coord:
        fig.add_trace(
            go.Scatter3d(
                x=X[:, j0, 0],
                y=X[:, j0, 1],
                z=X[:, j0, 2],
                mode="lines",
                name="curva u ↦ X(u,v₀)",
                line=dict(width=6),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=X[i0, :, 0],
                y=X[i0, :, 1],
                z=X[i0, :, 2],
                mode="lines",
                name="curva v ↦ X(u₀,v)",
                line=dict(width=6),
            )
        )

    if show_vectors:
        add_arrow(fig, p, Xu[i0, j0], "X_u")
        add_arrow(fig, p, Xv[i0, j0], "X_v")
        add_arrow(fig, p, N[i0, j0], "N")

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


st.title("Módulo 2 — Superfícies Regulares")

st.write("Neste módulo estudamos superfícies parametrizadas regulares:")
st.latex(r"X:U\subset\mathbb{R}^2\longrightarrow \mathbb{R}^3")
st.latex(r"X(u,v)=(x(u,v),y(u,v),z(u,v))")

st.write("A regularidade é garantida quando os vetores tangentes são linearmente independentes:")
st.latex(r"X_u(u,v)\times X_v(u,v)\neq 0")


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

    st.header("Ponto de leitura")

    u0 = st.slider(
        "Escolha u₀",
        float(umin),
        float(umax),
        float((umin + umax) / 2),
    )

    v0 = st.slider(
        "Escolha v₀",
        float(vmin),
        float(vmax),
        float((vmin + vmax) / 2),
    )

    show_vectors = st.checkbox("Mostrar vetores Xᵤ, Xᵥ e N", value=True)
    show_coord = st.checkbox("Mostrar curvas coordenadas", value=True)


if umax <= umin or vmax <= vmin:
    st.error("O domínio precisa satisfazer u mínimo < u máximo e v mínimo < v máximo.")
    st.stop()


try:
    U, V = make_grid(umin, umax, vmin, vmax, n)

    du = (umax - umin) / (n - 1)
    dv = (vmax - vmin) / (n - 1)

    X = surface(U, V, name, a=a, R=R, r=r, custom=custom)

    Xu, Xv = derivatives_surface(X, du, dv)
    N, area_density = unit_normal(Xu, Xv)

    i0 = closest_index(U, u0, axis=0)
    j0 = closest_index(V, v0, axis=1)

    p = X[i0, j0]

    st.subheader("Superfície escolhida")

    st.markdown("A superfície visualizada é a aplicação parametrizada:")
    st.latex(
        rf"X:U\subset\mathbb{{R}}^2\longrightarrow \mathbb{{R}}^3,"
        rf"\qquad U=[{fmt(umin,2)},\,{fmt(umax,2)}]\times[{fmt(vmin,2)},\,{fmt(vmax,2)}]"
    )

    st.latex(surface_latex(name, custom))

    st.markdown("Para cada par $(u,v)\in U$, obtemos um ponto")
    st.latex(r"X(u,v)=(x(u,v),y(u,v),z(u,v))\in\mathbb{R}^3")

    if name == "Plano":
        st.markdown("O plano é o exemplo mais simples de superfície regular.")
    elif name == "Esfera":
        st.markdown(f"Nesta parametrização, $R={fmt(R,2)}$ representa o raio da esfera.")
    elif name == "Cilindro":
        st.markdown(f"Nesta parametrização, $R={fmt(R,2)}$ representa o raio do cilindro.")
    elif name == "Catenoide":
        st.markdown(f"O parâmetro $a={fmt(a,2)}$ controla a abertura do catenoide.")
    elif name == "Helicoide":
        st.markdown(f"O parâmetro $a={fmt(a,2)}$ controla a inclinação vertical do helicoide.")
    elif name == "Toro":
        st.markdown(f"Neste toro, $R={fmt(R,2)}$ é o raio principal e $r={fmt(r,2)}$ é o raio menor.")
    elif name == "Personalizada":
        st.markdown("Esta superfície foi definida pelo estudante a partir das três funções coordenadas.")

    left, right = st.columns([2.2, 1])

    with left:
        st.plotly_chart(
            make_plot(X, U, V, i0, j0, Xu, Xv, N, show_vectors, show_coord),
            use_container_width=True,
        )

    with right:
        st.subheader("Valores no ponto escolhido")

        st.markdown("**Parâmetros escolhidos:**")
        st.latex(rf"(u_0,v_0)=({fmt(U[i0,j0])},\,{fmt(V[i0,j0])})")

        st.markdown("**Ponto da superfície:**")
        st.latex(rf"X(u_0,v_0)=\left({fmt(p[0])},\,{fmt(p[1])},\,{fmt(p[2])}\right)")

        st.markdown("**Vetores tangentes:**")
        st.latex(
            rf"X_u(u_0,v_0)=\left({fmt(Xu[i0,j0,0])},\,{fmt(Xu[i0,j0,1])},\,{fmt(Xu[i0,j0,2])}\right)"
        )
        st.latex(
            rf"X_v(u_0,v_0)=\left({fmt(Xv[i0,j0,0])},\,{fmt(Xv[i0,j0,1])},\,{fmt(Xv[i0,j0,2])}\right)"
        )

        st.markdown("**Produto vetorial:**")
        cross = np.cross(Xu[i0, j0], Xv[i0, j0])
        st.latex(
            rf"X_u\times X_v=\left({fmt(cross[0])},\,{fmt(cross[1])},\,{fmt(cross[2])}\right)"
        )

        st.markdown("**Norma do produto vetorial:**")
        st.latex(rf"|X_u\times X_v|={fmt(area_density[i0,j0])}")

        st.markdown("**Normal unitária:**")
        st.latex(
            rf"N=\left({fmt(N[i0,j0,0])},\,{fmt(N[i0,j0,1])},\,{fmt(N[i0,j0,2])}\right)"
        )

        df = pd.DataFrame(
            {
                "Objeto": ["X_u", "X_v", "X_u × X_v", "N"],
                "Coordenada x": [
                    fmt(Xu[i0, j0, 0]),
                    fmt(Xv[i0, j0, 0]),
                    fmt(cross[0]),
                    fmt(N[i0, j0, 0]),
                ],
                "Coordenada y": [
                    fmt(Xu[i0, j0, 1]),
                    fmt(Xv[i0, j0, 1]),
                    fmt(cross[1]),
                    fmt(N[i0, j0, 1]),
                ],
                "Coordenada z": [
                    fmt(Xu[i0, j0, 2]),
                    fmt(Xv[i0, j0, 2]),
                    fmt(cross[2]),
                    fmt(N[i0, j0, 2]),
                ],
            }
        )

        st.subheader("Tabela numérica")
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Fórmulas usadas")

    st.latex(r"X_u=\frac{\partial X}{\partial u},\qquad X_v=\frac{\partial X}{\partial v}")
    st.latex(r"X_u\times X_v\neq 0")
    st.latex(r"N=\frac{X_u\times X_v}{|X_u\times X_v|}")

    st.subheader("Interpretação geométrica")

    st.write(
        """
        Os vetores \(X_u\) e \(X_v\) são tangentes às curvas coordenadas da superfície.
        Quando \(X_u\times X_v\neq 0\), esses vetores são linearmente independentes
        e determinam o plano tangente à superfície no ponto escolhido.
        """
    )

except Exception as e:
    st.error("Não foi possível gerar a superfície. Verifique as expressões e o domínio escolhidos.")
    st.exception(e)
