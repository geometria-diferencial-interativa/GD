import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp


st.set_page_config(
    page_title="Superfícies Regulares",
    page_icon="🌐",
    layout="wide",
)


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

DIGITS = 2
TOL = 1e-8

u, v = sp.symbols("u v", real=True)
x, y, z = sp.symbols("x y z", real=True)


with st.sidebar:
    st.title("📐 Geometria Diferencial")
    st.page_link("app.py", label="🏠 Início")
    st.page_link("pages/01_Curvas.py", label="1. Curvas em R³")
    st.page_link("pages/02_Superficies_Regulares.py", label="2. Superfícies Regulares")
    st.page_link("pages/03_Plano_Tangente_Normal.py", label="3. Plano Tangente e Vetor Normal")
    st.page_link("pages/04_Formas_Fundamentais.py", label="4. Formas Fundamentais")
    st.page_link("pages/05_Curvaturas.py", label="5. Curvaturas")
    st.page_link(
        "pages/06_Superficies_Minimas_Variacao_Area.py",
        label="6. Superfícies Mínimas e Variação da Área",
    )


def fmt(value, digits=DIGITS):
    """Formata todos os valores numéricos com duas casas decimais."""
    try:
        value = float(value)
        if not np.isfinite(value):
            return "não definido"
        return f"{value:.{digits}f}"
    except Exception:
        return "não definido"


def latex_num(value):
    return fmt(value, DIGITS).replace("-", r"-")


def latex_expr(expr):
    return sp.latex(sp.simplify(expr))


def safe_sympify(expr, variables):
    allowed = {
        **variables,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
        "abs": sp.Abs,
        "pi": sp.pi,
    }
    return sp.sympify(expr, locals=allowed)


def make_grid(umin, umax, vmin, vmax, n):
    u_values = np.linspace(umin, umax, n)
    v_values = np.linspace(vmin, vmax, n)
    return np.meshgrid(u_values, v_values, indexing="ij")


def broadcast_array(value, shape):
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        return np.full(shape, float(array))
    return np.broadcast_to(array, shape).astype(float)


def evaluate_parametrization(exprs, U, V):
    functions = [sp.lambdify((u, v), expr, modules="numpy") for expr in exprs]
    values = [broadcast_array(function(U, V), U.shape) for function in functions]
    return np.stack(values, axis=-1)


def evaluate_vector(exprs, U, V):
    return evaluate_parametrization(exprs, U, V)


def symbolic_surface(name, a=1.0, R=1.0, r=0.35, custom=None):
    """Retorna a parametrização simbólica da superfície escolhida."""
    aa = sp.Float(a)
    RR = sp.Float(R)
    rr = sp.Float(r)

    if name == "Plano":
        return (u, v, sp.Integer(0))
    if name == "Paraboloide elíptico":
        return (u, v, u**2 + v**2)
    if name == "Esfera":
        return (
            RR * sp.cos(u) * sp.sin(v),
            RR * sp.sin(u) * sp.sin(v),
            RR * sp.cos(v),
        )
    if name == "Cilindro":
        return (RR * sp.cos(u), RR * sp.sin(u), v)
    if name == "Catenoide":
        return (
            aa * sp.cosh(v / aa) * sp.cos(u),
            aa * sp.cosh(v / aa) * sp.sin(u),
            v,
        )
    if name == "Helicoide":
        return (u * sp.cos(v), u * sp.sin(v), aa * v)
    if name == "Toro":
        return (
            (RR + rr * sp.cos(v)) * sp.cos(u),
            (RR + rr * sp.cos(v)) * sp.sin(u),
            rr * sp.sin(v),
        )
    if name == "Hiperboloide de uma folha":
        return (
            sp.cosh(v) * sp.cos(u),
            sp.cosh(v) * sp.sin(u),
            sp.sinh(v),
        )
    if name == "Hiperboloide de duas folhas":
        return (
            sp.sinh(v) * sp.cos(u),
            sp.sinh(v) * sp.sin(u),
            sp.cosh(v),
        )
    if name == "Personalizada":
        return (
            safe_sympify(custom["x"], {"u": u, "v": v}),
            safe_sympify(custom["y"], {"u": u, "v": v}),
            safe_sympify(custom["z"], {"u": u, "v": v}),
        )
    return (u, v, sp.Integer(0))


def default_domain(name):
    if name == "Esfera":
        return 0.0, float(2 * np.pi), 0.05, float(np.pi - 0.05)
    if name in ["Cilindro", "Catenoide"]:
        return 0.0, float(2 * np.pi), -2.0, 2.0
    if name == "Toro":
        return 0.0, float(2 * np.pi), 0.0, float(2 * np.pi)
    if name == "Helicoide":
        return -2.0, 2.0, 0.0, float(4 * np.pi)
    if name == "Paraboloide elíptico":
        return -2.0, 2.0, -2.0, 2.0
    if name == "Hiperboloide de uma folha":
        return 0.0, float(2 * np.pi), -1.5, 1.5
    if name == "Hiperboloide de duas folhas":
        return 0.0, float(2 * np.pi), 0.15, 1.5
    return -2.0, 2.0, -2.0, 2.0


def derivatives_symbolic(exprs):
    X_matrix = sp.Matrix(exprs)
    Xu = tuple(sp.simplify(entry) for entry in X_matrix.diff(u))
    Xv = tuple(sp.simplify(entry) for entry in X_matrix.diff(v))
    cross = tuple(
        sp.simplify(entry)
        for entry in sp.Matrix(Xu).cross(sp.Matrix(Xv))
    )
    norm_sq = sp.simplify(sum(entry**2 for entry in cross))
    return Xu, Xv, cross, norm_sq


def normal_from_cross(cross_values):
    norm = np.linalg.norm(cross_values, axis=-1, keepdims=True)
    normal = np.divide(
        cross_values,
        norm,
        out=np.zeros_like(cross_values),
        where=norm > TOL,
    )
    return normal, norm[..., 0]


def closest_index(grid, value, axis):
    line = grid[:, 0] if axis == 0 else grid[0, :]
    return int(np.argmin(np.abs(line - value)))


def add_arrow(fig, point, vector, label, scale=0.5):
    norm = np.linalg.norm(vector)
    if norm < TOL:
        return
    direction = vector / norm
    endpoint = point + scale * direction

    fig.add_trace(
        go.Scatter3d(
            x=[point[0], endpoint[0]],
            y=[point[1], endpoint[1]],
            z=[point[2], endpoint[2]],
            mode="lines",
            name=label,
            showlegend=True,
            line=dict(width=7),
        )
    )
    fig.add_trace(
        go.Cone(
            x=[endpoint[0]],
            y=[endpoint[1]],
            z=[endpoint[2]],
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


def make_plot(X, i0, j0, Xu, Xv, N, show_vectors=True, show_coord=True):
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=X[..., 0],
            y=X[..., 1],
            z=X[..., 2],
            opacity=0.75,
            colorscale="Viridis",
            showscale=False,
            name="Superfície",
            showlegend=True,
        )
    )

    point = X[i0, j0]
    fig.add_trace(
        go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
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
        add_arrow(fig, point, Xu[i0, j0], "Xᵤ(u₀,v₀)")
        add_arrow(fig, point, Xv[i0, j0], "Xᵥ(u₀,v₀)")
        add_arrow(fig, point, N[i0, j0], "N(u₀,v₀)")

    fig.update_layout(
        height=650,
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
# CLASSIFICAÇÃO DO MÉTODO DE PROVA
# ============================================================

METHOD_BY_SURFACE = {
    "Plano": "definição",
    "Helicoide": "definição",
    "Catenoide": "definição",
    "Paraboloide elíptico": "gráfico",
    "Esfera": "valor regular",
    "Cilindro": "valor regular",
    "Toro": "valor regular",
    "Hiperboloide de uma folha": "valor regular",
    "Hiperboloide de duas folhas": "valor regular",
}


def detect_graph_form(exprs):
    """
    Detecta gráficos explícitos nas três direções coordenadas.
    A detecção é deliberadamente conservadora.
    """
    X1, X2, X3 = map(sp.simplify, exprs)

    cases = [
        (sp.simplify(X1 - u) == 0 and sp.simplify(X2 - v) == 0, "z", X3),
        (sp.simplify(X1 - v) == 0 and sp.simplify(X2 - u) == 0, "z", X3.subs({u: v, v: u}, simultaneous=True)),
        (sp.simplify(X1 - u) == 0 and sp.simplify(X3 - v) == 0, "y", X2),
        (sp.simplify(X1 - v) == 0 and sp.simplify(X3 - u) == 0, "y", X2.subs({u: v, v: u}, simultaneous=True)),
        (sp.simplify(X2 - u) == 0 and sp.simplify(X3 - v) == 0, "x", X1),
        (sp.simplify(X2 - v) == 0 and sp.simplify(X3 - u) == 0, "x", X1.subs({u: v, v: u}, simultaneous=True)),
    ]
    for condition, dependent, function in cases:
        if condition:
            return {
                "is_graph": True,
                "dependent": dependent,
                "function": sp.simplify(function),
            }
    return {"is_graph": False}


def analyze_implicit_custom(F_text, c_value):
    F_expr = safe_sympify(F_text, {"x": x, "y": y, "z": z})
    grad = tuple(sp.simplify(sp.diff(F_expr, variable)) for variable in (x, y, z))
    return F_expr, sp.Float(c_value), grad


def recommend_custom_method(exprs, implicit_data=None):
    graph = detect_graph_form(exprs)
    if graph["is_graph"]:
        return "gráfico", graph

    if implicit_data is not None:
        return "valor regular", implicit_data

    return "definição", {
        "warning": (
            "A aplicação está dada apenas de forma paramétrica. O programa pode "
            "verificar simbolicamente o posto da diferencial e procurar pontos "
            "singulares no domínio exibido, mas a condição de homeomorfismo local "
            "deve ser justificada matematicamente."
        )
    }


# ============================================================
# DEMONSTRAÇÕES
# ============================================================

def show_definition_proof(name, exprs, Xu_expr, Xv_expr, cross_expr, norm_sq_expr, a):
    st.markdown("#### Demonstração pela definição de superfície regular")
    st.markdown(
        "Devemos verificar a suavidade da parametrização, a condição de "
        "homeomorfismo sobre a imagem e a injetividade da diferencial."
    )
    st.latex(
        rf"X(u,v)=\left({latex_expr(exprs[0])},"
        rf"{latex_expr(exprs[1])},{latex_expr(exprs[2])}\right)"
    )
    st.markdown(
        "As funções coordenadas são de classe $C^{\infty}$ no domínio considerado."
    )

    if name == "Plano":
        st.latex(r"X^{-1}(x,y,0)=(x,y)")
        st.markdown(
            "A inversa é contínua; portanto, $X$ é um homeomorfismo entre "
            "$\mathbb{R}^{2}$ e o plano."
        )
    elif name == "Helicoide":
        st.latex(
            rf"X^{{-1}}(x,y,z)=\left("
            rf"x\cos\left(\frac{{z}}{{{fmt(a)}}}\right)"
            rf"+y\sin\left(\frac{{z}}{{{fmt(a)}}}\right),"
            rf"\frac{{z}}{{{fmt(a)}}}\right)"
        )
        st.markdown(
            "Como $a>0$, essa inversa é contínua, logo a parametrização é um "
            "homeomorfismo sobre o helicoide."
        )
    elif name == "Catenoide":
        st.markdown(
            "Devido à periodicidade angular, restringimos $u$ a um intervalo "
            "aberto $I$ de comprimento menor que $2\pi$. Em cada setor angular, "
            "a inversa local é"
        )
        st.latex(r"X_I^{-1}(x,y,z)=\bigl(\operatorname{arg}_I(x,y),z\bigr)")
        st.markdown(
            "Essa aplicação é contínua, e as cartas obtidas cobrem o catenoide."
        )
    else:
        st.warning(
            "Para uma superfície personalizada, a verificação de que a "
            "parametrização é um homeomorfismo local não pode ser decidida "
            "automaticamente em geral. É necessário fornecer uma inversa local "
            "ou uma justificativa topológica."
        )

    st.latex(
        rf"X_u=\left({latex_expr(Xu_expr[0])},{latex_expr(Xu_expr[1])},"
        rf"{latex_expr(Xu_expr[2])}\right)"
    )
    st.latex(
        rf"X_v=\left({latex_expr(Xv_expr[0])},{latex_expr(Xv_expr[1])},"
        rf"{latex_expr(Xv_expr[2])}\right)"
    )
    st.latex(
        rf"X_u\times X_v=\left({latex_expr(cross_expr[0])},"
        rf"{latex_expr(cross_expr[1])},{latex_expr(cross_expr[2])}\right)"
    )
    st.latex(rf"\|X_u\times X_v\|^2={latex_expr(norm_sq_expr)}")
    st.markdown(
        "Quando essa expressão é estritamente positiva em todo o domínio, "
        "a diferencial possui posto $2$ e é injetora."
    )


def show_graph_proof(exprs, graph_data):
    st.markdown("#### Demonstração pela proposição do gráfico")
    dependent = graph_data["dependent"]
    function = graph_data["function"]

    if dependent == "z":
        st.latex(rf"z=f(u,v)={latex_expr(function)}")
        st.markdown(
            "As duas primeiras coordenadas da parametrização são $u$ e $v$. "
            "Logo, a superfície é o gráfico da função"
        )
        st.latex(rf"f:\mathbb{{R}}^2\longrightarrow\mathbb{{R}},\qquad f(u,v)={latex_expr(function)}")
    elif dependent == "y":
        st.latex(rf"y=f(u,v)={latex_expr(function)}")
        st.markdown(
            "Após escolhermos as coordenadas independentes $x$ e $z$, a "
            "superfície é o gráfico de uma função suave."
        )
    else:
        st.latex(rf"x=f(u,v)={latex_expr(function)}")
        st.markdown(
            "Após escolhermos as coordenadas independentes $y$ e $z$, a "
            "superfície é o gráfico de uma função suave."
        )

    st.markdown(
        "Como a função é de classe $C^{\infty}$ no domínio considerado, "
        "a proposição do gráfico garante que a superfície é regular."
    )


def implicit_data_for_preset(name, R, r):
    RR = sp.Float(R)
    rr = sp.Float(r)

    if name == "Esfera":
        F = x**2 + y**2 + z**2
        c = RR**2
    elif name == "Cilindro":
        F = x**2 + y**2
        c = RR**2
    elif name == "Hiperboloide de uma folha":
        F = x**2 + y**2 - z**2
        c = sp.Integer(1)
    elif name == "Hiperboloide de duas folhas":
        F = x**2 + y**2 - z**2
        c = sp.Integer(-1)
    elif name == "Toro":
        F = (x**2 + y**2 + z**2 + RR**2 - rr**2)**2 - 4 * RR**2 * (x**2 + y**2)
        c = sp.Integer(0)
    else:
        return None

    grad = tuple(sp.simplify(sp.diff(F, variable)) for variable in (x, y, z))
    return F, c, grad


def show_regular_value_proof(name, implicit_data):
    F_expr, c_expr, grad = implicit_data
    st.markdown("#### Demonstração pelo Teorema da Pré-imagem de Valor Regular")
    st.latex(
        rf"F:\mathbb{{R}}^3\longrightarrow\mathbb{{R}},\qquad "
        rf"F(x,y,z)={latex_expr(F_expr)}"
    )
    st.latex(rf"S=F^{{-1}}\left({latex_expr(c_expr)}\right)")
    st.latex(
        rf"\nabla F(x,y,z)=\left({latex_expr(grad[0])},"
        rf"{latex_expr(grad[1])},{latex_expr(grad[2])}\right)"
    )

    if name == "Esfera":
        st.markdown(
            "Sobre o conjunto de nível, $x^{2}+y^{2}+z^{2}=R^{2}>0$. "
            "Assim, $(x,y,z)\neq(0,0,0)$ e o gradiente não se anula."
        )
    elif name == "Cilindro":
        st.markdown(
            "Sobre o conjunto de nível, $x^{2}+y^{2}=R^{2}>0$. "
            "Logo, $x$ e $y$ não se anulam simultaneamente e "
            "$\nabla F\neq\vec 0$."
        )
    elif name in ["Hiperboloide de uma folha", "Hiperboloide de duas folhas"]:
        st.markdown(
            "O gradiente se anula apenas na origem, mas a origem não pertence "
            "ao conjunto de nível correspondente a $c\neq0$."
        )
    elif name == "Toro":
        st.markdown(
            "Para $R>r>0$, a análise das componentes do gradiente mostra que "
            "não existe ponto de $F^{-1}(0)$ no qual $\nabla F$ seja nulo."
        )
    else:
        st.info(
            "Para concluir pelo teorema, é necessário verificar que o gradiente "
            "não se anula em nenhum ponto do conjunto de nível."
        )

    st.markdown(
        "Portanto, o valor considerado é regular e o Teorema da Pré-imagem "
        "de Valor Regular garante que o conjunto é uma superfície regular."
    )


# ============================================================
# INTERFACE
# ============================================================

st.title("2 - Superfícies Regulares")
st.write(
    "Neste módulo analisamos superfícies regulares utilizando três métodos: "
    "a definição, a proposição do gráfico e o Teorema da Pré-imagem de "
    "Valor Regular."
)

with st.sidebar:
    st.header("Superfície")
    name = st.selectbox(
        "Escolha uma superfície",
        [
            "Plano",
            "Paraboloide elíptico",
            "Helicoide",
            "Catenoide",
            "Esfera",
            "Cilindro",
            "Hiperboloide de uma folha",
            "Hiperboloide de duas folhas",
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
        r_max = max(0.15, min(1.2, R - 0.05))
        r_default = min(0.35, r_max)
        r = st.slider("Raio menor r", 0.1, float(r_max), float(r_default), 0.05)
        st.caption("Para o toro regular usual, consideramos R > r > 0.")

    custom = {}
    implicit_custom = None

    if name == "Personalizada":
        st.markdown("Digite as coordenadas usando as variáveis `u` e `v`.")
        st.caption(
            "Funções disponíveis: sin, cos, tan, sinh, cosh, tanh, "
            "exp, log, sqrt, abs e pi."
        )
        custom["x"] = st.text_input("x(u,v)", "u")
        custom["y"] = st.text_input("y(u,v)", "v")
        custom["z"] = st.text_input("z(u,v)", "sin(u)*cos(v)")

        st.markdown("##### Informação implícita opcional")
        has_implicit = st.checkbox(
            "Também conheço uma equação F(x,y,z)=c para esta superfície"
        )
        if has_implicit:
            F_text = st.text_input("F(x,y,z)", "x**2+y**2+z**2")
            c_value = st.number_input("Valor c", value=1.0)
            implicit_custom = {"F_text": F_text, "c": c_value}

    st.header("Domínio")
    du0, du1, dv0, dv1 = default_domain(name)
    umin = st.number_input("u mínimo", value=float(du0))
    umax = st.number_input("u máximo", value=float(du1))
    vmin = st.number_input("v mínimo", value=float(dv0))
    vmax = st.number_input("v máximo", value=float(dv1))
    n = st.slider("Resolução", 40, 160, 80, 10)

    st.header("Ponto do domínio U")
    u0 = st.slider("Escolha u₀", float(umin), float(umax), float((umin + umax) / 2))
    v0 = st.slider("Escolha v₀", float(vmin), float(vmax), float((vmin + vmax) / 2))
    show_vectors = st.checkbox("Mostrar vetores Xᵤ, Xᵥ e N", value=True)
    show_coord = st.checkbox("Mostrar curvas coordenadas", value=True)


if umax <= umin or vmax <= vmin:
    st.error("O domínio precisa satisfazer u mínimo < u máximo e v mínimo < v máximo.")
    st.stop()


try:
    exprs = symbolic_surface(name, a=a, R=R, r=r, custom=custom)
    Xu_expr, Xv_expr, cross_expr, norm_sq_expr = derivatives_symbolic(exprs)

    U, V = make_grid(umin, umax, vmin, vmax, n)
    X = evaluate_parametrization(exprs, U, V)
    Xu = evaluate_vector(Xu_expr, U, V)
    Xv = evaluate_vector(Xv_expr, U, V)
    cross_values = evaluate_vector(cross_expr, U, V)
    N, area_density = normal_from_cross(cross_values)

    if not np.all(np.isfinite(X)):
        st.error("A parametrização não está definida em todos os pontos do domínio escolhido.")
        st.stop()

    i0 = closest_index(U, u0, axis=0)
    j0 = closest_index(V, v0, axis=1)
    point = X[i0, j0]
    cross_point = cross_values[i0, j0]

    st.subheader("Superfície escolhida")
    st.latex(
        rf"X:U\subset\mathbb{{R}}^2\longrightarrow\mathbb{{R}}^3,\qquad "
        rf"U=[{fmt(umin)},\,{fmt(umax)}]\times[{fmt(vmin)},\,{fmt(vmax)}]"
    )
    st.latex(
        rf"X(u,v)=\left({latex_expr(exprs[0])},"
        rf"{latex_expr(exprs[1])},{latex_expr(exprs[2])}\right)"
    )

    left, right = st.columns([2.2, 1])

    with left:
        st.plotly_chart(
            make_plot(X, i0, j0, Xu, Xv, N, show_vectors, show_coord),
            use_container_width=True,
        )

    with right:
        st.subheader("Valores no ponto escolhido")
        st.latex(rf"(u_0,v_0)=({fmt(U[i0,j0])},\,{fmt(V[i0,j0])})")
        st.latex(
            rf"X(u_0,v_0)=\left({fmt(point[0])},\,{fmt(point[1])},\,{fmt(point[2])}\right)"
        )
        st.markdown("**Vetores tangentes:**")
        st.latex(
            rf"X_u(u_0,v_0)=\left({fmt(Xu[i0,j0,0])},"
            rf"{fmt(Xu[i0,j0,1])},{fmt(Xu[i0,j0,2])}\right)"
        )
        st.latex(
            rf"X_v(u_0,v_0)=\left({fmt(Xv[i0,j0,0])},"
            rf"{fmt(Xv[i0,j0,1])},{fmt(Xv[i0,j0,2])}\right)"
        )
        st.markdown("**Produto vetorial:**")
        st.latex(
            rf"X_u\times X_v=\left({fmt(cross_point[0])},"
            rf"{fmt(cross_point[1])},{fmt(cross_point[2])}\right)"
        )
        st.latex(rf"\|X_u\times X_v\|={fmt(area_density[i0,j0])}")

        if area_density[i0, j0] > TOL:
            st.success("No ponto escolhido, a diferencial possui posto 2.")
        else:
            st.error("No ponto escolhido, a parametrização é singular.")

        st.markdown("**Normal unitária:**")
        if area_density[i0, j0] > TOL:
            st.latex(
                rf"N=\left({fmt(N[i0,j0,0])},"
                rf"{fmt(N[i0,j0,1])},{fmt(N[i0,j0,2])}\right)"
            )
        else:
            st.latex(r"N=\text{não definida}")

        df = pd.DataFrame(
            {
                "Objeto": ["X_u", "X_v", "X_u × X_v", "N"],
                "Coordenada x": [
                    fmt(Xu[i0, j0, 0]),
                    fmt(Xv[i0, j0, 0]),
                    fmt(cross_point[0]),
                    fmt(N[i0, j0, 0]) if area_density[i0, j0] > TOL else "não definida",
                ],
                "Coordenada y": [
                    fmt(Xu[i0, j0, 1]),
                    fmt(Xv[i0, j0, 1]),
                    fmt(cross_point[1]),
                    fmt(N[i0, j0, 1]) if area_density[i0, j0] > TOL else "não definida",
                ],
                "Coordenada z": [
                    fmt(Xu[i0, j0, 2]),
                    fmt(Xv[i0, j0, 2]),
                    fmt(cross_point[2]),
                    fmt(N[i0, j0, 2]) if area_density[i0, j0] > TOL else "não definida",
                ],
            }
        )
        st.subheader("Tabela numérica")
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Análise simbólica da diferencial")
    st.latex(
        rf"X_u=\left({latex_expr(Xu_expr[0])},{latex_expr(Xu_expr[1])},"
        rf"{latex_expr(Xu_expr[2])}\right)"
    )
    st.latex(
        rf"X_v=\left({latex_expr(Xv_expr[0])},{latex_expr(Xv_expr[1])},"
        rf"{latex_expr(Xv_expr[2])}\right)"
    )
    st.latex(
        rf"X_u\times X_v=\left({latex_expr(cross_expr[0])},"
        rf"{latex_expr(cross_expr[1])},{latex_expr(cross_expr[2])}\right)"
    )
    st.latex(rf"\|X_u\times X_v\|^2={latex_expr(norm_sq_expr)}")

    sampled_min = float(np.nanmin(area_density))
    st.caption(
        "No reticulado numérico exibido, o menor valor encontrado para "
        f"‖Xᵤ×Xᵥ‖ foi {fmt(sampled_min)}."
    )
    if sampled_min <= TOL:
        st.warning(
            "Foram detectados pontos singulares ou muito próximos de pontos "
            "singulares no domínio amostrado."
        )

    st.subheader("Demonstração de que a superfície é regular")

    if name != "Personalizada":
        method = METHOD_BY_SURFACE[name]
        st.info(
            {
                "definição": "Método escolhido: Definição de superfície regular.",
                "gráfico": "Método escolhido: Proposição do gráfico de uma função suave.",
                "valor regular": "Método escolhido: Teorema da Pré-imagem de Valor Regular.",
            }[method]
        )

        if method == "definição":
            show_definition_proof(
                name, exprs, Xu_expr, Xv_expr, cross_expr, norm_sq_expr, a
            )
        elif method == "gráfico":
            show_graph_proof(exprs, detect_graph_form(exprs))
        else:
            show_regular_value_proof(name, implicit_data_for_preset(name, R, r))

    else:
        implicit_data = None
        if implicit_custom is not None:
            implicit_data = analyze_implicit_custom(
                implicit_custom["F_text"], implicit_custom["c"]
            )

        method, details = recommend_custom_method(exprs, implicit_data)

        if method == "gráfico":
            st.success(
                "A forma inserida foi reconhecida como gráfico de uma função. "
                "Esse é o método mais direto."
            )
            show_graph_proof(exprs, details)

        elif method == "valor regular":
            st.success(
                "Como foi fornecida uma equação implícita, o método recomendado "
                "é o Teorema da Pré-imagem de Valor Regular."
            )
            show_regular_value_proof("Personalizada", details)

            F_expr, c_expr, grad = details
            st.warning(
                "A plataforma exibe o gradiente, mas a afirmação global "
                "“∇F não se anula em todo F⁻¹(c)” precisa ser demonstrada. "
                "Uma verificação numérica isolada não substitui essa prova."
            )

        else:
            st.info(
                "A superfície não foi reconhecida como um gráfico explícito e "
                "não foi fornecida uma equação implícita. Assim, o método "
                "disponível é a definição."
            )
            show_definition_proof(
                "Personalizada",
                exprs,
                Xu_expr,
                Xv_expr,
                cross_expr,
                norm_sq_expr,
                a,
            )
            st.warning(details["warning"])

    st.subheader("Interpretação geométrica")
    st.write(
        "A condição $X_u\times X_v\neq\vec 0$ garante que a diferencial "
        "possui posto $2$. Entretanto, para uma parametrização definir uma "
        "superfície regular pela definição, também precisamos verificar a "
        "condição topológica de homeomorfismo local."
    )

except Exception as error:
    st.error(
        "Não foi possível gerar ou analisar a superfície. "
        "Verifique as expressões e o domínio escolhidos."
    )
    st.exception(error)
