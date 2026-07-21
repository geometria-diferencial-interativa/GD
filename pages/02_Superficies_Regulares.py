import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(
    page_title="Superfícies Regulares",
    page_icon="🌐",
    layout="wide",
)

DIGITS = 2
TOL = 1e-8


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
    try:
        value = float(value)
        if not np.isfinite(value):
            return "não definido"
        return f"{value:.{digits}f}"
    except Exception:
        return "não definido"


def eval_expr_uv(expr, U, V):
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


def eval_expr_xyz(expr, X, Y, Z):
    allowed = {
        "x": X,
        "y": Y,
        "z": Z,
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
    u_values = np.linspace(umin, umax, n)
    v_values = np.linspace(vmin, vmax, n)
    return np.meshgrid(u_values, v_values, indexing="ij")


def broadcast_array(value, shape):
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        return np.full(shape, float(array))
    return np.broadcast_to(array, shape).astype(float)


def surface(U, V, name, a=1.0, R=1.0, r=0.35, custom=None):
    if name == "Plano":
        return np.stack((U, V, 0 * U), axis=-1)
    if name == "Paraboloide elíptico":
        return np.stack((U, V, U**2 + V**2), axis=-1)
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
    if name == "Hiperboloide de uma folha":
        X = np.cosh(V) * np.cos(U)
        Y = np.cosh(V) * np.sin(U)
        Z = np.sinh(V)
        return np.stack((X, Y, Z), axis=-1)
    if name == "Hiperboloide de duas folhas":
        X = np.sinh(V) * np.cos(U)
        Y = np.sinh(V) * np.sin(U)
        Z = np.cosh(V)
        return np.stack((X, Y, Z), axis=-1)
    if name == "Personalizada":
        X = broadcast_array(eval_expr_uv(custom["x"], U, V), U.shape)
        Y = broadcast_array(eval_expr_uv(custom["y"], U, V), U.shape)
        Z = broadcast_array(eval_expr_uv(custom["z"], U, V), U.shape)
        return np.stack((X, Y, Z), axis=-1)
    return np.stack((U, V, 0 * U), axis=-1)


def surface_latex(name, a=1.0, R=1.0, r=0.35, custom=None):
    if name == "Plano":
        return r"X(u,v)=(u,v,0)"
    if name == "Paraboloide elíptico":
        return r"X(u,v)=(u,v,u^2+v^2)"
    if name == "Esfera":
        return r"X(u,v)=(R\cos u\sin v,R\sin u\sin v,R\cos v)"
    if name == "Cilindro":
        return r"X(u,v)=(R\cos u,R\sin u,v)"
    if name == "Catenoide":
        return r"X(u,v)=(a\cosh(v/a)\cos u,a\cosh(v/a)\sin u,v)"
    if name == "Helicoide":
        return r"X(u,v)=(u\cos v,u\sin v,av)"
    if name == "Toro":
        return r"X(u,v)=((R+r\cos v)\cos u,(R+r\cos v)\sin u,r\sin v)"
    if name == "Hiperboloide de uma folha":
        return r"X(u,v)=(\cosh v\cos u,\cosh v\sin u,\sinh v)"
    if name == "Hiperboloide de duas folhas":
        return r"X(u,v)=(\sinh v\cos u,\sinh v\sin u,\cosh v)"
    if name == "Personalizada":
        return rf"X(u,v)=\left({custom['x']},{custom['y']},{custom['z']}\right)"
    return r"X(u,v)=(x(u,v),y(u,v),z(u,v))"


def default_domain(name):
    if name == "Esfera":
        return 0.0, float(2 * np.pi), 0.05, float(np.pi - 0.05)
    if name in ["Cilindro", "Catenoide"]:
        return 0.0, float(2 * np.pi), -2.0, 2.0
    if name == "Toro":
        return 0.0, float(2 * np.pi), 0.0, float(2 * np.pi)
    if name == "Helicoide":
        return -2.0, 2.0, 0.0, float(4 * np.pi)
    if name == "Hiperboloide de uma folha":
        return 0.0, float(2 * np.pi), -1.5, 1.5
    if name == "Hiperboloide de duas folhas":
        return 0.0, float(2 * np.pi), 0.15, 1.5
    return -2.0, 2.0, -2.0, 2.0


def derivatives_surface(X, du, dv):
    Xu = np.gradient(X, du, axis=0, edge_order=2)
    Xv = np.gradient(X, dv, axis=1, edge_order=2)
    return Xu, Xv


def normal_from_tangents(Xu, Xv):
    cross = np.cross(Xu, Xv)
    norm = np.linalg.norm(cross, axis=-1, keepdims=True)
    normal = np.divide(cross, norm, out=np.zeros_like(cross), where=norm > TOL)
    return cross, normal, norm[..., 0]


def closest_index(grid, value, axis):
    line = grid[:, 0] if axis == 0 else grid[0, :]
    return int(np.argmin(np.abs(line - value)))


def add_arrow(fig, point, vector, label, scale=0.5):
    norm = np.linalg.norm(vector)
    if norm < TOL:
        return
    direction = vector / norm
    endpoint = point + scale * direction
    fig.add_trace(go.Scatter3d(
        x=[point[0], endpoint[0]], y=[point[1], endpoint[1]], z=[point[2], endpoint[2]],
        mode="lines", name=label, line=dict(width=7)
    ))
    fig.add_trace(go.Cone(
        x=[endpoint[0]], y=[endpoint[1]], z=[endpoint[2]],
        u=[direction[0]], v=[direction[1]], w=[direction[2]],
        sizemode="absolute", sizeref=0.18, anchor="tip", showscale=False, showlegend=False
    ))


def make_plot(X, i0, j0, Xu, Xv, N, show_vectors=True, show_coord=True):
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X[..., 0], y=X[..., 1], z=X[..., 2], opacity=0.75,
        colorscale="Viridis", showscale=False, name="Superfície", showlegend=True
    ))
    point = X[i0, j0]
    fig.add_trace(go.Scatter3d(
        x=[point[0]], y=[point[1]], z=[point[2]], mode="markers",
        name="Ponto X(u₀,v₀)", marker=dict(size=7)
    ))
    if show_coord:
        fig.add_trace(go.Scatter3d(
            x=X[:, j0, 0], y=X[:, j0, 1], z=X[:, j0, 2],
            mode="lines", name="Curva u ↦ X(u,v₀)", line=dict(width=6)
        ))
        fig.add_trace(go.Scatter3d(
            x=X[i0, :, 0], y=X[i0, :, 1], z=X[i0, :, 2],
            mode="lines", name="Curva v ↦ X(u₀,v)", line=dict(width=6)
        ))
    if show_vectors:
        add_arrow(fig, point, Xu[i0, j0], "Xᵤ(u₀,v₀)")
        add_arrow(fig, point, Xv[i0, j0], "Xᵥ(u₀,v₀)")
        add_arrow(fig, point, N[i0, j0], "N(u₀,v₀)")
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            font=dict(size=15), itemsizing="constant", x=0.02, y=0.02,
            xanchor="left", yanchor="bottom", bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(100,100,100,0.4)", borderwidth=1
        ),
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
    )
    return fig


def normalize_text(text):
    return text.replace(" ", "").replace("**", "^").lower()


def detect_custom_method(custom, has_implicit):
    x_text = normalize_text(custom["x"])
    y_text = normalize_text(custom["y"])
    z_text = normalize_text(custom["z"])

    if x_text == "u" and y_text == "v":
        return "gráfico", "A parametrização tem a forma X(u,v)=(u,v,f(u,v))."
    if x_text == "u" and z_text == "v":
        return "gráfico", "A parametrização tem a forma X(u,v)=(u,f(u,v),v)."
    if y_text == "u" and z_text == "v":
        return "gráfico", "A parametrização tem a forma X(u,v)=(f(u,v),u,v)."
    if has_implicit:
        return "valor regular", "Foi fornecida uma equação implícita F(x,y,z)=c."
    return "definição", "A superfície foi fornecida apenas por uma parametrização geral."


def numerical_gradient_implicit(F_expr, X, Y, Z, h=1e-5):
    Fx = (eval_expr_xyz(F_expr, X + h, Y, Z) - eval_expr_xyz(F_expr, X - h, Y, Z)) / (2 * h)
    Fy = (eval_expr_xyz(F_expr, X, Y + h, Z) - eval_expr_xyz(F_expr, X, Y - h, Z)) / (2 * h)
    Fz = (eval_expr_xyz(F_expr, X, Y, Z + h) - eval_expr_xyz(F_expr, X, Y, Z - h)) / (2 * h)
    return np.stack((Fx, Fy, Fz), axis=-1)


def show_preset_proof(name, a, R, r):
    if name == "Plano":
        st.markdown("#### Demonstração pela definição")
        st.latex(r"X(u,v)=(u,v,0)")
        st.write("As funções coordenadas são suaves e a inversa é")
        st.latex(r"X^{-1}(x,y,0)=(x,y).")
        st.latex(r"X_u=(1,0,0),\qquad X_v=(0,1,0)")
        st.latex(r"X_u\times X_v=(0,0,1)\neq\vec 0.")
        st.write("Logo, X é um homeomorfismo sobre sua imagem e sua diferencial é injetora em todos os pontos.")
    elif name == "Helicoide":
        st.markdown("#### Demonstração pela definição")
        st.latex(r"X(u,v)=(u\cos v,u\sin v,av)")
        st.write("Como a terceira coordenada determina v, obtemos a inversa")
        st.latex(rf"X^{{-1}}(x,y,z)=\left(x\cos\left(\frac{{z}}{{{fmt(a)}}}\right)+y\sin\left(\frac{{z}}{{{fmt(a)}}}\right),\frac{{z}}{{{fmt(a)}}}\right).")
        st.latex(r"X_u\times X_v=(a\sin v,-a\cos v,u)")
        st.latex(r"\|X_u\times X_v\|^2=a^2+u^2>0.")
        st.write("Portanto, todas as condições da definição são satisfeitas.")
    elif name == "Catenoide":
        st.markdown("#### Demonstração pela definição")
        st.write("Restringimos o parâmetro angular u a um intervalo aberto I de comprimento menor que 2π. A inversa local é")
        st.latex(r"X_I^{-1}(x,y,z)=(\operatorname{arg}_I(x,y),z).")
        st.latex(r"X_u\times X_v=\left(a\cosh(v/a)\cos u,a\cosh(v/a)\sin u,-a\cosh(v/a)\sinh(v/a)\right)")
        st.write("Como a>0 e cosh(v/a)>0, esse produto vetorial nunca se anula.")
    elif name == "Paraboloide elíptico":
        st.markdown("#### Demonstração pela proposição do gráfico")
        st.latex(r"S=\{(u,v,u^2+v^2):(u,v)\in\mathbb{R}^2\}=\operatorname{graf}(f)")
        st.latex(r"f(u,v)=u^2+v^2.")
        st.write("Como f é de classe C∞, a proposição do gráfico garante que S é uma superfície regular.")
    elif name == "Esfera":
        st.markdown("#### Demonstração pelo Teorema da Pré-imagem de Valor Regular")
        st.latex(r"F(x,y,z)=x^2+y^2+z^2,\qquad \mathbb{S}^2(R)=F^{-1}(R^2)")
        st.latex(r"\nabla F=(2x,2y,2z).")
        st.write("Em F⁻¹(R²), temos x²+y²+z²=R²>0; logo o gradiente não se anula.")
    elif name == "Cilindro":
        st.markdown("#### Demonstração pelo Teorema da Pré-imagem de Valor Regular")
        st.latex(r"F(x,y,z)=x^2+y^2,\qquad C_R=F^{-1}(R^2)")
        st.latex(r"\nabla F=(2x,2y,0).")
        st.write("Em F⁻¹(R²), x e y não se anulam simultaneamente.")
    elif name in ["Hiperboloide de uma folha", "Hiperboloide de duas folhas"]:
        c = 1 if name == "Hiperboloide de uma folha" else -1
        st.markdown("#### Demonstração pelo Teorema da Pré-imagem de Valor Regular")
        st.latex(rf"F(x,y,z)=x^2+y^2-z^2,\qquad S=F^{{-1}}({c})")
        st.latex(r"\nabla F=(2x,2y,-2z).")
        st.write("O gradiente se anula somente na origem, que não pertence ao conjunto de nível considerado.")
    elif name == "Toro":
        st.markdown("#### Demonstração pelo Teorema da Pré-imagem de Valor Regular")
        st.latex(r"F=(x^2+y^2+z^2+R^2-r^2)^2-4R^2(x^2+y^2)")
        st.latex(r"T_{R,r}=F^{-1}(0).")
        st.write("Para R>r>0, a análise das componentes de ∇F mostra que não existe ponto de F⁻¹(0) no qual o gradiente seja nulo.")
        st.write("Assim, 0 é valor regular e o toro é uma superfície regular.")


st.title("2 - Superfícies Regulares")
st.write(
    "Neste módulo analisamos superfícies regulares pela definição, pela proposição "
    "do gráfico e pelo Teorema da Pré-imagem de Valor Regular."
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

    a, R, r = 1.0, 1.0, 0.35
    if name in ["Catenoide", "Helicoide"]:
        a = st.slider("Parâmetro a", 0.2, 3.0, 1.0, 0.1)
    if name in ["Esfera", "Cilindro", "Toro"]:
        R = st.slider("Raio principal R", 0.2, 3.0, 1.0, 0.1)
    if name == "Toro":
        r_max = max(0.15, min(1.2, R - 0.05))
        r = st.slider("Raio menor r", 0.1, float(r_max), min(0.35, float(r_max)), 0.05)
        st.caption("Consideramos R > r > 0.")

    custom = {}
    has_implicit = False
    F_text = ""
    c_value = 0.0

    if name == "Personalizada":
        st.markdown("Digite as coordenadas usando u e v.")
        st.caption("Funções: sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt, abs e pi.")
        custom["x"] = st.text_input("x(u,v)", "u")
        custom["y"] = st.text_input("y(u,v)", "v")
        custom["z"] = st.text_input("z(u,v)", "sin(u)*cos(v)")
        st.markdown("##### Equação implícita opcional")
        has_implicit = st.checkbox("Também conheço uma equação F(x,y,z)=c")
        if has_implicit:
            F_text = st.text_input("F(x,y,z)", "x**2+y**2+z**2")
            c_value = st.number_input("Valor c", value=1.0)

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
    U, V = make_grid(umin, umax, vmin, vmax, n)
    du = (umax - umin) / (n - 1)
    dv = (vmax - vmin) / (n - 1)

    X = surface(U, V, name, a=a, R=R, r=r, custom=custom)
    if not np.all(np.isfinite(X)):
        st.error("A parametrização não está definida em todo o domínio escolhido.")
        st.stop()

    Xu, Xv = derivatives_surface(X, du, dv)
    cross, N, area_density = normal_from_tangents(Xu, Xv)

    i0 = closest_index(U, u0, axis=0)
    j0 = closest_index(V, v0, axis=1)
    point = X[i0, j0]
    cross_point = cross[i0, j0]

    st.subheader("Superfície escolhida")
    st.latex(
        rf"X:U\subset\mathbb{{R}}^2\longrightarrow\mathbb{{R}}^3,\qquad "
        rf"U=[{fmt(umin)},\,{fmt(umax)}]\times[{fmt(vmin)},\,{fmt(vmax)}]"
    )
    st.latex(surface_latex(name, a=a, R=R, r=r, custom=custom))

    left, right = st.columns([2.2, 1])
    with left:
        st.plotly_chart(
            make_plot(X, i0, j0, Xu, Xv, N, show_vectors, show_coord),
            use_container_width=True,
        )

    with right:
        st.subheader("Valores no ponto escolhido")
        st.latex(rf"(u_0,v_0)=({fmt(U[i0,j0])},\,{fmt(V[i0,j0])})")
        st.latex(rf"X(u_0,v_0)=\left({fmt(point[0])},\,{fmt(point[1])},\,{fmt(point[2])}\right)")
        st.markdown("**Vetores tangentes:**")
        st.latex(rf"X_u=\left({fmt(Xu[i0,j0,0])},\,{fmt(Xu[i0,j0,1])},\,{fmt(Xu[i0,j0,2])}\right)")
        st.latex(rf"X_v=\left({fmt(Xv[i0,j0,0])},\,{fmt(Xv[i0,j0,1])},\,{fmt(Xv[i0,j0,2])}\right)")
        st.markdown("**Produto vetorial:**")
        st.latex(rf"X_u\times X_v=\left({fmt(cross_point[0])},\,{fmt(cross_point[1])},\,{fmt(cross_point[2])}\right)")
        st.latex(rf"\|X_u\times X_v\|={fmt(area_density[i0,j0])}")

        if area_density[i0, j0] > TOL:
            st.success("No ponto escolhido, a diferencial possui posto 2.")
            st.latex(rf"N=\left({fmt(N[i0,j0,0])},\,{fmt(N[i0,j0,1])},\,{fmt(N[i0,j0,2])}\right)")
        else:
            st.error("No ponto escolhido, a parametrização é singular.")
            st.latex(r"N=\text{não definida}")

        df = pd.DataFrame({
            "Objeto": ["X_u", "X_v", "X_u × X_v", "N"],
            "Coordenada x": [fmt(Xu[i0,j0,0]), fmt(Xv[i0,j0,0]), fmt(cross_point[0]), fmt(N[i0,j0,0]) if area_density[i0,j0] > TOL else "não definida"],
            "Coordenada y": [fmt(Xu[i0,j0,1]), fmt(Xv[i0,j0,1]), fmt(cross_point[1]), fmt(N[i0,j0,1]) if area_density[i0,j0] > TOL else "não definida"],
            "Coordenada z": [fmt(Xu[i0,j0,2]), fmt(Xv[i0,j0,2]), fmt(cross_point[2]), fmt(N[i0,j0,2]) if area_density[i0,j0] > TOL else "não definida"],
        })
        st.subheader("Tabela numérica")
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Análise numérica da regularidade da parametrização")
    sampled_min = float(np.nanmin(area_density))
    st.write(
        "No reticulado numérico exibido, o menor valor encontrado para "
        f"$\\|X_u\\times X_v\\|$ foi {fmt(sampled_min)}."
    )
    if sampled_min <= TOL:
        st.warning("Foram detectados pontos singulares ou muito próximos de pontos singulares no domínio amostrado.")
    else:
        st.success("Não foram detectados pontos singulares no reticulado numérico escolhido.")

    st.subheader("Demonstração de que a superfície é regular")
    if name != "Personalizada":
        show_preset_proof(name, a, R, r)
    else:
        method, reason = detect_custom_method(custom, has_implicit)
        st.info(f"Método recomendado: {method}. {reason}")

        if method == "gráfico":
            st.markdown("#### Aplicação da proposição do gráfico")
            st.write("Duas coordenadas coincidem com os parâmetros u e v. Assim, a terceira coordenada define uma função suave das outras duas.")
            st.write("Se a expressão estiver definida e for suave em todo o domínio, a proposição do gráfico garante a regularidade da superfície.")
        elif method == "valor regular":
            st.markdown("#### Aplicação do Teorema da Pré-imagem de Valor Regular")
            st.latex(rf"F(x,y,z)={F_text},\qquad c={fmt(c_value)}")
            grad_values = numerical_gradient_implicit(F_text, X[...,0], X[...,1], X[...,2])
            grad_norm = np.linalg.norm(grad_values, axis=-1)
            level_error = np.abs(eval_expr_xyz(F_text, X[...,0], X[...,1], X[...,2]) - c_value)
            st.write(f"Maior erro numérico encontrado em F(X(u,v))=c: {fmt(np.nanmax(level_error))}.")
            st.write(f"Menor valor numérico encontrado para ‖∇F‖ sobre a malha: {fmt(np.nanmin(grad_norm))}.")
            st.warning("Essa verificação é apenas numérica. Para uma demonstração, ainda é necessário provar que ∇F não se anula em nenhum ponto de F⁻¹(c).")
        else:
            st.markdown("#### Verificação pela definição")
            st.write("A plataforma verifica numericamente o posto da diferencial por meio de $X_u\times X_v$.")
            st.warning("Para concluir pela definição, ainda é necessário justificar que a parametrização é um homeomorfismo local sobre sua imagem. Essa condição não pode ser decidida automaticamente para uma parametrização arbitrária.")

    
except Exception as error:
    st.error("Não foi possível gerar ou analisar a superfície. Verifique as expressões e o domínio escolhidos.")
    st.exception(error)
