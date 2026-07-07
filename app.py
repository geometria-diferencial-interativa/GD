import streamlit as st
import numpy as np
import pandas as pd
from modules.surfaces import make_grid, surface, default_domain
from modules.geometry import compute_geometry, apply_variation, total_area
from modules.plotter import make_plot

st.set_page_config(page_title="Geometria Diferencial Interativa", layout="wide")

SURFACES = [
    "Plano",
    "Esfera",
    "Cilindro",
    "Catenoide",
    "Helicoide",
    "Scherk",
    "Enneper",
    "Toro",
    "Personalizada",
]


def closest_index(grid, value):
    line = grid[:, 0] if grid.ndim == 2 else grid
    return int(np.argmin(np.abs(line - value)))


def fmt(x, digits=6):
    try:
        x = float(x)
        if not np.isfinite(x):
            return "não definido"
        if abs(x) >= 10000 or (0 < abs(x) < 1e-4):
            return f"{x:.{digits}e}"
        return f"{x:.{digits}f}"
    except Exception:
        return "não definido"


def point_table(geom, i, j):
    data = {
        "Coeficiente": ["E", "F", "G", "e", "f", "g", "H(P)", "K(P)", "dA(P)"],
        "Valor numérico no ponto P": [
            fmt(geom["E"][i, j]),
            fmt(geom["F"][i, j]),
            fmt(geom["G"][i, j]),
            fmt(geom["e"][i, j]),
            fmt(geom["f"][i, j]),
            fmt(geom["g"][i, j]),
            fmt(geom["H"][i, j]),
            fmt(geom["K"][i, j]),
            fmt(geom["area_density"][i, j]),
        ],
    }
    return pd.DataFrame(data)


st.title("Geometria Diferencial Interativa")
st.caption(
    "Visualização de superfícies, curvatura média, curvatura gaussiana e variações normais de área."
)

with st.sidebar:
    st.header("1. Superfície")
    name = st.selectbox("Escolha a superfície", SURFACES)

    params = {}
    if name in ["Esfera", "Cilindro", "Toro"]:
        params["R"] = st.slider("Raio principal R", 0.2, 3.0, 1.0, 0.1)
    if name in ["Catenoide", "Helicoide", "Scherk"]:
        params["a"] = st.slider("Parâmetro a", 0.2, 3.0, 1.0, 0.1)
    if name == "Toro":
        params["r"] = st.slider("Raio menor r", 0.1, 1.5, 0.35, 0.05)

    custom = {}
    if name == "Personalizada":
        st.markdown(
            "Use expressões em `u` e `v`: `sin`, `cos`, `cosh`, `sinh`, `exp`, `log`, `sqrt`, `pi`."
        )
        custom["x"] = st.text_input("x(u,v)", "u")
        custom["y"] = st.text_input("y(u,v)", "v")
        custom["z"] = st.text_input("z(u,v)", "sin(u)*cos(v)")

    st.header("2. Domínio")
    u0, u1, v0, v1 = default_domain(name)
    umin = st.number_input("u mínimo", value=float(u0))
    umax = st.number_input("u máximo", value=float(u1))
    vmin = st.number_input("v mínimo", value=float(v0))
    vmax = st.number_input("v máximo", value=float(v1))
    n = st.slider("Resolução", 35, 140, 75, 5)

    st.header("3. Ponto de leitura")
    st.caption("As curvaturas são calculadas pontualmente. Escolha o ponto P=(u,v).")
    uP = st.slider("u do ponto P", float(umin), float(umax), float((umin + umax) / 2))
    vP = st.slider("v do ponto P", float(vmin), float(vmax), float((vmin + vmax) / 2))

    st.header("4. Variação normal")
    h_expr = st.text_input("h(u,v)", "sin(u)*cos(v)")
    t = st.slider("t em X_t = X + t h N", -1.0, 1.0, 0.0, 0.05)

    st.header("5. Visualização")
    color_by = st.selectbox(
        "Colorir por",
        [
            "H_t: curvatura média da superfície variada",
            "K_t: curvatura gaussiana da superfície variada",
            "H: curvatura média da superfície inicial",
            "K: curvatura gaussiana da superfície inicial",
            "Densidade de área dA_t",
            "h(u,v)",
            "z",
        ],
    )
    show_normals = st.checkbox("Mostrar campo normal", value=False)
    show_tangent = st.checkbox("Mostrar direções tangentes em P", value=True)

try:
    if umax <= umin or vmax <= vmin:
        raise ValueError("O domínio precisa satisfazer u mínimo < u máximo e v mínimo < v máximo.")

    U, V = make_grid(umin, umax, vmin, vmax, n)
    du = (umax - umin) / (n - 1)
    dv = (vmax - vmin) / (n - 1)

    X = surface(name, U, V, params, custom)
    geom = compute_geometry(X, du, dv)

    Xt, h, h_parsed = apply_variation(X, geom["N"], U, V, h_expr, t)
    geom_t = compute_geometry(Xt, du, dv)

    area0 = total_area(geom["area_density"], du, dv)
    areat = total_area(geom_t["area_density"], du, dv)

    iP = closest_index(U, uP)
    jP = closest_index(V.T, vP)
    uP_grid = float(U[iP, jP])
    vP_grid = float(V[iP, jP])
    P0 = X[iP, jP]
    Pt = Xt[iP, jP]

    st.markdown(
        r"""
Nesta versão, o estudante escolhe uma superfície parametrizada $X(u,v)$, altera parâmetros,
seleciona um ponto $P=(u,v)$ e aplica uma variação normal

$$
X_t(u,v)=X(u,v)+t\,h(u,v)N(u,v).
$$

Atenção: $H$ e $K$ são funções pontuais. Portanto, os valores numéricos exibidos no painel
correspondem ao ponto escolhido $P$.
"""
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Área aproximada de X", f"{area0:.4f}")
    col2.metric("Área aproximada de X_t", f"{areat:.4f}", f"{areat - area0:+.4f}")
    col3.metric("H_t(P)", fmt(geom_t["H"][iP, jP]))
    col4.metric("K_t(P)", fmt(geom_t["K"][iP, jP]))

    if color_by.startswith("H_t"):
        C = geom_t["H"]
        ct = "H_t"
    elif color_by.startswith("K_t"):
        C = geom_t["K"]
        ct = "K_t"
    elif color_by.startswith("H:"):
        C = geom["H"]
        ct = "H"
    elif color_by.startswith("K:"):
        C = geom["K"]
        ct = "K"
    elif color_by.startswith("Densidade"):
        C = geom_t["area_density"]
        ct = "dA_t"
    elif color_by.startswith("h"):
        C = h
        ct = "h"
    else:
        C = Xt[..., 2]
        ct = "z"

    left, right = st.columns([2.2, 1])

    with left:
        st.plotly_chart(
            make_plot(
                U,
                V,
                Xt,
                C,
                f"{name}: superfície variada X_t",
                ct,
                show_normals=show_normals,
                N=geom_t["N"],
                stride=max(5, n // 12),
                point_index=(iP, jP),
                show_tangent=show_tangent,
                Xu=geom_t["Xu"],
                Xv=geom_t["Xv"],
            ),
            use_container_width=True,
        )

    with right:
        st.subheader("Ponto escolhido")
        st.write(rf"Parâmetros: $u={fmt(uP_grid,4)}$, $v={fmt(vP_grid,4)}$")
        st.write(
            rf"$X(u,v)=({fmt(P0[0],4)}, {fmt(P0[1],4)}, {fmt(P0[2],4)})$"
        )
        st.write(
            rf"$X_t(u,v)=({fmt(Pt[0],4)}, {fmt(Pt[1],4)}, {fmt(Pt[2],4)})$"
        )
        st.write(rf"$h(u,v)={h_parsed}$")
        st.write(rf"$h(P)={fmt(h[iP,jP])}$")

        st.subheader("Valores pontuais em P")
        st.dataframe(point_table(geom_t, iP, jP), hide_index=True, use_container_width=True)

        st.subheader("Fórmulas usadas")
        st.latex(r"E=\langle X_u,X_u\rangle,\quad F=\langle X_u,X_v\rangle,\quad G=\langle X_v,X_v\rangle")
        st.latex(r"e=\langle X_{uu},N\rangle,\quad f=\langle X_{uv},N\rangle,\quad g=\langle X_{vv},N\rangle")
        st.latex(r"H=\frac{eG-2fF+gE}{2(EG-F^2)}")
        st.latex(r"K=\frac{eg-f^2}{EG-F^2}")

    st.subheader("Interpretação")
    st.write(
        "A coloração mostra a distribuição da quantidade escolhida sobre a superfície. "
        "Assim, quando colorimos por H ou K, não estamos mostrando um único número, mas sim uma função definida ponto a ponto."
    )
    st.write(
        "Para uma superfície mínima, espera-se que H seja aproximadamente zero em todos os pontos interiores. "
        "Diferenças pequenas podem aparecer por erro numérico, baixa resolução ou efeitos de borda."
    )

    st.subheader("Área aproximada em função de t")
    tmin, tmax = -1.0, 1.0
    ts = np.linspace(tmin, tmax, 31)
    vals = []
    hp_vals = []
    kp_vals = []
    for tau in ts:
        Xtau, _, _ = apply_variation(X, geom["N"], U, V, h_expr, float(tau))
        gtau = compute_geometry(Xtau, du, dv)
        vals.append(total_area(gtau["area_density"], du, dv))
        hp_vals.append(float(gtau["H"][iP, jP]))
        kp_vals.append(float(gtau["K"][iP, jP]))

    df_area = pd.DataFrame(
        {"t": ts, "Área aproximada": vals, "H_t(P)": hp_vals, "K_t(P)": kp_vals}
    ).set_index("t")
    st.line_chart(df_area[["Área aproximada"]])

    with st.expander("Ver tabela numérica da variação"):
        st.dataframe(df_area.reset_index(), use_container_width=True)

except Exception as e:
    st.error("Não foi possível gerar a superfície. Verifique domínio, parâmetros e expressões digitadas.")
    st.exception(e)
