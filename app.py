import streamlit as st


# ============================================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# ============================================================


st.set_page_config(
    page_title="Geometria Diferencial Interativa",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ESTILO DA PLATAFORMA
# ============================================================

st.markdown(
    """
    <style>

    /* Título principal */
    .main-title {
        font-size: 3.2rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Subtítulo */
    .subtitle {
        font-size: 1.3rem;
        text-align: center;
        color: #666666;
        margin-bottom: 2.5rem;
    }

    /* Caixa de apresentação */
    .presentation-box {
        padding: 2rem;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.08);
        margin-top: 1rem;
        margin-bottom: 2rem;
    }

    /* Títulos das seções */
    .section-title {
        font-size: 2rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* Cartões */
    .card {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.08);
        min-height: 220px;
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 1.35rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:

    st.title("📐 Geometria Diferencial")

    st.markdown(
        """
        Esta plataforma foi desenvolvida para o estudo
        interativo de conceitos de Geometria Diferencial.
        """
    )

    st.divider()

    st.subheader("Percurso de estudo")

    st.page_link(
        "app.py",
        label="🏠 Início"
    )

    st.page_link(
        "pages/01_Curvas.py",
        label="1. Curvas em R³"
    )

    st.page_link(
        "pages/02_Superficies_Regulares.py",
        label="2. Superfícies Regulares"
    )

    st.page_link(
        "pages/03_Plano_Tangente_Normal.py",
        label="3. Plano Tangente e Vetor Normal"
    )

    st.page_link(
        "pages/04_Primeira_Forma_Fundamental.py",
        label="4. Primeira Forma Fundamental"
    )

    st.markdown("5. Segunda Forma Fundamental")
    st.markdown("6. Curvaturas")
    st.markdown("7. Superfícies Mínimas")
    st.markdown("8. Variação da Área")
    
    st.divider()

    st.caption(
        "Plataforma educacional para visualização "
        "e exploração de conceitos de Geometria Diferencial."
    )

# ============================================================
# CABEÇALHO
# ============================================================

st.markdown(
    """
    <div class="main-title">
        Geometria Diferencial Interativa
    </div>

    <div class="subtitle">
        Uma plataforma para visualizar, explorar e compreender
        conceitos de Geometria Diferencial
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# APRESENTAÇÃO
# ============================================================

st.markdown(
    """
    <div class="presentation-box">

    A <b>Geometria Diferencial</b> estuda objetos geométricos
    utilizando ferramentas do Cálculo Diferencial e Integral.

    Nesta plataforma, conceitos matemáticos são apresentados
    juntamente com visualizações computacionais interativas.

    O estudante poderá modificar parâmetros, escolher pontos,
    alterar superfícies e observar, em tempo real, como diferentes
    quantidades geométricas se comportam.

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OBJETIVO
# ============================================================

st.markdown(
    '<div class="section-title">Objetivo da plataforma</div>',
    unsafe_allow_html=True,
)

st.write(
    """
    O objetivo desta plataforma é proporcionar uma abordagem
    visual, interativa e computacional para o estudo da
    Geometria Diferencial.

    Ao longo dos módulos, o estudante poderá relacionar
    definições e resultados teóricos com representações
    gráficas tridimensionais e experimentos computacionais.
    """
)


# ============================================================
# TRÊS PILARES
# ============================================================

st.markdown(
    '<div class="section-title">Como funciona a plataforma?</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        📖 Teoria
        </div>

        Apresentação das definições, conceitos,
        propriedades e resultados fundamentais
        da Geometria Diferencial.

        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        🧮 Cálculos
        </div>

        Cálculo de quantidades geométricas como
        vetores tangentes, vetores normais,
        formas fundamentais e curvaturas.

        </div>
        """,
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        🌐 Visualização
        </div>

        Exploração interativa de curvas e
        superfícies tridimensionais por meio
        de gráficos computacionais.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PERCURSO MATEMÁTICO
# ============================================================

st.markdown(
    '<div class="section-title">Percurso matemático</div>',
    unsafe_allow_html=True,
)

st.write(
    """
    A organização dos conteúdos segue uma progressão natural
    dos conceitos fundamentais da Geometria Diferencial.
    """
)


# ============================================================
# MÓDULO 1
# ============================================================

with st.expander("1. Curvas em R³"):

    st.write(
        """
        Neste módulo serão estudadas curvas parametrizadas
        no espaço tridimensional.
        """
    )

    st.markdown(
        """
        **Principais conceitos:**

        - curvas parametrizadas;
        - curvas regulares;
        - vetor tangente;
        - comprimento de arco;
        - triedro de Frenet;
        - curvatura;
        - torção.
        """
    )


# ============================================================
# MÓDULO 2
# ============================================================

with st.expander("2. Superfícies Regulares"):

    st.write(
        """
        Introdução às superfícies parametrizadas e às
        condições de regularidade.
        """
    )

    st.markdown(
        """
        **Principais conceitos:**

        - parametrização de superfícies;
        - superfícies regulares;
        - curvas coordenadas;
        - derivadas parciais da parametrização.
        """
    )


# ============================================================
# MÓDULO 3
# ============================================================

with st.expander("3. Plano Tangente e Vetor Normal"):

    st.write(
        """
        Estudo da geometria local de uma superfície
        em torno de um ponto.
        """
    )

    st.markdown(
        """
        **Principais conceitos:**

        - vetores tangentes;
        - plano tangente;
        - produto vetorial;
        - vetor normal unitário;
        - campo normal.
        """
    )


# ============================================================
# MÓDULO 4
# ============================================================

with st.expander("4. Primeira Forma Fundamental"):

    st.write(
        """
        Estudo das propriedades métricas das superfícies.
        """
    )

    st.markdown(
        """
        **Principais conceitos:**

        - coeficientes E, F e G;
        - comprimento de curvas;
        - ângulo entre curvas;
        - área de superfícies.
        """
    )


# ============================================================
# MÓDULO 5
# ============================================================

with st.expander("5. Segunda Forma Fundamental"):

    st.write(
        """
        Introdução ao estudo da curvatura das superfícies.
        """
    )

    st.markdown(
        """
        **Principais conceitos:**

        - aplicação normal de Gauss;
        - operador de forma;
        - coeficientes e, f e g;
        - curvaturas principais.
        """
    )


# ============================================================
# MÓDULO 6
# ============================================================

with st.expander("6. Curvaturas"):

    st.write(
        """
        Estudo das principais medidas de curvatura
        associadas a uma superfície.
        """
    )

    st.markdown(
        r"""
        **Principais conceitos:**

        - curvaturas principais \(k_1\) e \(k_2\);
        - curvatura gaussiana \(K\);
        - curvatura média \(H\);
        - classificação dos pontos de uma superfície.
        """
    )


# ============================================================
# MÓDULO 7
# ============================================================

with st.expander("7. Superfícies Mínimas"):

    st.write(
        """
        Estudo das superfícies cuja curvatura média
        é identicamente nula.
        """
    )

    st.markdown(
        r"""
        **Principais conceitos:**

        - definição de superfície mínima;
        - condição \(H=0\);
        - plano;
        - catenoide;
        - helicoide;
        - superfície de Enneper;
        - superfície de Scherk.
        """
    )


# ============================================================
# MÓDULO 8
# ============================================================

with st.expander("8. Variação da Área"):

    st.write(
        """
        Estudo do comportamento da área de uma superfície
        submetida a deformações normais.
        """
    )

    st.markdown(
        r"""
        **Principais conceitos:**

        - variações normais;
        - família de superfícies \(X_t\);
        - função de variação \(h(u,v)\);
        - primeira variação da área;
        - relação entre superfícies mínimas e pontos críticos
          do funcional área.
        """
    )


# ============================================================
# EXPERIÊNCIA INTERATIVA
# ============================================================

st.markdown(
    '<div class="section-title">O que o estudante poderá fazer?</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    Ao longo da plataforma será possível:

    - escolher diferentes curvas e superfícies;
    - alterar parâmetros das parametrizações;
    - modificar os domínios dos parâmetros;
    - selecionar pontos sobre as superfícies;
    - visualizar vetores tangentes;
    - visualizar o plano tangente;
    - visualizar o vetor normal;
    - calcular as formas fundamentais;
    - calcular a curvatura média;
    - calcular a curvatura gaussiana;
    - visualizar a distribuição das curvaturas sobre a superfície;
    - estudar superfícies mínimas;
    - aplicar variações normais;
    - modificar a função \(h(u,v)\);
    - observar a deformação da superfície em tempo real.
    """
)


# ============================================================
# OBSERVAÇÃO FINAL
# ============================================================

st.info(
    """
    Utilize o menu lateral para acompanhar a organização dos conteúdos.

    À medida que os módulos forem desenvolvidos, novas páginas
    interativas serão acrescentadas à plataforma.
    """
)


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align: center; color: #777777;">

    <b>Geometria Diferencial Interativa</b>

    Plataforma educacional para o estudo e visualização
    de conceitos de Geometria Diferencial. 
    Esta plataforma está sendo desenvolvida pela Profª Drª Jackeline Conrado e a mestranda Chryslane Louzada

    </div>
    """,
    unsafe_allow_html=True,
)
