import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title='Plano Tangente, Vetor Normal e Orientação', page_icon='📐', layout='wide')

DIGITS = 2
TOL = 1e-9

with st.sidebar:
    st.title('📐 Geometria Diferencial')
    st.page_link('app.py', label='🏠 Início')
    st.page_link('pages/01_Curvas.py', label='1. Curvas')
    st.page_link('pages/02_Superficies_Regulares.py', label='2. Superfícies Regulares')
    st.page_link('pages/03_Plano_Tangente_Normal.py', label='3. Plano Tangente, Normal e Orientação')
    st.page_link('pages/04_Formas_Fundamentais.py', label='4. Formas Fundamentais')
    st.page_link('pages/05_Curvaturas.py', label='5. Curvaturas')
    st.page_link('pages/06_Superficies_Minimas_Variacao_Area.py', label='6. Superfícies Mínimas e Variação da Área')


def fmt(value, digits=DIGITS):
    try:
        value = float(value)
        if not np.isfinite(value):
            return 'não definido'
        return f'{value:.{digits}f}'
    except Exception:
        return 'não definido'


def vector_latex(vector, digits=DIGITS):
    values = r',\;'.join(fmt(entry, digits) for entry in np.asarray(vector).reshape(-1))
    return rf'\left({values}\right)'


def matrix_latex(matrix, digits=DIGITS):
    rows = [' & '.join(fmt(entry, digits) for entry in row) for row in np.asarray(matrix)]
    return r'\begin{pmatrix}' + r'\\'.join(rows) + r'\end{pmatrix}'


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < TOL:
        return None
    return vector / norm


def broadcast_array(value, shape):
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        return np.full(shape, float(array))
    return np.broadcast_to(array, shape).astype(float)


def eval_expr(expr, U, V):
    allowed = {
        'u': U, 'v': V,
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
        'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
        'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
        'abs': np.abs, 'pi': np.pi,
    }
    return eval(expr, {'__builtins__': {}}, allowed)


def make_grid(umin, umax, vmin, vmax, n):
    u_values = np.linspace(umin, umax, n)
    v_values = np.linspace(vmin, vmax, n)
    return np.meshgrid(u_values, v_values, indexing='ij')


def closest_index(grid, value, axis):
    line = grid[:, 0] if axis == 0 else grid[0, :]
    return int(np.argmin(np.abs(line - value)))


SURFACE_NAMES = [
    'Plano', 'Paraboloide elíptico', 'Esfera', 'Cilindro',
    'Catenoide', 'Helicoide', 'Toro',
    'Hiperboloide de uma folha', 'Hiperboloide de duas folhas',
    'Personalizada',
]


def surface(U, V, name, a=1.0, R=1.0, r=0.35, custom=None):
    if name == 'Plano':
        return np.stack((U, V, 0 * U), axis=-1)
    if name == 'Paraboloide elíptico':
        return np.stack((U, V, U**2 + V**2), axis=-1)
    if name == 'Esfera':
        return np.stack((R*np.cos(U)*np.sin(V), R*np.sin(U)*np.sin(V), R*np.cos(V)), axis=-1)
    if name == 'Cilindro':
        return np.stack((R*np.cos(U), R*np.sin(U), V), axis=-1)
    if name == 'Catenoide':
        return np.stack((a*np.cosh(V/a)*np.cos(U), a*np.cosh(V/a)*np.sin(U), V), axis=-1)
    if name == 'Helicoide':
        return np.stack((U*np.cos(V), U*np.sin(V), a*V), axis=-1)
    if name == 'Toro':
        return np.stack(((R+r*np.cos(V))*np.cos(U), (R+r*np.cos(V))*np.sin(U), r*np.sin(V)), axis=-1)
    if name == 'Hiperboloide de uma folha':
        return np.stack((np.cosh(V)*np.cos(U), np.cosh(V)*np.sin(U), np.sinh(V)), axis=-1)
    if name == 'Hiperboloide de duas folhas':
        return np.stack((np.sinh(V)*np.cos(U), np.sinh(V)*np.sin(U), np.cosh(V)), axis=-1)
    if name == 'Personalizada':
        X = broadcast_array(eval_expr(custom['x'], U, V), U.shape)
        Y = broadcast_array(eval_expr(custom['y'], U, V), U.shape)
        Z = broadcast_array(eval_expr(custom['z'], U, V), U.shape)
        return np.stack((X, Y, Z), axis=-1)
    raise ValueError('Superfície não reconhecida.')


def analytic_derivatives(U, V, name, a=1.0, R=1.0, r=0.35):
    zeros = np.zeros_like(U)
    if name == 'Plano':
        Xu = np.stack((np.ones_like(U), zeros, zeros), axis=-1)
        Xv = np.stack((zeros, np.ones_like(U), zeros), axis=-1)
    elif name == 'Paraboloide elíptico':
        Xu = np.stack((np.ones_like(U), zeros, 2*U), axis=-1)
        Xv = np.stack((zeros, np.ones_like(U), 2*V), axis=-1)
    elif name == 'Esfera':
        Xu = np.stack((-R*np.sin(U)*np.sin(V), R*np.cos(U)*np.sin(V), zeros), axis=-1)
        Xv = np.stack((R*np.cos(U)*np.cos(V), R*np.sin(U)*np.cos(V), -R*np.sin(V)), axis=-1)
    elif name == 'Cilindro':
        Xu = np.stack((-R*np.sin(U), R*np.cos(U), zeros), axis=-1)
        Xv = np.stack((zeros, zeros, np.ones_like(U)), axis=-1)
    elif name == 'Catenoide':
        c, s = np.cosh(V/a), np.sinh(V/a)
        Xu = np.stack((-a*c*np.sin(U), a*c*np.cos(U), zeros), axis=-1)
        Xv = np.stack((s*np.cos(U), s*np.sin(U), np.ones_like(U)), axis=-1)
    elif name == 'Helicoide':
        Xu = np.stack((np.cos(V), np.sin(V), zeros), axis=-1)
        Xv = np.stack((-U*np.sin(V), U*np.cos(V), a*np.ones_like(U)), axis=-1)
    elif name == 'Toro':
        q = R + r*np.cos(V)
        Xu = np.stack((-q*np.sin(U), q*np.cos(U), zeros), axis=-1)
        Xv = np.stack((-r*np.sin(V)*np.cos(U), -r*np.sin(V)*np.sin(U), r*np.cos(V)), axis=-1)
    elif name == 'Hiperboloide de uma folha':
        Xu = np.stack((-np.cosh(V)*np.sin(U), np.cosh(V)*np.cos(U), zeros), axis=-1)
        Xv = np.stack((np.sinh(V)*np.cos(U), np.sinh(V)*np.sin(U), np.cosh(V)), axis=-1)
    elif name == 'Hiperboloide de duas folhas':
        Xu = np.stack((-np.sinh(V)*np.sin(U), np.sinh(V)*np.cos(U), zeros), axis=-1)
        Xv = np.stack((np.cosh(V)*np.cos(U), np.cosh(V)*np.sin(U), np.sinh(V)), axis=-1)
    else:
        raise ValueError('Não há derivadas analíticas cadastradas.')
    return Xu, Xv


def numerical_derivatives(X, du, dv):
    return (
        np.gradient(X, du, axis=0, edge_order=2),
        np.gradient(X, dv, axis=1, edge_order=2),
    )


def normal_data(Xu, Xv):
    cross = np.cross(Xu, Xv)
    norm = np.linalg.norm(cross, axis=-1, keepdims=True)
    N = np.divide(cross, norm, out=np.zeros_like(cross), where=norm > TOL)
    return cross, norm[..., 0], N


def surface_latex(name, custom=None):
    formulas = {
        'Plano': r'X(u,v)=(u,v,0)',
        'Paraboloide elíptico': r'X(u,v)=(u,v,u^2+v^2)',
        'Esfera': r'X(u,v)=(R\cos u\sin v,R\sin u\sin v,R\cos v)',
        'Cilindro': r'X(u,v)=(R\cos u,R\sin u,v)',
        'Catenoide': r'X(u,v)=(a\cosh(v/a)\cos u,a\cosh(v/a)\sin u,v)',
        'Helicoide': r'X(u,v)=(u\cos v,u\sin v,av)',
        'Toro': r'X(u,v)=((R+r\cos v)\cos u,(R+r\cos v)\sin u,r\sin v)',
        'Hiperboloide de uma folha': r'X(u,v)=(\cosh v\cos u,\cosh v\sin u,\sinh v)',
        'Hiperboloide de duas folhas': r'X(u,v)=(\sinh v\cos u,\sinh v\sin u,\cosh v)',
    }
    if name == 'Personalizada':
        return rf"X(u,v)=\left({custom['x']},{custom['y']},{custom['z']}\right)"
    return formulas[name]


def tangent_latex(name):
    data = {
        'Plano': (r'X_u=(1,0,0)', r'X_v=(0,1,0)', r'X_u\times X_v=(0,0,1)'),
        'Paraboloide elíptico': (r'X_u=(1,0,2u)', r'X_v=(0,1,2v)', r'X_u\times X_v=(-2u,-2v,1)'),
        'Esfera': (r'X_u=(-R\sin u\sin v,R\cos u\sin v,0)', r'X_v=(R\cos u\cos v,R\sin u\cos v,-R\sin v)', r'X_u\times X_v=-R\sin v\,X(u,v)'),
        'Cilindro': (r'X_u=(-R\sin u,R\cos u,0)', r'X_v=(0,0,1)', r'X_u\times X_v=(R\cos u,R\sin u,0)'),
        'Catenoide': (r'X_u=(-a\cosh(v/a)\sin u,a\cosh(v/a)\cos u,0)', r'X_v=(\sinh(v/a)\cos u,\sinh(v/a)\sin u,1)', r'X_u\times X_v=a\cosh(v/a)(\cos u,\sin u,-\sinh(v/a))'),
        'Helicoide': (r'X_u=(\cos v,\sin v,0)', r'X_v=(-u\sin v,u\cos v,a)', r'X_u\times X_v=(a\sin v,-a\cos v,u)'),
        'Toro': (r'X_u=(-(R+r\cos v)\sin u,(R+r\cos v)\cos u,0)', r'X_v=(-r\sin v\cos u,-r\sin v\sin u,r\cos v)', r'X_u\times X_v=r(R+r\cos v)(\cos v\cos u,\cos v\sin u,\sin v)'),
        'Hiperboloide de uma folha': (r'X_u=(-\cosh v\sin u,\cosh v\cos u,0)', r'X_v=(\sinh v\cos u,\sinh v\sin u,\cosh v)', r'X_u\times X_v=\cosh v(\cosh v\cos u,\cosh v\sin u,-\sinh v)'),
        'Hiperboloide de duas folhas': (r'X_u=(-\sinh v\sin u,\sinh v\cos u,0)', r'X_v=(\cosh v\cos u,\cosh v\sin u,\sinh v)', r'X_u\times X_v=\sinh v(\sinh v\cos u,\sinh v\sin u,-\cosh v)'),
    }
    return data.get(name)


def orientability_data(name):
    return {
        'Plano': (r'N(x,y,0)=(0,0,1)', 'O campo normal unitário é constante e contínuo em todo o plano.'),
        'Paraboloide elíptico': (r'N(u,v)=\frac{(-2u,-2v,1)}{\sqrt{4u^2+4v^2+1}}', 'O denominador nunca se anula, logo o campo é global e contínuo.'),
        'Esfera': (r'N(p)=\frac{p}{R},\qquad p\in\mathbb{S}^2(R)', 'O campo radial exterior está definido em toda a esfera. A normal produzida pela ordem X_u\times X_v na carta escolhida aponta no sentido oposto.'),
        'Cilindro': (r'N(x,y,z)=\frac{(x,y,0)}{R}', 'O campo radial exterior está definido continuamente em todo o cilindro.'),
        'Catenoide': (r'N(u,v)=\frac{(\cos u,\sin u,-\sinh(v/a))}{\cosh(v/a)}', r'Como $\cosh(v/a)>0$, a expressão define um campo normal unitário global.'),
        'Helicoide': (r'N(u,v)=\frac{(a\sin v,-a\cos v,u)}{\sqrt{a^2+u^2}}', 'Como a>0, o denominador nunca se anula.'),
        'Toro': (r'N(u,v)=(\cos v\cos u,\cos v\sin u,\sin v)', 'Para R>r>0, a expressão é periódica e compatível com as identificações dos parâmetros.'),
        'Hiperboloide de uma folha': (r'N(x,y,z)=\frac{(x,y,-z)}{\sqrt{x^2+y^2+z^2}}', 'O gradiente normalizado de F(x,y,z)=x^2+y^2-z^2 fornece uma orientação global.'),
        'Hiperboloide de duas folhas': (r'N(x,y,z)=\frac{(x,y,-z)}{\sqrt{x^2+y^2+z^2}}', 'Cada folha, e também a união das duas folhas, admite esse campo normal contínuo.'),
    }[name]


def default_domain(name):
    if name == 'Esfera':
        return 0.0, float(2*np.pi), 0.05, float(np.pi-0.05)
    if name in ['Cilindro', 'Catenoide']:
        return 0.0, float(2*np.pi), -2.0, 2.0
    if name == 'Toro':
        return 0.0, float(2*np.pi), 0.0, float(2*np.pi)
    if name == 'Helicoide':
        return -2.0, 2.0, 0.0, float(4*np.pi)
    if name == 'Hiperboloide de uma folha':
        return 0.0, float(2*np.pi), -1.5, 1.5
    if name == 'Hiperboloide de duas folhas':
        return 0.0, float(2*np.pi), 0.15, 1.5
    return -2.0, 2.0, -2.0, 2.0


def add_arrow(fig, point, vector, label, scale=0.7):
    direction = unit(vector)
    if direction is None:
        return
    endpoint = point + scale*direction
    fig.add_trace(go.Scatter3d(x=[point[0], endpoint[0]], y=[point[1], endpoint[1]], z=[point[2], endpoint[2]], mode='lines', name=label, line=dict(width=7)))
    fig.add_trace(go.Cone(x=[endpoint[0]], y=[endpoint[1]], z=[endpoint[2]], u=[direction[0]], v=[direction[1]], w=[direction[2]], sizemode='absolute', sizeref=0.18, anchor='tip', showscale=False, showlegend=False))


def tangent_plane(point, xu, xv, size=1.2, n=25):
    e1 = unit(xu)
    if e1 is None:
        return None
    xv_perp = xv - np.dot(xv, e1)*e1
    e2 = unit(xv_perp)
    if e2 is None:
        return None
    s_values = np.linspace(-size, size, n)
    t_values = np.linspace(-size, size, n)
    S, T = np.meshgrid(s_values, t_values)
    return point[None,None,:] + S[...,None]*e1[None,None,:] + T[...,None]*e2[None,None,:]


def make_plot(X, i0, j0, Xu, Xv, N, plane_size, show_surface=True, show_coord=True, show_vectors=True, show_plane=True, show_normal_line=True, show_normal_field=False):
    fig = go.Figure()
    point = X[i0, j0]
    if show_surface:
        fig.add_trace(go.Surface(x=X[...,0], y=X[...,1], z=X[...,2], opacity=0.72, colorscale='Viridis', showscale=False, name='Superfície', showlegend=True))
    fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode='markers', name='Ponto p=X(u₀,v₀)', marker=dict(size=7)))
    if show_coord:
        fig.add_trace(go.Scatter3d(x=X[:,j0,0], y=X[:,j0,1], z=X[:,j0,2], mode='lines', name='Curva u ↦ X(u,v₀)', line=dict(width=6)))
        fig.add_trace(go.Scatter3d(x=X[i0,:,0], y=X[i0,:,1], z=X[i0,:,2], mode='lines', name='Curva v ↦ X(u₀,v)', line=dict(width=6)))
    if show_plane:
        P = tangent_plane(point, Xu[i0,j0], Xv[i0,j0], plane_size)
        if P is not None:
            fig.add_trace(go.Surface(x=P[...,0], y=P[...,1], z=P[...,2], opacity=0.60, colorscale=[[0,'rgba(255,180,0,0.75)'],[1,'rgba(255,180,0,0.75)']], showscale=False, name='Plano tangente', showlegend=True))
    if show_vectors:
        add_arrow(fig, point, Xu[i0,j0], 'Xᵤ(u₀,v₀)')
        add_arrow(fig, point, Xv[i0,j0], 'Xᵥ(u₀,v₀)')
        add_arrow(fig, point, N[i0,j0], 'N(u₀,v₀)')
    if show_normal_line:
        normal = unit(N[i0,j0])
        if normal is not None:
            q1, q2 = point-plane_size*normal, point+plane_size*normal
            fig.add_trace(go.Scatter3d(x=[q1[0],q2[0]], y=[q1[1],q2[1]], z=[q1[2],q2[2]], mode='lines', name='Reta normal', line=dict(width=5,dash='dash')))
    if show_normal_field:
        su, sv = max(1,X.shape[0]//10), max(1,X.shape[1]//10)
        points = X[::su,::sv].reshape(-1,3)
        normals = N[::su,::sv].reshape(-1,3)
        valid = np.linalg.norm(normals,axis=1)>TOL
        points, normals = points[valid], normals[valid]
        fig.add_trace(go.Cone(x=points[:,0], y=points[:,1], z=points[:,2], u=normals[:,0], v=normals[:,1], w=normals[:,2], sizemode='absolute', sizeref=0.25, anchor='tail', showscale=False, name='Campo normal', showlegend=True, opacity=0.75))
    fig.update_layout(height=700, margin=dict(l=0,r=0,t=40,b=0), legend=dict(font=dict(size=14), itemsizing='constant', x=0.02, y=0.02, xanchor='left', yanchor='bottom', bgcolor='rgba(255,255,255,0.75)', bordercolor='rgba(100,100,100,0.4)', borderwidth=1), scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z', aspectmode='data'))
    return fig


st.title('3 — Plano Tangente, Vetor Normal e Orientação')
st.write('Neste módulo, analisamos a geometria local de uma superfície regular e relacionamos a parametrização com o plano tangente, a diferencial, o vetor normal unitário e a orientabilidade.')

with st.expander('Conceitos matemáticos utilizados', expanded=True):
    st.markdown('#### Plano tangente')
    st.write(r'Seja $S$ uma superfície regular e $p\in S$. O plano tangente $T_pS$ é o espaço vetorial formado pelos vetores velocidade das curvas diferenciáveis contidas em $S$ e que passam por $p$.')
    st.latex(r'T_pS=\{\alpha\prime(0)\in\mathbb{R}^3:\alpha:(-\varepsilon,\varepsilon)\to S,\ \alpha(0)=p\}')
    st.latex(r'T_pS=\operatorname{span}\{X_u(u_0,v_0),X_v(u_0,v_0)\}.')
    st.markdown('#### Diferencial da parametrização')
    st.latex(r'dX_{(u_0,v_0)}:\mathbb{R}^2\to T_pS,\qquad dX_{(u_0,v_0)}(\xi,\eta)=\xi X_u+\eta X_v.')
    st.markdown('#### Vetor normal e orientação')
    st.latex(r'N(u,v)=\frac{X_u(u,v)\times X_v(u,v)}{\|X_u(u,v)\times X_v(u,v)\|}.')
    st.write('Uma superfície é orientável quando admite um campo contínuo de vetores normais unitários definido em toda a superfície.')

with st.sidebar:
    st.header('Superfície')
    name = st.selectbox('Escolha uma superfície', SURFACE_NAMES)
    a, R, r = 1.0, 1.0, 0.35
    if name in ['Catenoide','Helicoide']:
        a = st.slider('Parâmetro a',0.2,3.0,1.0,0.1)
    if name in ['Esfera','Cilindro','Toro']:
        R = st.slider('Raio principal R',0.2,3.0,1.0,0.1)
    if name == 'Toro':
        r_max = max(0.1,min(1.2,R-0.05))
        r = st.slider('Raio menor r',0.05,float(r_max),min(0.35,r_max),0.05)
        st.caption('Usamos R>r>0 para obter o toro regular usual.')
    custom = {}
    if name == 'Personalizada':
        st.markdown('Digite as coordenadas usando `u` e `v`.')
        custom['x'] = st.text_input('x(u,v)','u')
        custom['y'] = st.text_input('y(u,v)','v')
        custom['z'] = st.text_input('z(u,v)','sin(u)*cos(v)')
    st.header('Domínio')
    du0,du1,dv0,dv1 = default_domain(name)
    umin = st.number_input('u mínimo',value=float(du0))
    umax = st.number_input('u máximo',value=float(du1))
    vmin = st.number_input('v mínimo',value=float(dv0))
    vmax = st.number_input('v máximo',value=float(dv1))
    n = st.slider('Resolução',40,160,80,10)
    st.header('Ponto no domínio')
    u0 = st.slider('Escolha u₀',float(umin),float(umax),float((umin+umax)/2))
    v0 = st.slider('Escolha v₀',float(vmin),float(vmax),float((vmin+vmax)/2))
    st.header('Elementos geométricos')
    show_surface = st.checkbox('Mostrar superfície',True)
    show_coord = st.checkbox('Mostrar curvas coordenadas',True)
    show_vectors = st.checkbox('Mostrar Xᵤ, Xᵥ e N',True)
    show_plane = st.checkbox('Mostrar plano tangente',True)
    show_normal_line = st.checkbox('Mostrar reta normal',True)
    show_normal_field = st.checkbox('Mostrar campo normal',False)
    plane_size = st.slider('Tamanho do plano tangente',0.3,3.0,1.4,0.1)

if umax<=umin or vmax<=vmin:
    st.error('O domínio precisa satisfazer u mínimo < u máximo e v mínimo < v máximo.')
    st.stop()

try:
    U,V = make_grid(umin,umax,vmin,vmax,n)
    X = surface(U,V,name,a=a,R=R,r=r,custom=custom)
    if name == 'Personalizada':
        du,dv = (umax-umin)/(n-1),(vmax-vmin)/(n-1)
        Xu,Xv = numerical_derivatives(X,du,dv)
    else:
        Xu,Xv = analytic_derivatives(U,V,name,a=a,R=R,r=r)
    cross,cross_norm,N = normal_data(Xu,Xv)
    i0,j0 = closest_index(U,u0,0),closest_index(V,v0,1)
    point,xu,xv,cp = X[i0,j0],Xu[i0,j0],Xv[i0,j0],cross[i0,j0]
    cp_norm = float(cross_norm[i0,j0])
    nvec = unit(cp)

    st.subheader('Superfície escolhida')
    st.latex(surface_latex(name,custom))
    if name == 'Esfera':
        st.info('A parametrização esférica utilizada não é regular nos polos. Por isso, o domínio padrão evita v=0 e v=π. A esfera, entretanto, é regular e orientável em todos os seus pontos.')
    elif name == 'Hiperboloide de duas folhas':
        st.info('A parametrização exibida descreve a folha superior para v>0. O ponto v=0 é excluído porque esta parametrização se torna singular nesse valor, embora o hiperboloide seja uma superfície regular.')

    left,right = st.columns([2.15,1])
    with left:
        st.plotly_chart(make_plot(X,i0,j0,Xu,Xv,N,plane_size,show_surface,show_coord,show_vectors,show_plane,show_normal_line,show_normal_field),use_container_width=True)
    with right:
        st.subheader('Valores no ponto escolhido')
        st.latex(rf'(u_0,v_0)=({fmt(U[i0,j0])},{fmt(V[i0,j0])})')
        st.latex(rf'p=X(u_0,v_0)={vector_latex(point)}')
        st.latex(rf'X_u(u_0,v_0)={vector_latex(xu)}')
        st.latex(rf'X_v(u_0,v_0)={vector_latex(xv)}')
        st.latex(rf'X_u\times X_v={vector_latex(cp)}')
        st.latex(rf'\|X_u\times X_v\|={fmt(cp_norm)}')
        if nvec is None:
            st.error('O produto vetorial se anula no ponto escolhido. Esta parametrização não é regular nesse ponto.')
        else:
            st.latex(rf'N(u_0,v_0)={vector_latex(nvec)}')
        df = pd.DataFrame({'Objeto':['p','X_u','X_v','X_u × X_v','N'],'x':[fmt(point[0]),fmt(xu[0]),fmt(xv[0]),fmt(cp[0]),fmt(nvec[0]) if nvec is not None else 'não definido'],'y':[fmt(point[1]),fmt(xu[1]),fmt(xv[1]),fmt(cp[1]),fmt(nvec[1]) if nvec is not None else 'não definido'],'z':[fmt(point[2]),fmt(xu[2]),fmt(xv[2]),fmt(cp[2]),fmt(nvec[2]) if nvec is not None else 'não definido']})
        st.dataframe(df,hide_index=True,use_container_width=True)

    tabs = st.tabs(['Plano tangente','Diferencial','Vetor normal','Orientabilidade','Fórmulas do exemplo'])
    with tabs[0]:
        st.markdown('### Plano tangente no ponto escolhido')
        st.latex(r'T_pS=\operatorname{span}\{X_u(u_0,v_0),X_v(u_0,v_0)\}')
        st.latex(r'P(s,t)=p+sX_u(u_0,v_0)+tX_v(u_0,v_0)')
        st.latex(rf'P(s,t)={vector_latex(point)}+s{vector_latex(xu)}+t{vector_latex(xv)}')
        if nvec is not None:
            st.latex(r'\langle (x,y,z)-p,N(u_0,v_0)\rangle=0')
            st.latex(rf'{fmt(nvec[0])}(x-{fmt(point[0])})+{fmt(nvec[1])}(y-{fmt(point[1])})+{fmt(nvec[2])}(z-{fmt(point[2])})=0')
    with tabs[1]:
        st.markdown('### Diferencial da parametrização')
        st.latex(r'dX_{(u_0,v_0)}(\xi,\eta)=\xi X_u(u_0,v_0)+\eta X_v(u_0,v_0)')
        jacobian = np.column_stack((xu,xv))
        st.latex(rf'[dX_{{(u_0,v_0)}}]={matrix_latex(jacobian)}')
        rank = int(np.linalg.matrix_rank(jacobian,tol=TOL))
        st.latex(rf'\operatorname{{posto}}(dX_{{(u_0,v_0)}})={rank}')
        if rank==2:
            st.success('A diferencial é injetora no ponto escolhido e sua imagem é o plano tangente.')
        else:
            st.error('A diferencial não possui posto 2 no ponto escolhido.')
    with tabs[2]:
        st.markdown('### Vetor normal unitário')
        st.latex(r'N(u_0,v_0)=\frac{X_u(u_0,v_0)\times X_v(u_0,v_0)}{\|X_u(u_0,v_0)\times X_v(u_0,v_0)\|}')
        if nvec is not None:
            st.latex(rf'N(u_0,v_0)={vector_latex(nvec)}')
            st.latex(rf'\langle N,X_u\rangle={fmt(np.dot(nvec,xu))},\qquad\langle N,X_v\rangle={fmt(np.dot(nvec,xv))}')
            st.latex(rf'\|N\|={fmt(np.linalg.norm(nvec))}')
    with tabs[3]:
        st.markdown('### Orientabilidade')
        if name != 'Personalizada':
            field,reason = orientability_data(name)
            st.success(f'A superfície **{name} é orientável**.')
            st.latex(field)
            st.write(reason)
            st.write('A existência desse campo em toda a superfície fornece uma escolha contínua de uma das duas direções normais.')
        else:
            st.warning('A orientabilidade é uma propriedade global e não pode ser decidida, em geral, apenas por uma malha numérica ou por uma única parametrização fornecida pelo usuário.')
            st.write('Quando X_u×X_v não se anula no domínio e a parametrização é global, injetora e compatível com as identificações do domínio, a normal normalizada fornece uma orientação. Uma verificação apenas local não prova automaticamente a orientabilidade global.')
            min_norm = float(np.nanmin(cross_norm))
            st.latex(rf'\min_{{(u,v)\text{{ na malha}}}}\|X_u\times X_v\|\approx {fmt(min_norm)}')
            if min_norm>1e-5:
                st.info('Na malha exibida, o produto vetorial não se anulou. Isso indica regularidade local no domínio testado, mas não constitui uma prova de orientabilidade global.')
            else:
                st.error('A malha contém pontos em que o produto vetorial é nulo ou numericamente muito pequeno.')
    with tabs[4]:
        st.markdown('### Fórmulas do exemplo')
        if name == 'Personalizada':
            st.write('Para a superfície personalizada, as derivadas são aproximadas numericamente. As fórmulas exatas devem ser calculadas a partir das expressões inseridas.')
        else:
            xu_formula,xv_formula,cross_formula = tangent_latex(name)
            st.latex(surface_latex(name))
            st.latex(xu_formula)
            st.latex(xv_formula)
            st.latex(cross_formula)
except Exception as error:
    st.error('Não foi possível gerar a superfície. Verifique as expressões e o domínio escolhidos.')
    st.exception(error)
