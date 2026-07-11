import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Curvas planas e espaciais", page_icon="📈", layout="wide")
EPS = 1e-10

# ============================================================
# MENU DA PLATAFORMA
# ============================================================
with st.sidebar:
    st.title("📐 Geometria Diferencial")
    st.page_link("app.py", label="🏠 Início")
    st.page_link("pages/01_Curvas.py", label="1. Curvas")
    st.page_link("pages/02_Superficies_Regulares.py", label="2. Superfícies Regulares")
    st.page_link("pages/03_Plano_Tangente_Normal.py", label="3. Plano Tangente e Vetor Normal")
    st.page_link("pages/04_Formas_Fundamentais.py", label="4. Formas Fundamentais")
    st.page_link("pages/05_Curvaturas.py", label="5. Curvaturas")
    st.page_link("pages/06_Superficies_Minimas_Variacao_Area.py", label="6. Superfícies Mínimas e Variação da Área")

# ============================================================
# UTILITÁRIOS
# ============================================================
def fmt(x: float, digits: int = 5) -> str:
    try:
        v = float(x)
        return f"{v:.{digits}f}" if np.isfinite(v) else "não definido"
    except Exception:
        return "não definido"


def latex_scalar(x: float, digits: int = 5) -> str:
    text = fmt(x, digits)
    return r"\text{não definido}" if text == "não definido" else text


def vector_latex(v: np.ndarray, digits: int = 5) -> str:
    return r"\left(" + r",\;".join(fmt(x, digits) for x in np.asarray(v).reshape(-1)) + r"\right)"


def safe_unit(v: np.ndarray) -> Optional[np.ndarray]:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return None if (not np.isfinite(n) or n < EPS) else v / n


def cumulative_trapezoid_local(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y, x = np.asarray(y, float), np.asarray(x, float)
    if len(x) != len(y):
        raise ValueError("x e y precisam ter o mesmo número de pontos.")
    if len(x) < 2:
        return np.array([], dtype=float)
    return np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))


def as_array(value, grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    return np.full_like(grid, float(arr)) if arr.ndim == 0 else arr


def eval_expr(expr: str, variable: str, grid: np.ndarray) -> np.ndarray:
    allowed = {
        variable: grid, "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
        "pi": np.pi, "e": np.e, "where": np.where,
        "minimum": np.minimum, "maximum": np.maximum,
        "cbrt": np.cbrt,
    }
    return as_array(eval(expr, {"__builtins__": {}}, allowed), grid)


def numerical_derivatives(values: np.ndarray, parameter: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d1 = np.gradient(values, parameter, axis=0, edge_order=2)
    d2 = np.gradient(d1, parameter, axis=0, edge_order=2)
    d3 = np.gradient(d2, parameter, axis=0, edge_order=2)
    return d1, d2, d3


def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.all(np.isfinite(arr), axis=1) if arr.ndim > 1 else np.isfinite(arr)
    return mask


def planar_curvature(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    den = np.linalg.norm(d1, axis=1) ** 3
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > EPS)


def spatial_curvature(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(np.cross(d1, d2), axis=1)
    den = np.linalg.norm(d1, axis=1) ** 3
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > EPS)


def spatial_torsion(d1: np.ndarray, d2: np.ndarray, d3: np.ndarray) -> np.ndarray:
    c = np.cross(d1, d2)
    num = np.einsum("ij,ij->i", c, d3)
    den = np.linalg.norm(c, axis=1) ** 2
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > EPS)


@dataclass
class CurveData:
    parameter: np.ndarray
    alpha: np.ndarray
    alpha1: np.ndarray
    alpha2: np.ndarray
    alpha3: np.ndarray
    latex: str
    note: str = ""
    numerical: bool = False

# ============================================================
# EXEMPLOS DO CAPÍTULO E EXEMPLOS COMPLEMENTARES
# ============================================================
def planar_curve(t: np.ndarray, name: str, params: Dict[str, float], custom=None) -> CurveData:
    z = np.zeros_like(t)
    if name == "Reta y=2x (Ex. 2.2.1 e 2.2.4)":
        a = np.column_stack((t, 2*t)); d1=np.tile([1.,2.],(len(t),1)); d2=np.zeros_like(a); d3=np.zeros_like(a)
        return CurveData(t,a,d1,d2,d3,r"\alpha(t)=(t,2t)","Curva regular e simples.")
    if name == "Circunferência (Ex. 2.1.1 e 2.1.10)":
        r=params["r"]; xc=params.get("xc",0.); yc=params.get("yc",0.)
        a=np.column_stack((xc+r*np.cos(t),yc+r*np.sin(t)))
        d1=np.column_stack((-r*np.sin(t),r*np.cos(t))); d2=np.column_stack((-r*np.cos(t),-r*np.sin(t))); d3=np.column_stack((r*np.sin(t),-r*np.cos(t)))
        return CurveData(t,a,d1,d2,d3,rf"\alpha(t)=({xc:.3g}+{r:.3g}\cos t,{yc:.3g}+{r:.3g}\sin t)","Curvatura constante e curva regular.")
    if name == "Parábola (Ex. 2.1.2 e 2.1.14)":
        a0=params.get("a",1.); a=np.column_stack((t,a0*t**2)); d1=np.column_stack((np.ones_like(t),2*a0*t)); d2=np.column_stack((z,np.full_like(t,2*a0))); d3=np.zeros_like(a)
        return CurveData(t,a,d1,d2,d3,rf"\alpha(t)=(t,{a0:.3g}t^2)","Curva regular e simples.")
    if name == "Exponencial (Ex. 2.1.9)":
        a=np.column_stack((np.exp(t),t)); d1=np.column_stack((np.exp(t),np.ones_like(t))); d2=np.column_stack((np.exp(t),z)); d3=np.column_stack((np.exp(t),z))
        return CurveData(t,a,d1,d2,d3,r"\alpha(t)=(e^t,t)","No instante t=0, a reta tangente é y=x-1.")
    if name == "Senoide (Ex. 2.1.11)":
        a=np.column_stack((t,np.sin(t))); d1=np.column_stack((np.ones_like(t),np.cos(t))); d2=np.column_stack((z,-np.sin(t))); d3=np.column_stack((z,-np.cos(t)))
        return CurveData(t,a,d1,d2,d3,r"\alpha(t)=(t,\sin t)","No instante t=0, a reta tangente é y=x.")
    if name == "Catenária (Ex. 2.1.12)":
        a0=params["a"]; a=np.column_stack((t,a0*np.cosh(t/a0))); d1=np.column_stack((np.ones_like(t),np.sinh(t/a0))); d2=np.column_stack((z,np.cosh(t/a0)/a0)); d3=np.column_stack((z,np.sinh(t/a0)/a0**2))
        return CurveData(t,a,d1,d2,d3,rf"\alpha(t)=\left(t,{a0:.3g}\cosh(t/{a0:.3g})\right)","No instante t=0, a reta tangente é horizontal.")
    if name == "Cúspide singular (Ex. 2.1.13)":
        a=np.column_stack((t**3,t**2)); d1=np.column_stack((3*t**2,2*t)); d2=np.column_stack((6*t,2*np.ones_like(t))); d3=np.column_stack((6*np.ones_like(t),z))
        return CurveData(t,a,d1,d2,d3,r"\alpha(t)=(t^3,t^2)","A curva não é regular em t=0.")
    if name == "Auto-interseção (Ex. 2.1.15)":
        a=np.column_stack((np.sin(t),np.sin(t)*np.cos(t)))
        d1=np.column_stack((np.cos(t),np.cos(2*t))); d2=np.column_stack((-np.sin(t),-2*np.sin(2*t))); d3=np.column_stack((-np.cos(t),-4*np.cos(2*t)))
        return CurveData(t,a,d1,d2,d3,r"\alpha(t)=(\sin t,\sin t\cos t)","Temos α(0)=α(π), mas as retas tangentes nesses instantes são distintas.")
    if name == "Não diferenciável: valor absoluto (Ex. 2.1.5–2.1.6)":
        a=np.column_stack((np.abs(t),t)); d1,d2,d3=numerical_derivatives(a,t)
        return CurveData(t,a,d1,d2,d3,r"\alpha(t)=(|t|,t)","Não é diferenciável em t=0; portanto, não é uma curva parametrizada diferenciável segundo a convenção do capítulo.",True)
    if name == "Não diferenciável: raiz cúbica (Ex. 2.1.7)":
        a=np.column_stack((np.cbrt(t),t)); d1,d2,d3=numerical_derivatives(a,t)
        return CurveData(t,a,d1,d2,d3,r"\alpha(t)=(t^{1/3},t)","Não é diferenciável em t=0.",True)
    if name == "Diferenciável, mas não C¹ (Ex. 2.1.8)":
        f=np.where(np.abs(t)>EPS,t**2*np.sin(1/t),0.0); a=np.column_stack((t,f)); d1,d2,d3=numerical_derivatives(a,t)
        return CurveData(t,a,d1,d2,d3,r"\alpha(t)=\left(t,t^2\sin(1/t)\right),\quad \alpha(0)=(0,0)","A segunda coordenada é diferenciável em 0, mas sua derivada não é contínua em 0.",True)
    if name == "Personalizada":
        x=eval_expr(custom["x"],"t",t); y=eval_expr(custom["y"],"t",t); a=np.column_stack((x,y)); d1,d2,d3=numerical_derivatives(a,t)
        return CurveData(t,a,d1,d2,d3,rf"\alpha(t)=\left({custom['x']},{custom['y']}\right)","Derivadas aproximadas numericamente.",True)
    raise ValueError("Curva plana desconhecida.")


def spatial_curve(t: np.ndarray, name: str, params: Dict[str,float], custom=None) -> CurveData:
    z=np.zeros_like(t)
    if name == "Reta espacial (Ex. 2.1.3)":
        p=np.array([params["x0"],params["y0"],params["z0"]]); v=np.array([params["vx"],params["vy"],params["vz"]])
        a=p+t[:,None]*v; d1=np.tile(v,(len(t),1)); d2=np.zeros_like(a); d3=np.zeros_like(a)
        return CurveData(t,a,d1,d2,d3,rf"\alpha(t)={vector_latex(p,3)}+t{vector_latex(v,3)}","Curvatura nula.")
    if name == "Hélice circular (Ex. 2.1.4 e 2.2.3)":
        A=params["a"]; b=params["b"]; a=np.column_stack((A*np.cos(t),A*np.sin(t),b*t)); d1=np.column_stack((-A*np.sin(t),A*np.cos(t),np.full_like(t,b))); d2=np.column_stack((-A*np.cos(t),-A*np.sin(t),z)); d3=np.column_stack((A*np.sin(t),-A*np.cos(t),z))
        return CurveData(t,a,d1,d2,d3,rf"\alpha(t)=({A:.3g}\cos t,{A:.3g}\sin t,{b:.3g}t)",rf"Velocidade constante $\sqrt{{a^2+b^2}}={math.sqrt(A*A+b*b):.4f}$.")
    if name == "Parábola espacial":
        c=params["c"]; a=np.column_stack((t,t**2,c*t)); d1=np.column_stack((np.ones_like(t),2*t,np.full_like(t,c))); d2=np.column_stack((z,2*np.ones_like(t),z)); d3=np.zeros_like(a)
        return CurveData(t,a,d1,d2,d3,rf"\alpha(t)=(t,t^2,{c:.3g}t)")
    if name == "Cúbica torcida":
        a=np.column_stack((t,t**2,t**3)); d1=np.column_stack((np.ones_like(t),2*t,3*t**2)); d2=np.column_stack((z,2*np.ones_like(t),6*t)); d3=np.column_stack((z,z,6*np.ones_like(t)))
        return CurveData(t,a,d1,d2,d3,r"\alpha(t)=(t,t^2,t^3)")
    if name == "Curva toroidal":
        R=params["R"]; r=params["r"]; m=params["m"]; n=params["n"]; q=R+r*np.cos(n*t); a=np.column_stack((q*np.cos(m*t),q*np.sin(m*t),r*np.sin(n*t))); d1,d2,d3=numerical_derivatives(a,t)
        return CurveData(t,a,d1,d2,d3,r"\alpha(t)=((R+r\cos nt)\cos mt,(R+r\cos nt)\sin mt,r\sin nt)","Derivadas aproximadas numericamente.",True)
    if name == "Curva de Viviani":
        A=params["a"]; a=np.column_stack((A*(1+np.cos(t)),A*np.sin(t),2*A*np.sin(t/2))); d1,d2,d3=numerical_derivatives(a,t)
        return CurveData(t,a,d1,d2,d3,rf"\alpha(t)=({A:.3g}(1+\cos t),{A:.3g}\sin t,{2*A:.3g}\sin(t/2))","Derivadas aproximadas numericamente.",True)
    if name == "Personalizada":
        a=np.column_stack((eval_expr(custom["x"],"t",t),eval_expr(custom["y"],"t",t),eval_expr(custom["z"],"t",t))); d1,d2,d3=numerical_derivatives(a,t)
        return CurveData(t,a,d1,d2,d3,rf"\alpha(t)=({custom['x']},{custom['y']},{custom['z']})","Derivadas aproximadas numericamente.",True)
    raise ValueError("Curva espacial desconhecida.")

# ============================================================
# MOVIMENTOS RÍGIDOS
# ============================================================
def rotation2(theta: float) -> np.ndarray:
    c,s=np.cos(theta),np.sin(theta); return np.array([[c,-s],[s,c]],float)


def rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    u=safe_unit(axis)
    if u is None: raise ValueError("O eixo de rotação não pode ser nulo.")
    x,y,z=u; K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float); I=np.eye(3)
    return I*np.cos(theta)+(1-np.cos(theta))*np.outer(u,u)+np.sin(theta)*K


def rigid_controls_2d(key: str) -> Tuple[np.ndarray,np.ndarray,str]:
    with st.sidebar:
        st.subheader("Movimento rígido em ℝ²")
        typ=st.selectbox("Tipo",["Identidade","Translação","Rotação","Reflexão","Reflexão deslizante"],key=f"{key}_rig2_type")
        Q=np.eye(2); a=np.zeros(2); desc=r"F(p)=p"
        if typ=="Translação":
            a=np.array([st.number_input("a₁",value=1.0,key=f"{key}_tx"),st.number_input("a₂",value=0.5,key=f"{key}_ty")]); desc=rf"F(p)=p+{vector_latex(a,2)}"
        elif typ=="Rotação":
            deg=st.slider("Ângulo (graus)",-180,180,45,5,key=f"{key}_rot"); c=np.array([st.number_input("Centro c₁",value=0.0,key=f"{key}_cx"),st.number_input("Centro c₂",value=0.0,key=f"{key}_cy")]); Q=rotation2(math.radians(deg)); a=c-Q@c; desc=rf"F(p)=R_{{{deg}^\circ}}(p-c)+c"
        elif typ=="Reflexão":
            line=st.selectbox("Eixo de reflexão",["eixo x","eixo y","reta y=x","reta pela origem com ângulo θ"],key=f"{key}_refl")
            if line=="eixo x": Q=np.diag([1.,-1.])
            elif line=="eixo y": Q=np.diag([-1.,1.])
            elif line=="reta y=x": Q=np.array([[0.,1.],[1.,0.]])
            else:
                deg=st.slider("Ângulo θ da reta",-180,180,30,5,key=f"{key}_reflang"); u=np.array([np.cos(np.radians(deg)),np.sin(np.radians(deg))]); Q=2*np.outer(u,u)-np.eye(2)
            desc=r"F(p)=Mp,\quad M^TM=I,\ \det M=-1"
        elif typ=="Reflexão deslizante":
            deg=st.slider("Ângulo do eixo",-180,180,0,5,key=f"{key}_glang"); dist=st.slider("Deslizamento",-5.0,5.0,1.0,0.1,key=f"{key}_gld"); u=np.array([np.cos(np.radians(deg)),np.sin(np.radians(deg))]); Q=2*np.outer(u,u)-np.eye(2); a=dist*u; desc=r"F(p)=Mp+a\quad\text{(reflexão seguida de translação paralela ao eixo)}"
    return Q,a,desc


def rigid_controls_3d(key: str) -> Tuple[np.ndarray,np.ndarray,str]:
    with st.sidebar:
        st.subheader("Movimento rígido em ℝ³")
        typ=st.selectbox("Tipo",["Identidade","Translação","Rotação em torno de eixo","Reflexão em plano","Movimento helicoidal","Rotoreflexão"],key=f"{key}_rig3_type")
        Q=np.eye(3); a=np.zeros(3); desc=r"F(p)=p"
        if typ=="Translação":
            a=np.array([st.number_input("a₁",value=1.,key=f"{key}_tx"),st.number_input("a₂",value=.5,key=f"{key}_ty"),st.number_input("a₃",value=.25,key=f"{key}_tz")]); desc=rf"F(p)=p+{vector_latex(a,2)}"
        elif typ in {"Rotação em torno de eixo","Movimento helicoidal","Rotoreflexão"}:
            axis=np.array([st.number_input("Eixo: x",value=0.,key=f"{key}_ax"),st.number_input("Eixo: y",value=0.,key=f"{key}_ay"),st.number_input("Eixo: z",value=1.,key=f"{key}_az")]); deg=st.slider("Ângulo (graus)",-180,180,45,5,key=f"{key}_ang"); Q=rodrigues(axis,np.radians(deg)); u=safe_unit(axis)
            if typ=="Movimento helicoidal":
                h=st.slider("Translação ao longo do eixo",-5.,5.,1.,.1,key=f"{key}_pitch"); a=h*u; desc=r"F(p)=R_\theta p+h\,u"
            elif typ=="Rotoreflexão":
                H=np.eye(3)-2*np.outer(u,u); Q=H@Q; desc=r"F(p)=H\,R_\theta p\quad\text{(rotação e reflexão no plano perpendicular ao eixo)}"
            else: desc=r"F(p)=R_\theta p"
        elif typ=="Reflexão em plano":
            n=np.array([st.number_input("Normal: x",value=0.,key=f"{key}_nx"),st.number_input("Normal: y",value=0.,key=f"{key}_ny"),st.number_input("Normal: z",value=1.,key=f"{key}_nz")]); u=safe_unit(n)
            if u is None: raise ValueError("A normal do plano não pode ser nula.")
            Q=np.eye(3)-2*np.outer(u,u); desc=r"F(p)=Mp,\quad M=I-2nn^T"
    return Q,a,desc


def transform_points(points: np.ndarray,Q: np.ndarray,a: np.ndarray)->np.ndarray:
    return points@Q.T+a

# ============================================================
# GRÁFICOS
# ============================================================
def add_vec2(fig,p,v,name,scale=1.):
    if v is None:return
    q=p+scale*v; fig.add_trace(go.Scatter(x=[p[0],q[0]],y=[p[1],q[1]],mode="lines+markers",name=name,line=dict(width=4),marker=dict(size=[0,9],symbol=["circle","triangle-up"])))


def add_vec3(fig,p,v,name,scale=1.):
    if v is None:return
    q=p+scale*v; fig.add_trace(go.Scatter3d(x=[p[0],q[0]],y=[p[1],q[1]],z=[p[2],q[2]],mode="lines",name=name,line=dict(width=7)))
    fig.add_trace(go.Cone(x=[q[0]],y=[q[1]],z=[q[2]],u=[v[0]],v=[v[1]],w=[v[2]],anchor="tip",sizemode="absolute",sizeref=max(.08,.18*scale),showscale=False,showlegend=False))


def plot2(alpha,p,T=None,N=None,velocity=None,center=None,radius=None,show=(True,True,False,True,False,True),scale=1.,title=""):
    fig=go.Figure(); fig.add_trace(go.Scatter(x=alpha[:,0],y=alpha[:,1],mode="lines",name="Curva",line=dict(width=5))); fig.add_trace(go.Scatter(x=[p[0]],y=[p[1]],mode="markers",name="Ponto selecionado",marker=dict(size=11)))
    sv,sT,sN,lt,ln,sc=show
    if sv:add_vec2(fig,p,velocity,"Vetor velocidade",scale)
    if sT:add_vec2(fig,p,T,"T: tangente unitário",scale)
    if sN:add_vec2(fig,p,N,"N: normal unitário",scale)
    ext=max(np.ptp(alpha[:,0]),np.ptp(alpha[:,1]),1.)
    for flag,d,nm in [(lt,T,"Reta tangente"),(ln,N,"Reta normal")]:
        if flag and d is not None:
            d=safe_unit(d); q1=p-.6*ext*d; q2=p+.6*ext*d; fig.add_trace(go.Scatter(x=[q1[0],q2[0]],y=[q1[1],q2[1]],mode="lines",name=nm,line=dict(dash="dash",width=2)))
    if sc and center is not None and radius is not None and np.isfinite(radius) and radius<100*ext:
        q=np.linspace(0,2*np.pi,250); c=center+radius*np.column_stack((np.cos(q),np.sin(q))); fig.add_trace(go.Scatter(x=c[:,0],y=c[:,1],mode="lines",name="Círculo osculador",line=dict(dash="dot",width=3))); fig.add_trace(go.Scatter(x=[center[0]],y=[center[1]],mode="markers",name="Centro de curvatura",marker=dict(symbol="x",size=9)))
    fig.update_layout(height=620,title=title,xaxis_title="x",yaxis_title="y",margin=dict(l=0,r=0,t=45,b=0),legend=dict(x=.02,y=.02,xanchor="left",yanchor="bottom",bgcolor="rgba(255,255,255,.8)")); fig.update_yaxes(scaleanchor="x",scaleratio=1); return fig


def plot3(alpha,p,T=None,N=None,B=None,scale=1.,title=""):
    fig=go.Figure(); fig.add_trace(go.Scatter3d(x=alpha[:,0],y=alpha[:,1],z=alpha[:,2],mode="lines",name="Curva",line=dict(width=6))); fig.add_trace(go.Scatter3d(x=[p[0]],y=[p[1]],z=[p[2]],mode="markers",name="Ponto selecionado",marker=dict(size=6)))
    add_vec3(fig,p,T,"T: tangente",scale); add_vec3(fig,p,N,"N: normal principal",scale); add_vec3(fig,p,B,"B: binormal",scale)
    fig.update_layout(height=650,title=title,margin=dict(l=0,r=0,t=45,b=0),legend=dict(x=.02,y=.02,xanchor="left",yanchor="bottom",bgcolor="rgba(255,255,255,.8)"),scene=dict(xaxis_title="x",yaxis_title="y",zaxis_title="z",aspectmode="data")); return fig

# ============================================================
# CONCEITOS COMUNS
# ============================================================
def render_fundamental_concepts():
    with st.expander("📘 Conceitos fundamentais e convenção dos parâmetros",expanded=False):
        st.latex(r"\alpha:I\subset\mathbb R\longrightarrow\mathbb R^n,\qquad \alpha(t)=(x_1(t),\ldots,x_n(t))")
        st.markdown(r"""
O parâmetro **$t$** será usado para uma parametrização arbitrária. Para uma mudança de parâmetro geral, escreveremos $u\in J$ e $\beta(u)=\alpha(\varphi(u))$. Reservaremos o símbolo **$s$ exclusivamente para o comprimento de arco**. Assim, quando aparece $\alpha(s)$, entende-se que $\|\alpha'(s)\|=1$.
""")
        c1,c2=st.columns(2)
        with c1:
            st.latex(r"\alpha(I)=\{\alpha(t):t\in I\}")
            st.latex(r"\alpha'(t)=(x_1'(t),\ldots,x_n'(t))")
            st.latex(r"\alpha'(t)\ne0\quad\Longleftrightarrow\quad\text{regularidade em }t")
        with c2:
            st.latex(r"\beta(u)=\alpha(\varphi(u)),\qquad \varphi:J\to I")
            st.latex(r"\beta'(u)=\alpha'(\varphi(u))\varphi'(u)")
            st.latex(r"s(t)=\int_{t_0}^{t}\|\alpha'(v)\|\,dv")

# ============================================================
# ANÁLISE DE CURVAS
# ============================================================
def render_planar_analysis():
    st.header(r"Análise de curvas planas em $\mathbb R^2$")
    names=["Reta y=2x (Ex. 2.2.1 e 2.2.4)","Circunferência (Ex. 2.1.1 e 2.1.10)","Parábola (Ex. 2.1.2 e 2.1.14)","Exponencial (Ex. 2.1.9)","Senoide (Ex. 2.1.11)","Catenária (Ex. 2.1.12)","Cúspide singular (Ex. 2.1.13)","Auto-interseção (Ex. 2.1.15)","Não diferenciável: valor absoluto (Ex. 2.1.5–2.1.6)","Não diferenciável: raiz cúbica (Ex. 2.1.7)","Diferenciável, mas não C¹ (Ex. 2.1.8)","Personalizada"]
    with st.sidebar:
        st.subheader("Curva plana"); name=st.selectbox("Exemplo",names,key="pa_name"); params={}; custom=None
        if "Circunferência" in name: params["r"]=st.slider("Raio r",.2,5.,1.,.1,key="pa_r"); params["xc"]=0.;params["yc"]=0.
        if "Parábola" in name: params["a"]=st.slider("Coeficiente a",.2,3.,1.,.1,key="pa_a")
        if "Catenária" in name: params["a"]=st.slider("Parâmetro a",.2,3.,1.,.1,key="pa_cat")
        if name=="Personalizada": custom={"x":st.text_input("x(t)","cos(t)",key="pa_x"),"y":st.text_input("y(t)","sin(t)",key="pa_y")}
        default=(-3.,3.)
        if "Circunferência" in name or "Auto-interseção" in name: default=(0.,2*np.pi)
        if "valor absoluto" in name or "raiz cúbica" in name or "não C¹" in name: default=(-1.,1.)
        tmin=st.number_input("t mínimo",value=float(default[0]),key=f"pa_tmin_{name}"); tmax=st.number_input("t máximo",value=float(default[1]),key=f"pa_tmax_{name}"); n=st.slider("Resolução",300,1600,800,100,key="pa_n"); t0=st.slider("Ponto t₀",float(tmin),float(tmax),float((tmin+tmax)/2),key=f"pa_t0_{name}") if tmax>tmin else tmin
        st.subheader("Objetos geométricos"); sv=st.checkbox("Vetor velocidade",False,key="pa_sv"); sT=st.checkbox("Tangente unitário",True,key="pa_sT"); sN=st.checkbox("Normal unitário",True,key="pa_sN"); lt=st.checkbox("Reta tangente",True,key="pa_lt"); ln=st.checkbox("Reta normal",False,key="pa_ln"); sc=st.checkbox("Círculo osculador",True,key="pa_sc"); scale=st.slider("Escala dos vetores",.1,3.,1.,.1,key="pa_scale")
    if tmax<=tmin: st.error("É necessário ter t mínimo < t máximo."); return
    t=np.linspace(tmin,tmax,n); data=planar_curve(t,name,params,custom); mask=finite_mask(data.alpha)
    i=nearest_index(t,t0); p=data.alpha[i]; v=data.alpha1[i]; T=safe_unit(v); N=np.array([-T[1],T[0]]) if T is not None else None; kaparr=planar_curvature(data.alpha1,data.alpha2); kap=kaparr[i] if np.isfinite(kaparr[i]) else np.nan; radius=center=None
    if N is not None and np.isfinite(kap) and abs(kap)>1e-8: radius=1/abs(kap); center=p+N/kap
    Q,a,desc=rigid_controls_2d("pa"); det=float(np.linalg.det(Q)); alphaF=transform_points(data.alpha[mask],Q,a); pF=Q@p+a; vF=Q@v; TF=Q@T if T is not None else None; NF=det*(Q@N) if N is not None else None; centerF=Q@center+a if center is not None else None; kapF=det*kap if np.isfinite(kap) else kap
    st.latex(data.latex); st.info(data.note); st.markdown("**Movimento rígido aplicado:**"); st.latex(desc)
    left,right=st.columns([1.5,1])
    with left: st.plotly_chart(plot2(alphaF,pF,TF,NF,vF,centerF,radius,(sv,sT,sN,lt,ln,sc),scale),use_container_width=True)
    with right:
        st.subheader("Valores no ponto escolhido"); st.latex(rf"t_0={fmt(t[i])}"); st.latex(rf"F(\alpha(t_0))={vector_latex(pF)}"); st.latex(rf"\|\alpha'(t_0)\|={fmt(np.linalg.norm(v))}")
        if T is None: st.error("A curva não é regular neste instante.")
        else: st.latex(rf"\widetilde T(t_0)={vector_latex(TF)}"); st.latex(rf"\widetilde N(t_0)={vector_latex(NF)}"); st.latex(rf"\widetilde\kappa(t_0)={latex_scalar(kapF)}")
    with st.expander("Comprimento de arco",False):
        arc=np.concatenate(([0.],cumulative_trapezoid_local(np.linalg.norm(data.alpha1,axis=1),t))); st.latex(r"\ell(t)=\int_{t_{\min}}^t\|\alpha'(u)\|\,du"); st.latex(rf"L\approx {fmt(arc[-1])}"); st.caption("O símbolo s fica reservado à parametrização pelo comprimento de arco.")


def render_spatial_analysis():
    st.header(r"Análise de curvas espaciais em $\mathbb R^3$")
    names=["Reta espacial (Ex. 2.1.3)","Hélice circular (Ex. 2.1.4 e 2.2.3)","Parábola espacial","Cúbica torcida","Curva toroidal","Curva de Viviani","Personalizada"]
    with st.sidebar:
        st.subheader("Curva espacial"); name=st.selectbox("Exemplo",names,key="sa_name"); params={};custom=None
        if "Reta" in name:
            for k,val in zip(["x0","y0","z0","vx","vy","vz"],[0.,0.,0.,1.,1.,1.]): params[k]=st.number_input(k,value=val,key=f"sa_{k}")
        elif "Hélice" in name: params["a"]=st.slider("a",.2,4.,1.,.1,key="sa_a");params["b"]=st.slider("b",-2.,2.,.4,.1,key="sa_b")
        elif name=="Parábola espacial": params["c"]=st.slider("c",-3.,3.,1.,.1,key="sa_c")
        elif name=="Curva toroidal": params.update(R=2.5,r=.8,m=2.,n=3.)
        elif name=="Curva de Viviani": params["a"]=1.
        elif name=="Personalizada": custom={"x":st.text_input("x(t)","cos(t)",key="sa_x"),"y":st.text_input("y(t)","sin(t)",key="sa_y"),"z":st.text_input("z(t)","0.3*t",key="sa_z")}
        default=(-2*np.pi,2*np.pi) if "Hélice" in name else (-3.,3.); tmin=st.number_input("t mínimo",value=float(default[0]),key=f"sa_tmin_{name}");tmax=st.number_input("t máximo",value=float(default[1]),key=f"sa_tmax_{name}");n=st.slider("Resolução",300,1600,800,100,key="sa_n");t0=st.slider("Ponto t₀",float(tmin),float(tmax),float((tmin+tmax)/2),key=f"sa_t0_{name}") if tmax>tmin else tmin;scale=st.slider("Escala do triedro",.1,3.,1.,.1,key="sa_scale")
    if tmax<=tmin:st.error("É necessário ter t mínimo < t máximo.");return
    t=np.linspace(tmin,tmax,n);data=spatial_curve(t,name,params,custom);i=nearest_index(t,t0);p=data.alpha[i];v1=data.alpha1[i];v2=data.alpha2[i];T=safe_unit(v1);B=safe_unit(np.cross(v1,v2));N=safe_unit(np.cross(B,T)) if B is not None and T is not None else None;karr=spatial_curvature(data.alpha1,data.alpha2);tarr=spatial_torsion(data.alpha1,data.alpha2,data.alpha3);kap=karr[i];tau=tarr[i]
    Q,a,desc=rigid_controls_3d("sa");det=float(round(np.linalg.det(Q)));alphaF=transform_points(data.alpha,Q,a);pF=Q@p+a;TF=Q@T if T is not None else None;NF=det*(Q@N) if N is not None else None;BF=det*(Q@B) if B is not None else None;tauF=det*tau if np.isfinite(tau) else tau
    st.latex(data.latex);st.info(data.note);st.markdown("**Movimento rígido aplicado:**");st.latex(desc)
    l,r=st.columns([1.5,1]);
    with l:st.plotly_chart(plot3(alphaF,pF,TF,NF,BF,scale),use_container_width=True)
    with r:
        st.latex(rf"t_0={fmt(t[i])}");st.latex(rf"F(\alpha(t_0))={vector_latex(pF)}");st.latex(rf"\kappa(t_0)={latex_scalar(kap)}");st.latex(rf"\widetilde\tau(t_0)={latex_scalar(tauF)}")
        if T is None:st.error("Curva não regular no ponto.")
        elif B is None:st.warning("O triedro de Frenet não está definido porque α'×α''=0.")
    st.caption("A curvatura é preservada por toda isometria. A torção muda de sinal quando a isometria inverte a orientação do espaço.")

# ============================================================
# REPARAMETRIZAÇÃO — DOIS GRÁFICOS COM PONTOS MÓVEIS
# ============================================================
def _two_plot_layout(alpha,beta,pa,pb,titlea,titleb,is3=False):
    c1,c2=st.columns(2)
    if not is3:
        with c1:st.plotly_chart(plot2(alpha,pa,show=(False,False,False,False,False,False),title=titlea),use_container_width=True)
        with c2:st.plotly_chart(plot2(beta,pb,show=(False,False,False,False,False,False),title=titleb),use_container_width=True)
    else:
        with c1:st.plotly_chart(plot3(alpha,pa,title=titlea),use_container_width=True)
        with c2:st.plotly_chart(plot3(beta,pb,title=titleb),use_container_width=True)


def render_planar_reparam():
    st.header("Mudança de parâmetro e comprimento de arco em ℝ²")
    st.warning(r"Convenção: $t$ representa uma parametrização arbitrária; $u$ será usado em uma reparametrização geral; $s$ será usado somente quando a nova parametrização for pelo comprimento de arco.")
    options=["Reta: φ(u)=2u+1 (Ex. 2.2.1)","Circunferência: mesma orientação (Ex. 2.2.2)","Circunferência: orientação oposta (Ex. 2.2.2)","Reta pelo comprimento de arco (Ex. 2.2.4)","Circunferência unitária pelo comprimento de arco (Ex. 2.2.5)"]
    with st.sidebar: ex=st.selectbox("Exemplo de reparametrização",options,key="rp_ex");progress=st.slider("Posição do ponto no percurso",0.,1.,.35,.01,key="rp_prog")
    if ex.startswith("Reta: φ"):
        I=(-1.,3.);J=(-1.,1.); phi=lambda u:2*u+1; al=lambda t:np.column_stack((t,2*t)); be=lambda u:al(phi(u)); formula=r"\varphi:J=(-1,1)\to I=(-1,3),\quad\varphi(u)=2u+1,\quad\beta(u)=\alpha(\varphi(u))"; deriv=2.; arc=False
    elif "mesma orientação" in ex:
        I=(0.,2*np.pi);J=(0.,np.pi);phi=lambda u:2*u; al=lambda t:np.column_stack((3+np.cos(t),3+np.sin(t)));be=lambda u:al(phi(u));formula=r"\varphi_1:J=[0,\pi]\to I=[0,2\pi],\quad\varphi_1(u)=2u";deriv=2.;arc=False
    elif "oposta" in ex:
        I=(0.,2*np.pi);J=(0.,2*np.pi);phi=lambda u:2*np.pi-u;al=lambda t:np.column_stack((3+np.cos(t),3+np.sin(t)));be=lambda u:al(phi(u));formula=r"\varphi_2:J=[0,2\pi]\to I=[0,2\pi],\quad\varphi_2(u)=2\pi-u";deriv=-1.;arc=False
    elif "Reta pelo" in ex:
        I=(-3.,3.);J=(-3*np.sqrt(5),3*np.sqrt(5));phi=lambda s:s/np.sqrt(5);al=lambda t:np.column_stack((t,2*t));be=lambda s:al(phi(s));formula=r"\ell(t)=\sqrt5\,t,\quad \varphi:J\to I,\quad\varphi(s)=\ell^{-1}(s)=\frac{s}{\sqrt5}";deriv=1/np.sqrt(5);arc=True
    else:
        I=(0.,2*np.pi);J=(0.,2*np.pi);phi=lambda s:s;al=lambda t:np.column_stack((np.cos(t),np.sin(t)));be=lambda s:al(phi(s));formula=r"\ell(t)=t,\quad\varphi(s)=s,\quad\beta(s)=(\cos s,\sin s)";deriv=1.;arc=True
    tg=np.linspace(*I,700);ug=np.linspace(*J,700);A=al(tg);B=be(ug);u0=J[0]+progress*(J[1]-J[0]);t0=phi(u0);pa=al(np.array([t0]))[0];pb=be(np.array([u0]))[0]
    st.latex(r"\alpha:I\to\mathbb R^2,\qquad \beta:J\to\mathbb R^2,\qquad \beta=\alpha\circ\varphi");st.latex(formula);st.latex(rf"\varphi'={fmt(deriv)}")
    if deriv>0:st.success("A mudança de parâmetro é crescente: a orientação é preservada.")
    else:st.error("A mudança de parâmetro é decrescente: a orientação é invertida.")
    label="s" if arc else "u";st.latex(rf"{label}_0={fmt(u0)},\qquad t_0=\varphi({label}_0)={fmt(t0)},\qquad \beta({label}_0)=\alpha(t_0)")
    _two_plot_layout(A,B,pa,pb,"Parametrização original α(t)",f"Reparametrização β({label})")
    if arc:st.latex(r"\|\beta'(s)\|=1\qquad\text{(parâmetro igual ao comprimento de arco)}")


def render_spatial_reparam():
    st.header("Mudança de parâmetro e comprimento de arco em ℝ³")
    st.warning(r"Convenção: $t$ é um parâmetro arbitrário; $u$ é usado em mudanças gerais; $s$ é reservado ao comprimento de arco.")
    with st.sidebar:
        ex=st.selectbox("Exemplo",["Hélice pelo comprimento de arco (Ex. 2.2.3)","Hélice com orientação preservada","Hélice com orientação invertida"],key="rs_ex");A0=st.slider("a",.2,3.,1.,.1,key="rs_a");b=st.slider("b",.1,2.,.5,.1,key="rs_b");progress=st.slider("Posição do ponto",0.,1.,.35,.01,key="rs_prog")
    I=(-np.pi,np.pi);al=lambda t:np.column_stack((A0*np.cos(t),A0*np.sin(t),b*t))
    if "comprimento" in ex:
        v=np.sqrt(A0*A0+b*b);J=(-np.pi*v,np.pi*v);phi=lambda s:s/v;formula=rf"\ell(t)=\sqrt{{a^2+b^2}}t,\quad\varphi(s)=\frac{{s}}{{\sqrt{{a^2+b^2}}}}";deriv=1/v;label="s";arc=True
    elif "preservada" in ex:
        J=(-np.pi/2,np.pi/2);phi=lambda u:2*u;formula=r"\varphi(u)=2u";deriv=2.;label="u";arc=False
    else:
        J=(-np.pi,np.pi);phi=lambda u:-u;formula=r"\varphi(u)=-u";deriv=-1.;label="u";arc=False
    be=lambda q:al(phi(q));tg=np.linspace(*I,700);ug=np.linspace(*J,700);u0=J[0]+progress*(J[1]-J[0]);t0=phi(u0);A=al(tg);B=be(ug);pa=al(np.array([t0]))[0];pb=be(np.array([u0]))[0]
    st.latex(r"\alpha:I\to\mathbb R^3,\qquad\beta:J\to\mathbb R^3,\quad\beta=\alpha\circ\varphi");st.latex(formula);st.latex(rf"\varphi'={fmt(deriv)}")
    if deriv>0:st.success("Orientação preservada.")
    else:st.error("Orientação invertida.")
    st.latex(rf"{label}_0={fmt(u0)},\quad t_0=\varphi({label}_0)={fmt(t0)}")
    _two_plot_layout(A,B,pa,pb,"Hélice α(t)",f"Hélice reparametrizada β({label})",True)
    if arc:st.latex(r"\|\beta'(s)\|=1")

# ============================================================
# TEOREMA FUNDAMENTAL
# ============================================================
def planar_kappa(s,name,p,expr):
    if name=="Nula":return np.zeros_like(s)
    if name=="Constante":return np.full_like(s,p["c"])
    if name=="Linear":return p["a"]*s
    if name=="Senoidal":return p["a"]*np.sin(p["b"]*s)
    if name=="Racional":return p["a"]/(1+s*s)
    return eval_expr(expr,"s",s)


def reconstruct_planar(s,k,p0,theta0):
    th=theta0+np.concatenate(([0.],cumulative_trapezoid_local(k,s)));T=np.column_stack((np.cos(th),np.sin(th)));x=p0[0]+np.concatenate(([0.],cumulative_trapezoid_local(T[:,0],s)));y=p0[1]+np.concatenate(([0.],cumulative_trapezoid_local(T[:,1],s)));N=np.column_stack((-T[:,1],T[:,0]));return np.column_stack((x,y)),T,N


def render_planar_theorem():
    st.header("Obter uma curva plana através do Teorema Fundamental")
    st.latex(r"\theta'(s)=\kappa(s),\qquad T(s)=(\cos\theta(s),\sin\theta(s)),\qquad\alpha'(s)=T(s)")
    with st.sidebar:
        name=st.selectbox("κ(s)",["Nula","Constante","Linear","Senoidal","Racional","Personalizada"],key="pt_k");p={};expr="1+0.2*sin(s)"
        if name=="Constante":p["c"]=st.slider("c",-3.,3.,1.,.1,key="pt_c")
        elif name in {"Linear","Senoidal","Racional"}:p["a"]=st.slider("a",-3.,3.,.5,.1,key="pt_a");p["b"]=st.slider("b",.1,5.,1.,.1,key="pt_b") if name=="Senoidal" else 1.
        elif name=="Personalizada":expr=st.text_input("Expressão",expr,key="pt_expr")
        smin=st.number_input("s mínimo",value=0.,key="pt_smin");smax=st.number_input("s máximo",value=12.,key="pt_smax");n=st.slider("Resolução",300,1800,900,100,key="pt_n");x0=st.number_input("x₀",value=0.,key="pt_x0");y0=st.number_input("y₀",value=0.,key="pt_y0");ang=st.slider("θ₀",-180,180,0,5,key="pt_ang");sm=st.slider("Ponto s₀",float(smin),float(smax),float((smin+smax)/2),key="pt_sm") if smax>smin else smin
    if smax<=smin:st.error("s mínimo deve ser menor que s máximo.");return
    s=np.linspace(smin,smax,n);k=planar_kappa(s,name,p,expr);alpha,T,N=reconstruct_planar(s,k,np.array([x0,y0]),np.radians(ang));i=nearest_index(s,sm);Q,a,desc=rigid_controls_2d("pt");det=np.linalg.det(Q);alphaF=transform_points(alpha,Q,a);TF=Q@T[i];NF=det*(Q@N[i]);pF=alphaF[i]
    formula={"Nula":r"\kappa(s)=0","Constante":rf"\kappa(s)={p.get('c',0):.3g}","Linear":rf"\kappa(s)={p.get('a',0):.3g}s","Senoidal":rf"\kappa(s)={p.get('a',0):.3g}\sin({p.get('b',1):.3g}s)","Racional":rf"\kappa(s)=\frac{{{p.get('a',0):.3g}}}{{1+s^2}}","Personalizada":rf"\kappa(s)={expr}"}[name]
    st.markdown("**Função escolhida e movimento rígido:**");st.latex(formula);st.latex(desc);st.plotly_chart(plot2(alphaF,pF,TF,NF,show=(False,True,True,True,True,False)),use_container_width=True)


def invariant(s,name,p,expr):
    if name=="Nula":return np.zeros_like(s)
    if name=="Constante":return np.full_like(s,p["c"])
    if name=="Linear":return p["a"]+p["b"]*s
    if name=="Senoidal":return p["a"]+p["b"]*np.sin(p["c"]*s)
    if name=="Racional":return p["a"]/(1+p["b"]*s*s)
    return eval_expr(expr,"s",s)


def orthonormal_frame(v1,v2):
    T=safe_unit(v1)
    if T is None:raise ValueError("T inicial não pode ser nulo.")
    N=safe_unit(v2-np.dot(v2,T)*T)
    if N is None:raise ValueError("T e N auxiliar não podem ser paralelos.")
    B=safe_unit(np.cross(T,N));N=np.cross(B,T);return T,N,B


def solve_frenet(s,k,tau,p0,T0,N0,B0):
    Y=np.zeros((len(s),12));Y[0]=np.concatenate((p0,T0,N0,B0))
    def rhs(x,y):
        T=y[3:6];N=y[6:9];B=y[9:12];ka=float(np.interp(x,s,k));ta=float(np.interp(x,s,tau));return np.concatenate((T,ka*N,-ka*T+ta*B,-ta*N))
    for i in range(len(s)-1):
        h=s[i+1]-s[i];x=s[i];y=Y[i];k1=rhs(x,y);k2=rhs(x+h/2,y+h*k1/2);k3=rhs(x+h/2,y+h*k2/2);k4=rhs(x+h,y+h*k3);yn=y+h*(k1+2*k2+2*k3+k4)/6;T,N,B=orthonormal_frame(yn[3:6],yn[6:9]);yn[3:6]=T;yn[6:9]=N;yn[9:12]=B;Y[i+1]=yn
    return Y[:,:3],Y[:,3:6],Y[:,6:9],Y[:,9:12]


def render_spatial_theorem():
    st.header("Obter uma curva espacial através do Teorema Fundamental")
    st.latex(r"\alpha'=T,\quad T'=\kappa N,\quad N'=-\kappa T+\tau B,\quad B'=-\tau N")
    with st.sidebar:
        kn=st.selectbox("Tipo de κ",["Constante","Linear","Senoidal","Racional","Personalizada"],key="st_kn");kp={};ke="1+0.2*sin(s)"
        if kn=="Constante":kp["c"]=st.slider("κ constante",.05,3.,1.,.05,key="st_kc")
        elif kn=="Linear":kp.update(a=.8,b=.02)
        elif kn=="Senoidal":kp.update(a=1.,b=.25,c=1.)
        elif kn=="Racional":kp.update(a=1.,b=.2)
        else:ke=st.text_input("κ(s)",ke,key="st_ke")
        tn=st.selectbox("Tipo de τ",["Nula","Constante","Linear","Senoidal","Racional","Personalizada"],key="st_tn");tp={};te="0.3*cos(s)"
        if tn=="Constante":tp["c"]=st.slider("τ constante",-2.,2.,.4,.05,key="st_tc")
        elif tn=="Linear":tp.update(a=0.,b=.03)
        elif tn=="Senoidal":tp.update(a=0.,b=.4,c=1.)
        elif tn=="Racional":tp.update(a=.5,b=.2)
        elif tn=="Personalizada":te=st.text_input("τ(s)",te,key="st_te")
        smin=st.number_input("s mínimo",value=0.,key="st_smin");smax=st.number_input("s máximo",value=15.,key="st_smax");n=st.slider("Resolução",300,1400,700,100,key="st_n");sm=st.slider("Ponto s₀",float(smin),float(smax),float((smin+smax)/2),key="st_sm") if smax>smin else smin;scale=st.slider("Escala do triedro",.1,3.,1.,.1,key="st_scale")
    if smax<=smin:st.error("Intervalo inválido.");return
    s=np.linspace(smin,smax,n);k=invariant(s,kn,kp,ke);tau=invariant(s,tn,tp,te)
    if np.any(k<=0):st.error("O teorema clássico requer κ(s)>0 em todo o intervalo.");return
    T0,N0,B0=orthonormal_frame(np.array([1.,0.,0.]),np.array([0.,1.,0.]));alpha,T,N,B=solve_frenet(s,k,tau,np.zeros(3),T0,N0,B0);i=nearest_index(s,sm);Q,a,desc=rigid_controls_3d("st");det=round(np.linalg.det(Q));alphaF=transform_points(alpha,Q,a);TF=Q@T[i];NF=det*(Q@N[i]);BF=det*(Q@B[i]);tauF=det*tau[i]
    st.markdown("**Funções escolhidas:**");st.latex(rf"\kappa(s_0)={fmt(k[i])},\qquad\tau(s_0)={fmt(tau[i])}");st.latex(desc);st.plotly_chart(plot3(alphaF,alphaF[i],TF,NF,BF,scale),use_container_width=True)
    with st.expander("Cálculos do sistema de Frenet–Serret",True):
        Tp=k[i]*N[i];Np=-k[i]*T[i]+tau[i]*B[i];Bp=-tau[i]*N[i]
        st.latex(rf"\alpha'(s_0)=T(s_0)={vector_latex(T[i])}");st.latex(rf"T'(s_0)=\kappa N={vector_latex(Tp)}");st.latex(rf"N'(s_0)=-\kappa T+\tau B={vector_latex(Np)}");st.latex(rf"B'(s_0)=-\tau N={vector_latex(Bp)}")
        st.caption("Após uma isometria que inverte a orientação, a curvatura permanece e a torção troca de sinal.")


# ============================================================
# MELHORIAS: MOVIMENTOS RÍGIDOS, REPARAMETRIZAÇÃO E TEOREMA
# ============================================================

def matrix_latex(M: np.ndarray, digits: int = 4) -> str:
    rows = [" & ".join(fmt(v, digits) for v in row) for row in np.asarray(M)]
    return r"\begin{pmatrix}" + r"\\".join(rows) + r"\end{pmatrix}"


def interval_latex(a: float, b: float, variable: str = "t") -> str:
    return rf"{variable}\in[{fmt(a,3)},{fmt(b,3)}]"


def rigid_controls_2d_full(key: str):
    with st.sidebar:
        st.subheader("Movimento rígido em ℝ²")
        typ = st.selectbox(
            "Tipo de movimento rígido",
            ["Identidade", "Translação", "Rotação", "Reflexão em uma reta", "Reflexão deslizante"],
            key=f"{key}_full_rig2_type",
        )
        Q = np.eye(2)
        a = np.zeros(2)
        meta = {"type": typ, "angle": 0.0, "center": np.zeros(2), "direction": np.array([1.0, 0.0]), "slide": 0.0}
        if typ == "Translação":
            a = np.array([
                st.number_input("Translação a₁", value=1.0, key=f"{key}_full_tx"),
                st.number_input("Translação a₂", value=0.5, key=f"{key}_full_ty"),
            ])
        elif typ == "Rotação":
            deg = st.slider("Ângulo de rotação θ (graus)", -180, 180, 45, 5, key=f"{key}_full_rot")
            c = np.array([
                st.number_input("Centro c₁", value=0.0, key=f"{key}_full_cx"),
                st.number_input("Centro c₂", value=0.0, key=f"{key}_full_cy"),
            ])
            Q = rotation2(math.radians(deg))
            a = c - Q @ c
            meta.update(angle=deg, center=c)
        elif typ in {"Reflexão em uma reta", "Reflexão deslizante"}:
            deg = st.slider("Ângulo θ da reta com o eixo x", -180, 180, 30, 5, key=f"{key}_full_lineang")
            c = np.array([
                st.number_input("Ponto da reta c₁", value=0.0, key=f"{key}_full_lcx"),
                st.number_input("Ponto da reta c₂", value=0.0, key=f"{key}_full_lcy"),
            ])
            u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
            Q = 2 * np.outer(u, u) - np.eye(2)
            slide = 0.0
            if typ == "Reflexão deslizante":
                slide = st.slider("Deslizamento h ao longo da reta", -5.0, 5.0, 1.0, 0.1, key=f"{key}_full_slide")
            a = c - Q @ c + slide * u
            meta.update(angle=deg, center=c, direction=u, slide=slide)
        meta["Q"] = Q
        meta["a"] = a
        meta["det"] = float(np.linalg.det(Q))
    return Q, a, meta


def rigid_controls_3d_full(key: str):
    with st.sidebar:
        st.subheader("Movimento rígido em ℝ³")
        typ = st.selectbox(
            "Tipo de movimento rígido",
            ["Identidade", "Translação", "Rotação em torno de uma reta", "Reflexão em um plano", "Movimento helicoidal", "Rotoreflexão"],
            key=f"{key}_full_rig3_type",
        )
        Q = np.eye(3)
        a = np.zeros(3)
        meta = {"type": typ, "angle": 0.0, "point": np.zeros(3), "axis": np.array([0.0, 0.0, 1.0]), "slide": 0.0}
        if typ == "Translação":
            a = np.array([
                st.number_input("Translação a₁", value=1.0, key=f"{key}_full_tx"),
                st.number_input("Translação a₂", value=0.5, key=f"{key}_full_ty"),
                st.number_input("Translação a₃", value=0.25, key=f"{key}_full_tz"),
            ])
        elif typ in {"Rotação em torno de uma reta", "Movimento helicoidal", "Rotoreflexão"}:
            c = np.array([
                st.number_input("Ponto do eixo c₁", value=0.0, key=f"{key}_full_cx"),
                st.number_input("Ponto do eixo c₂", value=0.0, key=f"{key}_full_cy"),
                st.number_input("Ponto do eixo c₃", value=0.0, key=f"{key}_full_cz"),
            ])
            axis = np.array([
                st.number_input("Direção do eixo u₁", value=0.0, key=f"{key}_full_ax"),
                st.number_input("Direção do eixo u₂", value=0.0, key=f"{key}_full_ay"),
                st.number_input("Direção do eixo u₃", value=1.0, key=f"{key}_full_az"),
            ])
            u = safe_unit(axis)
            if u is None:
                st.error("A direção do eixo não pode ser nula.")
                u = np.array([0.0, 0.0, 1.0])
            deg = st.slider("Ângulo de rotação θ (graus)", -180, 180, 45, 5, key=f"{key}_full_ang")
            R = rodrigues(u, np.radians(deg))
            slide = 0.0
            if typ == "Movimento helicoidal":
                slide = st.slider("Translação h ao longo do eixo", -5.0, 5.0, 1.0, 0.1, key=f"{key}_full_pitch")
                Q = R
                a = c - Q @ c + slide * u
            elif typ == "Rotoreflexão":
                H = np.eye(3) - 2 * np.outer(u, u)
                Q = H @ R
                a = c - Q @ c
            else:
                Q = R
                a = c - Q @ c
            meta.update(angle=deg, point=c, axis=u, slide=slide)
        elif typ == "Reflexão em um plano":
            c = np.array([
                st.number_input("Ponto do plano c₁", value=0.0, key=f"{key}_full_pcx"),
                st.number_input("Ponto do plano c₂", value=0.0, key=f"{key}_full_pcy"),
                st.number_input("Ponto do plano c₃", value=0.0, key=f"{key}_full_pcz"),
            ])
            n = np.array([
                st.number_input("Normal do plano n₁", value=0.0, key=f"{key}_full_nx"),
                st.number_input("Normal do plano n₂", value=0.0, key=f"{key}_full_ny"),
                st.number_input("Normal do plano n₃", value=1.0, key=f"{key}_full_nz"),
            ])
            u = safe_unit(n)
            if u is None:
                st.error("A normal do plano não pode ser nula.")
                u = np.array([0.0, 0.0, 1.0])
            Q = np.eye(3) - 2 * np.outer(u, u)
            a = c - Q @ c
            meta.update(point=c, axis=u)
        meta["Q"] = Q
        meta["a"] = a
        meta["det"] = float(np.linalg.det(Q))
    return Q, a, meta


def render_rigid_math_2d(meta, alpha_latex: str, tmin: float, tmax: float):
    Q, a = meta["Q"], meta["a"]
    st.markdown("### Formulação matemática do movimento rígido")
    st.latex(r"F:\mathbb R^2\longrightarrow\mathbb R^2,\qquad F(p)=Mp+a,\qquad M^TM=I")
    st.latex(rf"M={matrix_latex(Q)},\qquad a={vector_latex(a,4)},\qquad \det M={fmt(meta['det'],2)}")
    typ = meta["type"]
    if typ == "Rotação":
        c = meta["center"]
        st.latex(rf"c={vector_latex(c,4)},\qquad \theta={fmt(meta['angle'],2)}^\circ")
        st.latex(r"F(p)=c+R_\theta(p-c)")
    elif typ in {"Reflexão em uma reta", "Reflexão deslizante"}:
        c, u = meta["center"], meta["direction"]
        st.latex(rf"L=\{{c+\lambda u:\lambda\in\mathbb R\}},\quad c={vector_latex(c,4)},\quad u={vector_latex(u,4)}")
        if typ == "Reflexão deslizante":
            st.latex(rf"F(p)=c+M(p-c)+h u,\qquad h={fmt(meta['slide'],3)}")
        else:
            st.latex(r"F(p)=c+M(p-c)")
    elif typ == "Translação":
        st.latex(r"F(p)=p+a")
    else:
        st.latex(r"F(p)=p")
    st.markdown(r"A curva transformada é a composição $\widetilde\alpha=F\circ\alpha$:")
    st.latex(rf"\alpha:[{fmt(tmin,3)},{fmt(tmax,3)}]\to\mathbb R^2,\qquad {alpha_latex}")
    st.latex(r"\widetilde\alpha:[t_{\min},t_{\max}]\to\mathbb R^2,\qquad \widetilde\alpha(t)=M\alpha(t)+a")
    st.latex(
        rf"\widetilde\alpha(t)=\left({fmt(Q[0,0],4)}x(t)+{fmt(Q[0,1],4)}y(t)+{fmt(a[0],4)},\;"
        rf"{fmt(Q[1,0],4)}x(t)+{fmt(Q[1,1],4)}y(t)+{fmt(a[1],4)}\right)"
    )


def render_rigid_math_3d(meta, alpha_latex: str, tmin: float, tmax: float):
    Q, a = meta["Q"], meta["a"]
    st.markdown("### Formulação matemática do movimento rígido")
    st.latex(r"F:\mathbb R^3\longrightarrow\mathbb R^3,\qquad F(p)=Mp+a,\qquad M^TM=I")
    st.latex(rf"M={matrix_latex(Q)},\qquad a={vector_latex(a,4)},\qquad \det M={fmt(meta['det'],2)}")
    typ = meta["type"]
    if typ in {"Rotação em torno de uma reta", "Movimento helicoidal", "Rotoreflexão"}:
        c, u = meta["point"], meta["axis"]
        st.latex(rf"L=\{{c+\lambda u:\lambda\in\mathbb R\}},\quad c={vector_latex(c,4)},\quad u={vector_latex(u,4)}")
        st.latex(rf"\theta={fmt(meta['angle'],2)}^\circ")
        if typ == "Movimento helicoidal":
            st.latex(rf"F(p)=c+R_\theta(p-c)+hu,\qquad h={fmt(meta['slide'],3)}")
        elif typ == "Rotoreflexão":
            st.latex(r"F(p)=c+H_uR_\theta(p-c),\qquad H_u=I-2uu^T")
        else:
            st.latex(r"F(p)=c+R_\theta(p-c)")
    elif typ == "Reflexão em um plano":
        c, n = meta["point"], meta["axis"]
        st.latex(rf"\Pi=\{{p\in\mathbb R^3:\langle p-c,n\rangle=0\}},\quad c={vector_latex(c,4)},\quad n={vector_latex(n,4)}")
        st.latex(r"F(p)=c+(I-2nn^T)(p-c)")
    elif typ == "Translação":
        st.latex(r"F(p)=p+a")
    else:
        st.latex(r"F(p)=p")
    st.latex(rf"\alpha:[{fmt(tmin,3)},{fmt(tmax,3)}]\to\mathbb R^3,\qquad {alpha_latex}")
    st.latex(r"\widetilde\alpha(t)=F(\alpha(t))=M\alpha(t)+a")
    st.latex(
        rf"\widetilde\alpha(t)=\left("
        rf"{fmt(Q[0,0],3)}x+{fmt(Q[0,1],3)}y+{fmt(Q[0,2],3)}z+{fmt(a[0],3)},\;"
        rf"{fmt(Q[1,0],3)}x+{fmt(Q[1,1],3)}y+{fmt(Q[1,2],3)}z+{fmt(a[1],3)},\;"
        rf"{fmt(Q[2,0],3)}x+{fmt(Q[2,1],3)}y+{fmt(Q[2,2],3)}z+{fmt(a[2],3)}\right)_{{(x,y,z)=\alpha(t)}}"
    )


def render_planar_analysis():
    st.header(r"Análise de curvas planas em $\mathbb R^2$")
    names=["Reta y=2x (Ex. 2.2.1 e 2.2.4)","Circunferência (Ex. 2.1.1 e 2.1.10)","Parábola (Ex. 2.1.2 e 2.1.14)","Exponencial (Ex. 2.1.9)","Senoide (Ex. 2.1.11)","Catenária (Ex. 2.1.12)","Cúspide singular (Ex. 2.1.13)","Auto-interseção (Ex. 2.1.15)","Não diferenciável: valor absoluto (Ex. 2.1.5–2.1.6)","Não diferenciável: raiz cúbica (Ex. 2.1.7)","Diferenciável, mas não C¹ (Ex. 2.1.8)","Personalizada"]
    with st.sidebar:
        st.subheader("Curva plana")
        name=st.selectbox("Exemplo",names,key="pa2_name")
        params={};custom=None
        if "Circunferência" in name: params["r"]=st.slider("r",.2,5.,1.,.1,key="pa2_r");params["xc"]=0.;params["yc"]=0.
        elif "Parábola" in name: params["a"]=st.slider("a",.2,3.,1.,.1,key="pa2_a")
        elif "Catenária" in name: params["a"]=st.slider("a",.2,3.,1.,.1,key="pa2_cat")
        elif name=="Personalizada": custom={"x":st.text_input("x(t)","cos(t)",key="pa2_x"),"y":st.text_input("y(t)","sin(t)",key="pa2_y")}
        default=(0.,2*np.pi) if ("Circunferência" in name or "Auto" in name) else (-3.,3.)
        tmin=st.number_input("t mínimo",value=float(default[0]),key=f"pa2_tmin_{name}")
        tmax=st.number_input("t máximo",value=float(default[1]),key=f"pa2_tmax_{name}")
        n=st.slider("Resolução",300,1600,800,100,key="pa2_n")
        t0=st.slider("Ponto t₀",float(tmin),float(tmax),float((tmin+tmax)/2),key=f"pa2_t0_{name}") if tmax>tmin else tmin
        scale=st.slider("Escala dos vetores",.1,3.,1.,.1,key="pa2_scale")
    if tmax<=tmin: st.error("É necessário ter t mínimo < t máximo."); return
    t=np.linspace(tmin,tmax,n)
    try: data=planar_curve(t,name,params,custom)
    except Exception as exc: st.error(f"Não foi possível construir a curva: {exc}"); return
    i=nearest_index(t,t0); p=data.alpha[i]; v=data.alpha1[i]; T=safe_unit(v); N=np.array([-T[1],T[0]]) if T is not None else None
    kaparr=planar_curvature(data.alpha1,data.alpha2); kap=kaparr[i] if np.isfinite(kaparr[i]) else np.nan; radius=center=None
    if N is not None and np.isfinite(kap) and abs(kap)>1e-8: radius=1/abs(kap); center=p+N/kap
    Q,a,meta=rigid_controls_2d_full("pa2"); det=meta["det"]
    alphaF=transform_points(data.alpha,Q,a); pF=Q@p+a; vF=Q@v; TF=Q@T if T is not None else None; NF=det*(Q@N) if N is not None else None; centerF=Q@center+a if center is not None else None
    st.latex(rf"\alpha:[{fmt(tmin,3)},{fmt(tmax,3)}]\longrightarrow\mathbb R^2,\qquad {data.latex}")
    st.info(data.note)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(plot2(data.alpha,p,T,N,v,center,radius,(True,True,True,True,True,True),scale,title="Curva original"),use_container_width=True)
    with c2: st.plotly_chart(plot2(alphaF,pF,TF,NF,vF,centerF,radius,(True,True,True,True,True,True),scale,title="Curva após o movimento rígido"),use_container_width=True)
    render_rigid_math_2d(meta,data.latex,tmin,tmax)
    st.markdown("### Valores no ponto selecionado")
    c1,c2=st.columns(2)
    with c1:
        st.latex(rf"t_0={fmt(t[i])},\quad \alpha(t_0)={vector_latex(p)}")
        st.latex(rf"T(t_0)={vector_latex(T) if T is not None else r'\text{não definido}'},\quad \kappa(t_0)={latex_scalar(kap)}")
    with c2:
        st.latex(rf"\widetilde\alpha(t_0)={vector_latex(pF)}")
        st.latex(rf"\widetilde T(t_0)={vector_latex(TF) if TF is not None else r'\text{não definido}'}")


def render_spatial_analysis():
    st.header(r"Análise de curvas espaciais em $\mathbb R^3$")
    names=["Reta espacial (Ex. 2.1.3)","Hélice circular (Ex. 2.1.4 e 2.2.3)","Parábola espacial","Cúbica torcida","Curva toroidal","Curva de Viviani","Personalizada"]
    with st.sidebar:
        st.subheader("Curva espacial"); name=st.selectbox("Exemplo",names,key="sa2_name"); params={};custom=None
        if "Reta" in name:
            for k,val in zip(["x0","y0","z0","vx","vy","vz"],[0.,0.,0.,1.,1.,1.]): params[k]=st.number_input(k,value=val,key=f"sa2_{k}")
        elif "Hélice" in name: params["a"]=st.slider("a",.2,4.,1.,.1,key="sa2_a");params["b"]=st.slider("b",-2.,2.,.4,.1,key="sa2_b")
        elif name=="Parábola espacial": params["c"]=st.slider("c",-3.,3.,1.,.1,key="sa2_c")
        elif name=="Curva toroidal": params.update(R=2.5,r=.8,m=2.,n=3.)
        elif name=="Curva de Viviani": params["a"]=1.
        elif name=="Personalizada": custom={"x":st.text_input("x(t)","cos(t)",key="sa2_x"),"y":st.text_input("y(t)","sin(t)",key="sa2_y"),"z":st.text_input("z(t)","0.3*t",key="sa2_z")}
        default=(-2*np.pi,2*np.pi) if "Hélice" in name else (-3.,3.)
        tmin=st.number_input("t mínimo",value=float(default[0]),key=f"sa2_tmin_{name}");tmax=st.number_input("t máximo",value=float(default[1]),key=f"sa2_tmax_{name}");n=st.slider("Resolução",300,1600,800,100,key="sa2_n")
        t0=st.slider("Ponto t₀",float(tmin),float(tmax),float((tmin+tmax)/2),key=f"sa2_t0_{name}") if tmax>tmin else tmin;scale=st.slider("Escala do triedro",.1,3.,1.,.1,key="sa2_scale")
    if tmax<=tmin: st.error("É necessário ter t mínimo < t máximo."); return
    t=np.linspace(tmin,tmax,n)
    try: data=spatial_curve(t,name,params,custom)
    except Exception as exc: st.error(f"Não foi possível construir a curva: {exc}"); return
    i=nearest_index(t,t0); p=data.alpha[i]; v1=data.alpha1[i];v2=data.alpha2[i];T=safe_unit(v1);B=safe_unit(np.cross(v1,v2));N=safe_unit(np.cross(B,T)) if B is not None and T is not None else None
    kap=spatial_curvature(data.alpha1,data.alpha2)[i];tau=spatial_torsion(data.alpha1,data.alpha2,data.alpha3)[i]
    Q,a,meta=rigid_controls_3d_full("sa2");det=meta["det"];alphaF=transform_points(data.alpha,Q,a);pF=Q@p+a;TF=Q@T if T is not None else None;NF=det*(Q@N) if N is not None else None;BF=det*(Q@B) if B is not None else None
    st.latex(rf"\alpha:[{fmt(tmin,3)},{fmt(tmax,3)}]\longrightarrow\mathbb R^3,\qquad {data.latex}")
    st.info(data.note)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(plot3(data.alpha,p,T,N,B,scale,title="Curva original"),use_container_width=True)
    with c2: st.plotly_chart(plot3(alphaF,pF,TF,NF,BF,scale,title="Curva após o movimento rígido"),use_container_width=True)
    render_rigid_math_3d(meta,data.latex,tmin,tmax)
    st.latex(rf"\kappa(t_0)={latex_scalar(kap)},\qquad \widetilde\tau(t_0)={latex_scalar(det*tau if np.isfinite(tau) else tau)}")


def _affine_phi_controls(key: str, I, default_orientation="Preservar"):
    with st.sidebar:
        orientation=st.radio("Orientação da reparametrização",["Preservar","Inverter"],index=0 if default_orientation=="Preservar" else 1,key=f"{key}_orient")
        jmin=st.number_input("u mínimo",value=-1.0,key=f"{key}_jmin")
        jmax=st.number_input("u máximo",value=1.0,key=f"{key}_jmax")
    if jmax<=jmin: raise ValueError("É necessário ter u mínimo < u máximo.")
    slope=(I[1]-I[0])/(jmax-jmin)
    if orientation=="Inverter": slope=-slope; intercept=I[1]-slope*jmin
    else: intercept=I[0]-slope*jmin
    return (jmin,jmax), slope, intercept


def _reparam_display(alpha_fun, I, phi_fun, J, alpha_formula, phi_formula, dim, progress, original_parameter="t", new_parameter="u", title=""):
    pg=np.linspace(I[0],I[1],800); ug=np.linspace(J[0],J[1],800); phivals=phi_fun(ug)
    if np.any(phivals<I[0]-1e-7) or np.any(phivals>I[1]+1e-7):
        st.error("A função φ não envia todo o domínio J para o intervalo I escolhido."); return
    A=alpha_fun(pg); B=alpha_fun(phivals); u0=J[0]+progress*(J[1]-J[0]);t0=float(phi_fun(np.array([u0]))[0]);pa=alpha_fun(np.array([t0]))[0];pb=alpha_fun(np.array([t0]))[0]
    st.latex(rf"\alpha:I=[{fmt(I[0],3)},{fmt(I[1],3)}]\to\mathbb R^{dim},\qquad {alpha_formula}")
    st.latex(rf"\varphi:J=[{fmt(J[0],3)},{fmt(J[1],3)}]\to I,\qquad {phi_formula}")
    st.latex(rf"\beta:J\to\mathbb R^{dim},\qquad \beta({new_parameter})=\alpha(\varphi({new_parameter}))")
    dphi=np.gradient(phivals,ug,edge_order=2); d0=float(dphi[nearest_index(ug,u0)])
    st.latex(rf"\varphi'({new_parameter}_0)={fmt(d0,5)}")
    if np.all(dphi>0): st.success("φ é estritamente crescente; a reparametrização preserva a orientação.")
    elif np.all(dphi<0): st.error("φ é estritamente decrescente; a reparametrização inverte a orientação.")
    else: st.warning("φ' muda de sinal ou se anula; esta escolha não define uma mudança de parâmetro difeomorfa em todo J.")
    st.latex(rf"{new_parameter}_0={fmt(u0)},\qquad {original_parameter}_0=\varphi({new_parameter}_0)={fmt(t0)},\qquad \beta({new_parameter}_0)=\alpha({original_parameter}_0)")
    _two_plot_layout(A,B,pa,pb,f"Curva original α({original_parameter})",f"Curva reparametrizada β({new_parameter})",dim==3)


def render_planar_reparam():
    st.header("Mudança de parâmetro e comprimento de arco em ℝ²")
    st.warning(r"Usamos $t$ para uma parametrização arbitrária, $u$ para uma mudança de parâmetro geral e $s$ exclusivamente para o comprimento de arco.")
    options=["Reta do capítulo: φ(u)=2u+1","Circunferência: orientação preservada","Circunferência: orientação invertida","Reta pelo comprimento de arco","Circunferência unitária pelo comprimento de arco","Curva personalizada"]
    with st.sidebar:
        ex=st.selectbox("Exemplo de reparametrização",options,key="rp2_ex");progress=st.slider("Posição do ponto no percurso",0.,1.,.35,.01,key="rp2_prog")
    if ex=="Reta do capítulo: φ(u)=2u+1":
        I=(-1.,3.);J=(-1.,1.);alpha=lambda t:np.column_stack((t,2*t));phi=lambda u:2*u+1;af=r"\alpha(t)=(t,2t)";pf=r"\varphi(u)=2u+1";_reparam_display(alpha,I,phi,J,af,pf,2,progress);return
    if ex.startswith("Circunferência:"):
        I=(0.,2*np.pi);alpha=lambda t:np.column_stack((3+np.cos(t),3+np.sin(t)));J=(0.,np.pi) if "preservada" in ex else (0.,2*np.pi);phi=(lambda u:2*u) if "preservada" in ex else (lambda u:2*np.pi-u);pf=r"\varphi(u)=2u" if "preservada" in ex else r"\varphi(u)=2\pi-u";_reparam_display(alpha,I,phi,J,r"\alpha(t)=(3+\cos t,3+\sin t)",pf,2,progress);return
    if ex=="Reta pelo comprimento de arco":
        I=(-3.,3.);J=(-3*np.sqrt(5),3*np.sqrt(5));alpha=lambda t:np.column_stack((t,2*t));phi=lambda s:s/np.sqrt(5);_reparam_display(alpha,I,phi,J,r"\alpha(t)=(t,2t)",r"\varphi(s)=s/\sqrt5",2,progress,new_parameter="s");st.latex(r"\|\beta'(s)\|=1");return
    if ex=="Circunferência unitária pelo comprimento de arco":
        I=(0.,2*np.pi);J=I;alpha=lambda t:np.column_stack((np.cos(t),np.sin(t)));phi=lambda s:s;_reparam_display(alpha,I,phi,J,r"\alpha(t)=(\cos t,\sin t)",r"\varphi(s)=s",2,progress,new_parameter="s");st.latex(r"\|\beta'(s)\|=1");return
    if ex.startswith("Curva do Teorema"):
        with st.sidebar:
            smin=st.number_input("s mínimo",value=0.,key="rp2_th_smin");smax=st.number_input("s máximo",value=10.,key="rp2_th_smax");theta0=st.slider("θ₀",-180,180,0,5,key="rp2_th_theta")
        I=(smin,smax);sg=np.linspace(*I,1200)
        if ex.endswith("κ=0"):k=np.zeros_like(sg);kf=r"\kappa(s)=0"
        elif ex.endswith("κ constante"):k=np.ones_like(sg);kf=r"\kappa(s)=1"
        elif ex.endswith("κ linear"):k=.2*sg;kf=r"\kappa(s)=0.2s"
        elif ex.endswith("κ senoidal"):k=np.sin(sg);kf=r"\kappa(s)=\sin s"
        else:k=1/(1+sg**2);kf=r"\kappa(s)=1/(1+s^2)"
        curve,_,_=reconstruct_planar(sg,k,np.zeros(2),np.radians(theta0))
        def alpha_s(q): return np.column_stack((np.interp(q,sg,curve[:,0]),np.interp(q,sg,curve[:,1])))
        try:J,m,b=_affine_phi_controls("rp2_th",I)
        except ValueError as exc:st.error(str(exc));return
        phi=lambda u:m*u+b
        _reparam_display(alpha_s,I,phi,J,rf"\alpha'(s)=T(s),\ {kf}",rf"\varphi(u)={fmt(m,4)}u+{fmt(b,4)}",2,progress,original_parameter="s",new_parameter="u");return
    with st.sidebar:
        xexpr=st.text_input("x(t)","cos(t)",key="rp2_cx");yexpr=st.text_input("y(t)","sin(t)",key="rp2_cy");imin=st.number_input("t mínimo",value=0.,key="rp2_imin");imax=st.number_input("t máximo",value=float(2*np.pi),key="rp2_imax");jmin=st.number_input("u mínimo",value=0.,key="rp2_jmin");jmax=st.number_input("u máximo",value=1.,key="rp2_jmax");phiexpr=st.text_input("φ(u)","2*pi*u",key="rp2_phi")
    if imax<=imin or jmax<=jmin:st.error("Os intervalos devem ser válidos.");return
    alpha=lambda q:np.column_stack((eval_expr(xexpr,"t",q),eval_expr(yexpr,"t",q)));phi=lambda u:eval_expr(phiexpr,"u",u)
    try:_reparam_display(alpha,(imin,imax),phi,(jmin,jmax),rf"\alpha(t)=({xexpr},{yexpr})",rf"\varphi(u)={phiexpr}",2,progress)
    except Exception as exc:st.error(f"Não foi possível avaliar a reparametrização: {exc}")


def render_spatial_reparam():
    st.header("Mudança de parâmetro e comprimento de arco em ℝ³")
    st.warning(r"Usamos $t$ para uma parametrização arbitrária, $u$ para uma mudança geral e $s$ exclusivamente para o comprimento de arco.")
    options=["Hélice pelo comprimento de arco","Hélice: orientação preservada","Hélice: orientação invertida","Curva personalizada"]
    with st.sidebar:
        ex=st.selectbox("Exemplo",options,key="rs2_ex");progress=st.slider("Posição do ponto",0.,1.,.35,.01,key="rs2_prog");A0=st.slider("a",.2,3.,1.,.1,key="rs2_a");b0=st.slider("b",.1,2.,.5,.1,key="rs2_b")
    if ex.startswith("Hélice"):
        I=(-np.pi,np.pi);alpha=lambda t:np.column_stack((A0*np.cos(t),A0*np.sin(t),b0*t))
        if "comprimento" in ex:
            v=np.sqrt(A0*A0+b0*b0);J=(-np.pi*v,np.pi*v);phi=lambda s:s/v;pf=rf"\varphi(s)=\dfrac{{s}}{{\sqrt{{{A0:.3g}^2+{b0:.3g}^2}}}}";new="s"
        elif "preservada" in ex:J=(-np.pi/2,np.pi/2);phi=lambda u:2*u;pf=r"\varphi(u)=2u";new="u"
        else:J=(-np.pi,np.pi);phi=lambda u:-u;pf=r"\varphi(u)=-u";new="u"
        _reparam_display(alpha,I,phi,J,rf"\alpha(t)=({A0:.3g}\cos t,{A0:.3g}\sin t,{b0:.3g}t)",pf,3,progress,new_parameter=new)
        if new=="s":st.latex(r"\|\beta'(s)\|=1")
        return
    if ex.startswith("Curva do Teorema"):
        I=(0.,12.);sg=np.linspace(*I,1000)
        if "constantes" in ex:k=np.ones_like(sg);tau=.4*np.ones_like(sg);af=r"\kappa(s)=1,\ \tau(s)=0.4"
        else:k=.8+.2*np.sin(sg);tau=np.zeros_like(sg);af=r"\kappa(s)=0.8+0.2\sin s,\ \tau(s)=0"
        T0,N0,B0=orthonormal_frame(np.array([1.,0.,0.]),np.array([0.,1.,0.]));curve,_,_,_=solve_frenet(sg,k,tau,np.zeros(3),T0,N0,B0)
        def alpha_s(q):return np.column_stack(tuple(np.interp(q,sg,curve[:,j]) for j in range(3)))
        try:J,m,b=_affine_phi_controls("rs2_th",I)
        except ValueError as exc:st.error(str(exc));return
        phi=lambda u:m*u+b
        _reparam_display(alpha_s,I,phi,J,rf"\alpha'(s)=T(s),\ {af}",rf"\varphi(u)={fmt(m,4)}u+{fmt(b,4)}",3,progress,original_parameter="s",new_parameter="u");return
    with st.sidebar:
        xexpr=st.text_input("x(t)","cos(t)",key="rs2_cx");yexpr=st.text_input("y(t)","sin(t)",key="rs2_cy");zexpr=st.text_input("z(t)","0.3*t",key="rs2_cz");imin=st.number_input("t mínimo",value=-3.,key="rs2_imin");imax=st.number_input("t máximo",value=3.,key="rs2_imax");jmin=st.number_input("u mínimo",value=-1.,key="rs2_jmin");jmax=st.number_input("u máximo",value=1.,key="rs2_jmax");phiexpr=st.text_input("φ(u)","3*u",key="rs2_phi")
    if imax<=imin or jmax<=jmin:st.error("Os intervalos devem ser válidos.");return
    alpha=lambda q:np.column_stack((eval_expr(xexpr,"t",q),eval_expr(yexpr,"t",q),eval_expr(zexpr,"t",q)));phi=lambda u:eval_expr(phiexpr,"u",u)
    try:_reparam_display(alpha,(imin,imax),phi,(jmin,jmax),rf"\alpha(t)=({xexpr},{yexpr},{zexpr})",rf"\varphi(u)={phiexpr}",3,progress)
    except Exception as exc:st.error(f"Não foi possível avaliar a reparametrização: {exc}")


def planar_closed_form(name,p,smin,x0,y0,theta0):
    if name=="Nula":
        return rf"\alpha(s)=({fmt(x0,3)},{fmt(y0,3)})+(s-{fmt(smin,3)})(\cos({fmt(theta0,3)}),\sin({fmt(theta0,3)}))"
    if name=="Constante" and abs(p.get("c",0.0))>EPS:
        c=p["c"]
        return rf"\alpha(s)=({fmt(x0,3)},{fmt(y0,3)})+\frac1{{{fmt(c,3)}}}\left(\sin\theta(s)-\sin\theta_0,\;-\cos\theta(s)+\cos\theta_0\right)"
    return r"\alpha(s)=p_0+\int_{s_{\min}}^s(\cos\theta(u),\sin\theta(u))\,du"


def render_planar_theorem():
    st.header("Obter uma curva plana através do Teorema Fundamental")
    st.markdown(r"O estudante prescreve $\kappa(s)$; a plataforma integra $\theta'=\kappa$ e depois $\alpha'=T$.")
    with st.sidebar:
        name=st.selectbox("κ(s)",["Nula","Constante","Linear","Senoidal","Racional","Personalizada"],key="pt2_k");p={};expr="1+0.2*sin(s)"
        if name=="Constante":p["c"]=st.slider("c",-3.,3.,1.,.1,key="pt2_c")
        elif name in {"Linear","Senoidal","Racional"}:p["a"]=st.slider("a",-3.,3.,.5,.1,key="pt2_a");p["b"]=st.slider("b",.1,5.,1.,.1,key="pt2_b") if name=="Senoidal" else 1.
        elif name=="Personalizada":expr=st.text_input("Expressão",expr,key="pt2_expr")
        smin=st.number_input("s mínimo",value=0.,key="pt2_smin");smax=st.number_input("s máximo",value=12.,key="pt2_smax");n=st.slider("Resolução",300,1800,900,100,key="pt2_n");x0=st.number_input("x₀",value=0.,key="pt2_x0");y0=st.number_input("y₀",value=0.,key="pt2_y0");ang=st.slider("θ₀ (graus)",-180,180,0,5,key="pt2_ang");sm=st.slider("Ponto s₀",float(smin),float(smax),float((smin+smax)/2),key="pt2_sm") if smax>smin else smin;scale=st.slider("Escala",.1,3.,1.,.1,key="pt2_scale")
    if smax<=smin:st.error("s mínimo deve ser menor que s máximo.");return
    s=np.linspace(smin,smax,n);k=planar_kappa(s,name,p,expr)
    if not np.all(np.isfinite(k)):st.error("κ(s) produziu valores não finitos.");return
    theta0=np.radians(ang);alpha,T,N=reconstruct_planar(s,k,np.array([x0,y0]),theta0);theta=theta0+np.concatenate(([0.],cumulative_trapezoid_local(k,s)));i=nearest_index(s,sm)
    formula={"Nula":r"\kappa(s)=0","Constante":rf"\kappa(s)={p.get('c',0):.3g}","Linear":rf"\kappa(s)={p.get('a',0):.3g}s","Senoidal":rf"\kappa(s)={p.get('a',0):.3g}\sin({p.get('b',1):.3g}s)","Racional":rf"\kappa(s)=\frac{{{p.get('a',0):.3g}}}{{1+s^2}}","Personalizada":rf"\kappa(s)={expr}"}[name]
    Q,a,meta=rigid_controls_2d_full("pt2");det=meta["det"];alphaF=transform_points(alpha,Q,a);TF=Q@T[i];NF=det*(Q@N[i]);pF=alphaF[i]
    st.markdown("### Dados matemáticos da curva obtida")
    st.latex(rf"\alpha:[{fmt(smin,3)},{fmt(smax,3)}]\longrightarrow\mathbb R^2")
    st.latex(formula)
    st.latex(rf"\alpha({fmt(smin,3)})=({fmt(x0,3)},{fmt(y0,3)}),\qquad \theta({fmt(smin,3)})={fmt(theta0,4)}\ \text{{rad}}={ang}^\circ")
    st.latex(r"\theta(s)=\theta_0+\int_{s_{\min}}^s\kappa(u)\,du")
    st.latex(r"T(s)=(\cos\theta(s),\sin\theta(s)),\qquad N(s)=(-\sin\theta(s),\cos\theta(s))")
    st.latex(planar_closed_form(name,p,smin,x0,y0,theta0))
    st.caption("Para curvaturas gerais, a última integral é calculada numericamente pelo método dos trapézios.")
    st.latex(rf"s_0={fmt(s[i])},\quad \theta(s_0)={fmt(theta[i])},\quad \alpha(s_0)={vector_latex(alpha[i])}")
    c1,c2=st.columns(2)
    with c1:st.plotly_chart(plot2(alpha,alpha[i],T[i],N[i],show=(False,True,True,True,True,False),scale=scale,title="Curva determinada por κ(s)"),use_container_width=True)
    with c2:st.plotly_chart(plot2(alphaF,pF,TF,NF,show=(False,True,True,True,True,False),scale=scale,title="Mesma curva após movimento rígido"),use_container_width=True)
    render_rigid_math_2d(meta,r"\alpha(s)\text{ dada pela integral acima}",smin,smax)
    st.success("A alteração dos dados iniciais ou a aplicação de um movimento rígido produz uma curva congruente com a mesma curvatura.")


def render_spatial_theorem():
    st.header("Obter uma curva espacial através do Teorema Fundamental")
    st.latex(r"\alpha'=T,\quad T'=\kappa N,\quad N'=-\kappa T+\tau B,\quad B'=-\tau N")
    with st.sidebar:
        kn=st.selectbox("Tipo de κ",["Constante","Linear","Senoidal","Racional","Personalizada"],key="st2_kn");kp={};ke="1+0.2*sin(s)"
        if kn=="Constante":kp["c"]=st.slider("κ constante",.05,3.,1.,.05,key="st2_kc")
        elif kn=="Linear":kp.update(a=.8,b=.02)
        elif kn=="Senoidal":kp.update(a=1.,b=.25,c=1.)
        elif kn=="Racional":kp.update(a=1.,b=.2)
        else:ke=st.text_input("κ(s)",ke,key="st2_ke")
        tn=st.selectbox("Tipo de τ",["Nula","Constante","Linear","Senoidal","Racional","Personalizada"],key="st2_tn");tp={};te="0.3*cos(s)"
        if tn=="Constante":tp["c"]=st.slider("τ constante",-2.,2.,.4,.05,key="st2_tc")
        elif tn=="Linear":tp.update(a=0.,b=.03)
        elif tn=="Senoidal":tp.update(a=0.,b=.4,c=1.)
        elif tn=="Racional":tp.update(a=.5,b=.2)
        elif tn=="Personalizada":te=st.text_input("τ(s)",te,key="st2_te")
        smin=st.number_input("s mínimo",value=0.,key="st2_smin");smax=st.number_input("s máximo",value=15.,key="st2_smax");n=st.slider("Resolução",300,1400,700,100,key="st2_n");sm=st.slider("Ponto s₀",float(smin),float(smax),float((smin+smax)/2),key="st2_sm") if smax>smin else smin;scale=st.slider("Escala do triedro",.1,3.,1.,.1,key="st2_scale")
    if smax<=smin:st.error("Intervalo inválido.");return
    s=np.linspace(smin,smax,n);k=invariant(s,kn,kp,ke);tau=invariant(s,tn,tp,te)
    if np.any(k<=0):st.error("O teorema clássico requer κ(s)>0 em todo o intervalo.");return
    T0,N0,B0=orthonormal_frame(np.array([1.,0.,0.]),np.array([0.,1.,0.]));alpha,T,N,B=solve_frenet(s,k,tau,np.zeros(3),T0,N0,B0);i=nearest_index(s,sm)
    Q,a,meta=rigid_controls_3d_full("st2");det=meta["det"];alphaF=transform_points(alpha,Q,a);TF=Q@T[i];NF=det*(Q@N[i]);BF=det*(Q@B[i])
    st.markdown("### Dados matemáticos da curva obtida")
    st.latex(rf"\alpha:[{fmt(smin,3)},{fmt(smax,3)}]\to\mathbb R^3,\qquad \|\alpha'(s)\|=1")
    st.latex(rf"\kappa(s_0)={fmt(k[i])},\qquad \tau(s_0)={fmt(tau[i])}")
    c1,c2=st.columns(2)
    with c1:st.plotly_chart(plot3(alpha,alpha[i],T[i],N[i],B[i],scale,title="Curva determinada por κ e τ"),use_container_width=True)
    with c2:st.plotly_chart(plot3(alphaF,alphaF[i],TF,NF,BF,scale,title="Curva após movimento rígido"),use_container_width=True)
    render_rigid_math_3d(meta,r"\alpha(s)\text{ solução do sistema de Frenet--Serret}",smin,smax)
    with st.expander("Cálculos do sistema de Frenet–Serret",True):
        Tp=k[i]*N[i];Np=-k[i]*T[i]+tau[i]*B[i];Bp=-tau[i]*N[i]
        st.latex(rf"\alpha'(s_0)=T(s_0)={vector_latex(T[i])}");st.latex(rf"T'(s_0)=\kappa N={vector_latex(Tp)}");st.latex(rf"N'(s_0)=-\kappa T+\tau B={vector_latex(Np)}");st.latex(rf"B'(s_0)=-\tau N={vector_latex(Bp)}")

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
st.title("Curvas planas e espaciais")
st.markdown("Módulo interativo baseado no capítulo de curvas: exemplos, regularidade, reparametrização, comprimento de arco, referenciais de Frenet e Teoremas Fundamentais.")
render_fundamental_concepts()
with st.sidebar:
    st.divider()
    st.header("Ambiente de estudo")
    env = st.radio(
        "Espaço",
        ["Curvas planas em ℝ²", "Curvas espaciais em ℝ³"],
        key="env",
    )
    mode = st.radio(
        "Modo",
        [
            "Analisar uma curva conhecida",
            "Obter a curva através do Teorema Fundamental",
        ],
        key="mode",
    )

    analysis_topic = None
    if mode == "Analisar uma curva conhecida":
        st.markdown("#### Conteúdo da análise")
        analysis_topic = st.radio(
            "Escolha o conteúdo",
            [
                "Objetos geométricos e movimentos rígidos",
                "Mudança de parâmetro e comprimento de arco",
            ],
            key="analysis_topic",
        )

if env.startswith("Curvas planas"):
    if mode == "Obter a curva através do Teorema Fundamental":
        render_planar_theorem()
    elif analysis_topic == "Mudança de parâmetro e comprimento de arco":
        render_planar_reparam()
    else:
        render_planar_analysis()
else:
    if mode == "Obter a curva através do Teorema Fundamental":
        render_spatial_theorem()
    elif analysis_topic == "Mudança de parâmetro e comprimento de arco":
        render_spatial_reparam()
    else:
        render_spatial_analysis()
