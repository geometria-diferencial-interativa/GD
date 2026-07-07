# Geometria Diferencial Interativa

Plataforma em Streamlit para visualizar superfícies parametrizadas, curvatura média, curvatura gaussiana e variações normais de área.

## O que a plataforma faz

- Permite escolher uma superfície parametrizada.
- Permite alterar parâmetros e domínio.
- Permite escolher um ponto P=(u,v) sobre a superfície.
- Calcula pontualmente E, F, G, e, f, g, H(P) e K(P).
- Colore a superfície por H, K, densidade de área, h(u,v) ou altura z.
- Aplica variação normal X_t = X + t h N.
- Mostra área aproximada de X_t em função de t.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Arquivos principais

```text
app.py
requirements.txt
modules/
    __init__.py
    safe_eval.py
    surfaces.py
    geometry.py
    plotter.py
```
