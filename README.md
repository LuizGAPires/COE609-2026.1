# xCounter com Atributos do Football Manager

Projeto da disciplina **COE609 — Ciência de Dados Aplicada ao Futebol** (UFRJ, 2026.1).

## Contexto

O **xCounter** é um modelo de probabilidade de sucesso de contra-ataques que
tradicionalmente depende de dados de *tracking* contínuo (posição e velocidade de
todos os jogadores a 25 Hz) — caros e raramente disponíveis fora de clubes de elite.

Este trabalho investiga uma **hipótese de substituição**: será que atributos
físicos *estáticos* por jogador (velocidade, aceleração, resistência), extraídos da
base do **Football Manager**, podem fazer o papel de um *prior* físico e preservar o
poder preditivo, dispensando o *tracking*?

Para testar isso, construímos um conjunto unificado de **642 contra-ataques em cinco
competições masculinas** (Copa do Mundo 2022, Eurocopa 2020, La Liga 2020/21 e
Ligue 1 2021/22 e 2022/23), combinando:

- **Features espaciais** dos *freeze frames* StatsBomb 360 (geometria da jogada);
- **Features físicas** do Football Manager (cada competição casada à edição do FM
  mais próxima no tempo).

A avaliação é feita por uma **ablação** (modelo só espacial vs. espacial + físico)
com regressão logística, validação cruzada repetida e dupla métrica (AUC-ROC e
AUC-PR). **Resultado:** os atributos físicos do FM *não* agregam poder preditivo —
o sinal é dominado pela distância ao gol.

## Estrutura do repositório

| Caminho | Descrição |
|---|---|
| `notebooks/xcounter_simples.ipynb` | Pipeline de uma partida (identificação + features + FM) |
| `notebooks/xcounter_eda.ipynb` | Análise exploratória |
| `notebooks/xcounter_experimento_limiar.ipynb` | Sensibilidade ao limiar de xG |
| `notebooks/xcounter_modelagem.ipynb` | Modelagem e ablação (M1 vs M2) |
| `extrair_features.py` | Gera o dataset unificado a partir das bases do FM + StatsBomb |
| `dados_processados/` | Datasets já processados (consumidos pelos notebooks) |
| `data/` | **Não versionado** — bases brutas do Football Manager (ver abaixo) |

## Reprodução

### Dependências

```
numpy, pandas, matplotlib, mplsoccer, unidecode,
scikit-learn, xgboost, imbalanced-learn, scipy
```

Os dados de eventos e *freeze frames* 360 da StatsBomb são baixados ao vivo pela
rede via `mplsoccer.Sbopen` — é necessária conexão à internet.

### Opção A — Modelagem e EDA (rápido, sem FM)

Os notebooks de **modelagem** e **EDA** leem apenas `dados_processados/`, que já está
versionado. Basta rodá-los de cima para baixo — **não é preciso baixar o FM**.

### Opção B — Pipeline completo (com extração das features)

Para regenerar o dataset do zero com `extrair_features.py`, você precisa das bases
brutas do Football Manager, que **não são versionadas** (os CSVs têm 90–110 MB cada
e estouram o limite do GitHub).

1. Baixe os arquivos do Football Manager em:
   **https://www.kaggle.com/datasets/furkanuluta/football-manager-22-complete-player-dataset**
2. Coloque os CSVs na pasta `data/` com os nomes esperados pelo script
   (`data/fm2020.csv`, `data/fm2021.csv`, `data/fm2022.csv`, `data/fm2023.csv`).
3. Rode a extração:
   ```bash
   python extrair_features.py
   ```
   Isso (re)gera os arquivos em `dados_processados/`.
