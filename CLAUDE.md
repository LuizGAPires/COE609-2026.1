# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-notebook data-science proof-of-concept for **COE609 — Ciência de Dados Aplicada ao Futebol (UFRJ, 2026.1)**. The project (in Portuguese) builds **xCounter**, a predictive model for counter-attack success that replaces expensive continuous-tracking physical features with a *static prior* of per-player physical attributes scraped from **Football Manager 2023**. Core hypothesis: this substitution preserves predictive power, making the method viable where tracking data is unavailable.

Everything lives in [xcounter_wFM.ipynb](xcounter_wFM.ipynb). There is no application, build, or test suite — the deliverable is the notebook itself.

## Files

- `xcounter_wFM.ipynb` — the entire pipeline (28 cells, Python 3.11).
- `fm2023.csv` — Football Manager 2023 player attributes (shipped inside `fm2023.zip`; unzip before running). Referenced by `FM_CSV_PATH`.
- `Proposta_Projeto.pdf` — the course project brief.

## Running

Dependencies (no requirements file exists): `numpy`, `pandas`, `matplotlib`, `mplsoccer`, `unidecode`. StatsBomb event/360 data is fetched live over the network via `mplsoccer.Sbopen` — an internet connection is required.

```powershell
# one-time: extract the FM data the notebook expects at fm2023.csv
Expand-Archive fm2023.zip -DestinationPath .
jupyter notebook xcounter_wFM.ipynb   # run cells top-to-bottom
```

Cells are **order-dependent**: each later step consumes in-memory DataFrames produced earlier (`df_event`, `df_360`, `contra_ataques`, `df_dataset_final`, etc.). Always run from the top.

**Windows encoding gotcha:** the notebook output and source contain Portuguese accents and arrows (`→`). When running any helper script through the Bash tool's `python`, set `PYTHONIOENCODING=utf-8`, otherwise the default `cp1252` console codec raises `UnicodeEncodeError`. Read/parse the `.ipynb` with `encoding='utf-8'`.

## Pipeline architecture

The notebook is a linear ETL→feature→model-prep pipeline. Coordinates use the **StatsBomb system** (pitch 0–120 in x, 0–80 in y; the recovering team attacks toward x=120, goal center at `GOAL_X=120, GOAL_Y=40`).

1. **Etapa 1 — Load** (`parser.event`, `parser.frame`, `parser.match`): events, tactics, and 360 freeze-frames for one match (`MATCH_ID = 3857273`, Wales × Iran, 2022 World Cup).
2. **Etapa 2 — Identify counter-attacks** (`identificar_contra_ataques`): scans event chains starting from a recovery event and applies the config criteria to label each chain `sucesso ∈ {0,1}`.
3. **Etapa 3 — Spatial features** (`extrair_features_espaciais`): per chain, reads the start event's 360 freeze frame to compute distances/depth differentials between most-advanced attacker, defensive line, ball, and goal.
4. **Etapa 4 — FM2023 integration** (`norm_nome`, `buscar_fm`, `get_defender_attrs_medios`): name-normalizes and cascade-matches StatsBomb players to FM rows (nationality + birth date → looser fallbacks) to attach `FM_ATTRS` (Pac, Acc, Sta, Str, Agi, Bal, Jum); produces `df_dataset_final`.
5. **Etapa 5 — Validation viz**: heatmap of recovery starts, depth-vs-distance scatter, and a single 360 freeze-frame render (all via `mplsoccer.Pitch`).

## The config block (the one thing you'll edit)

Cell 3 centralizes all tunable criteria in a single `config` dict, validated by `validar_config`. **Change parameters only here; do not edit function bodies** (this rule is stated in the notebook). Key knobs:

- `zona_recuperacao`: `"metade_defensiva"` (x<60), `"terco_defensivo"` (x<40), `"campo_todo"`, or `"custom"` (uses `zona_recuperacao_x_max`).
- Chain window: `max_eventos_apos`, `max_tempo_s`, `progressao_vertical_min_m`.
- Break conditions (evaluated in order): `quebrar_em_fim_periodo`, `quebrar_em_perda_posse`, `quebrar_em_falta`, `quebrar_em_chute`.
- `eventos_inicio`: which event types start a chain (`Ball Recovery`, `Interception`, `Duel`).
- `criterio_sucesso`: `"qualquer_chute"`, `"xg_threshold"` (with `xg_threshold`, default 0.10), or `"gol"`.

Defaults marked "comportamento original" reproduce the baseline; the aliases `XG_THRESHOLD`, `CRIT_*` exist only so the visualization cells stay in sync. `rodar_analise_sensibilidade(lista_configs, match_ids)` re-runs identification across multiple configs/matches for comparison without touching the main flow.

## Known simplifications (documented in the notebook, intentional)

- **Defender proxy:** 360 frames carry no `player_id`, so the "defender" for FM lookup is the first defensive-position player (CB/LB/RB) from the team's starting XI — *fixed for all of that team's counter-attacks*. Flagged for correction in the "Próximos Passos" section.
- Validated on **one match** (~10–20 chains); scaling to a full competition is sketched in the final cell but not implemented.
