#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
montar_fm_copa.py

Fase 1: Scraper da Wikipedia (pt) → convocados_copa_2022.csv
         32 seleções, ~832 convocados, colunas: nome | nacionalidade | data_nascimento

Fase 2: Match convocados ↔ FM2023 → fm2023_copa_2022.csv
         Cascata de 3 níveis (Nat+DoB / só-DoB / Nat+nome)
         Schema: nome_wiki | dob_wiki | nacionalidade_wiki | nivel_match | <colunas FM>

Não toca em xcounter_wFM.ipynb.
Reaproveta cache HTML local (wiki_convocacoes.html) em execuções subsequentes.
"""

import os
import re
import sys
import logging
import requests
import pandas as pd
from typing import Optional
from bs4 import BeautifulSoup
from unidecode import unidecode

# Força UTF-8 no stdout/stderr (necessário no Windows com cp1252 padrão)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Configuração de logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Constantes ─────────────────────────────────────────────────────────────────
URL_WIKI   = "https://pt.wikipedia.org/wiki/Convoca%C3%A7%C3%B5es_para_a_Copa_do_Mundo_FIFA_de_2022"
CACHE_HTML = "wiki_convocacoes.html"
OUT_CONV   = "convocados_copa_2022.csv"
FM_PATH    = "fm2023.csv"
OUT_FM     = "fm2023_copa_2022.csv"
TOL_DIAS   = 2   # tolerância em dias para comparação de DoB


# ── Mapeamento: nome em PT (Wikipedia h3) → código FM (coluna Nat) ─────────────
# Construído a partir do COUNTRY_TO_FM do notebook (chaves em inglês)
# adaptando para os nomes em português que aparecem como <h3> na página da Wiki.
COUNTRY_PT_TO_FM = {
    "Argentina":      "ARG",
    "Austrália":      "AUS",
    "Bélgica":        "BEL",
    "Brasil":         "BRA",
    "Camarões":       "CMR",
    "Canadá":         "CAN",
    "Costa Rica":     "CRC",
    "Croácia":        "CRO",
    "Dinamarca":      "DEN",
    "Equador":        "ECU",
    "Inglaterra":     "ENG",
    "França":         "FRA",
    "Alemanha":       "GER",
    "Gana":           "GHA",
    "Irã":            "IRN",
    "Japão":          "JPN",
    "Coreia do Sul":  "KOR",
    "México":         "MEX",
    "Marrocos":       "MAR",
    "Países Baixos":  "NED",
    "Polônia":        "POL",
    "Portugal":       "POR",
    "Catar":          "QAT",
    "Arábia Saudita": "KSA",
    "Senegal":        "SEN",
    "Sérvia":         "SRB",
    "Espanha":        "ESP",
    "Suíça":          "SUI",
    "Tunísia":        "TUN",
    "Estados Unidos": "USA",
    "Uruguai":        "URU",
    "País de Gales":  "WAL",
}

# Meses em português → número
MESES_PT = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


# ══════════════════════════════════════════════════════════════════════════════
# Utilitários compartilhados
# ══════════════════════════════════════════════════════════════════════════════

def norm_nome(nome: str) -> str:
    """Cópia fiel de norm_nome() de xcounter_wFM.ipynb (Etapa 4A).
    unidecode + lowercase + remove não-letras + sufixos."""
    nome = unidecode(str(nome)).lower()
    nome = re.sub(r"[^a-z\s]", "", nome)
    nome = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", nome)
    return re.sub(r"\s+", " ", nome).strip()


def parse_dob_wiki(texto: str) -> Optional[str]:
    """'14 de janeiro de 1987 (39 anos)' → '1987-01-14'.
    Loga warning e retorna None se o formato não for reconhecido."""
    m = re.search(r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", texto.lower())
    if not m:
        if texto.strip():
            log.warning(f"DoB Wiki formato não reconhecido: '{texto}'")
        return None
    dia, mes_nome, ano = int(m.group(1)), m.group(2), m.group(3)
    mes = MESES_PT.get(mes_nome)
    if mes is None:
        log.warning(f"Mês não reconhecido: '{mes_nome}' em '{texto}'")
        return None
    return f"{ano}-{mes:02d}-{dia:02d}"


def parse_dob_fm(dob_str: str) -> Optional[str]:
    """'14/8/1984 (37 years old)' → '1984-08-14'.
    Cópia do notebook (Etapa 4A, função _parse_dob)."""
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", str(dob_str))
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else None


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Scraper da Wikipedia
# ══════════════════════════════════════════════════════════════════════════════

def baixar_ou_carregar_html() -> str:
    """Retorna HTML da página. Se CACHE_HTML existir, lê do disco."""
    if os.path.exists(CACHE_HTML):
        log.info(f"Cache encontrado — lendo '{CACHE_HTML}' sem requisição HTTP.")
        with open(CACHE_HTML, "r", encoding="utf-8") as f:
            return f.read()

    log.info("Baixando página da Wikipedia...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; xCounter-research/1.0; "
            "UFRJ Football Data Science; educational use)"
        )
    }
    resp = requests.get(URL_WIKI, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Wikipedia retornou HTTP {resp.status_code}. URL: {URL_WIKI}"
        )
    html = resp.text
    with open(CACHE_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"HTML salvo em '{CACHE_HTML}' ({len(html):,} bytes).")
    return html


def _idx_coluna(headers: list, candidatos: list) -> Optional[int]:
    """Retorna índice da primeira coluna cujo header contenha algum dos candidatos."""
    for i, h in enumerate(headers):
        if any(c.lower() in h.lower() for c in candidatos):
            return i
    return None


def _parsear_tabela(nome_selecao: str, table) -> list:
    """Extrai lista de dicts {nome, data_nascimento} de uma <table> BeautifulSoup."""
    rows = table.find_all("tr")
    if not rows:
        log.warning(f"[{nome_selecao}] Tabela sem linhas.")
        return []

    # O Wikipedia moderno tem um row 0 especial (mega-row) com 190+ células.
    # Procura o header real: primeira linha com 3–15 células onde uma é exatamente
    # "Jogador" ou "Nome" (diferentes times usam nomes de coluna diferentes).
    header_row_idx = 0
    for i, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        if not (3 <= len(cells) <= 15):
            continue  # pula mega-row (190+ células) e linhas verdadeiramente vazias
        texts = [c.get_text(strip=True).lower() for c in cells]
        if "jogador" in texts or "nome" in texts:
            header_row_idx = i
            break

    header_cells = rows[header_row_idx].find_all(["th", "td"])
    headers = [c.get_text(strip=True) for c in header_cells]

    idx_jogador = _idx_coluna(headers, ["Jogador", "Nome", "Player"])
    idx_dob     = _idx_coluna(headers, ["Idade", "DoB", "Nascimento", "Data"])

    if idx_jogador is None:
        log.warning(
            f"[{nome_selecao}] Coluna de jogador nao encontrada. "
            f"Headers: {headers}"
        )
        return []

    if idx_dob is None:
        # Alguns times têm tabela sem coluna de DoB (ex: Tunísia, Marrocos).
        # Extrai só o nome; data_nascimento ficará None para esses jogadores.
        log.warning(
            f"[{nome_selecao}] Sem coluna DoB — data_nascimento sera None "
            f"(matching via nivel 3 no Fase 2)."
        )

    jogadores = []
    for row in rows[header_row_idx + 1:]:
        cols = row.find_all(["td", "th"])
        min_cols = max(idx_jogador, idx_dob if idx_dob is not None else 0) + 1
        if len(cols) < min_cols:
            continue

        # Nome: prefere texto do link <a> (evita spans de bandeira/ícone)
        cel = cols[idx_jogador]
        link = cel.find("a")
        nome = link.get_text(strip=True) if link else cel.get_text(strip=True)
        if not nome:
            continue

        if idx_dob is not None and len(cols) > idx_dob:
            dob_texto = cols[idx_dob].get_text(" ", strip=True)
            dob_iso   = parse_dob_wiki(dob_texto)
        else:
            dob_iso = None

        jogadores.append({
            "nome":            nome,
            "nacionalidade":   nome_selecao,
            "data_nascimento": dob_iso,
        })

    return jogadores


def fase1_scraper(html: str) -> pd.DataFrame:
    log.info("=== FASE 1: Parsing das tabelas da Wikipedia ===")
    soup    = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output")
    if content is None:
        raise RuntimeError("Não encontrei 'div.mw-parser-output' no HTML.")

    todos_convocados = []
    audit_rows       = []
    secoes_ignoradas = []

    # Estrutura do Wikipedia moderno (Vector 2022):
    #   mw-parser-output
    #     section (grupo)
    #       div.mw-heading2  → "Grupo A"
    #       section (seleção)
    #         div.mw-heading3 → h3#Equador "Equador"
    #         p → treinador
    #         table → plantel (sem classe wikitable nesta versão da página)
    #       section (próxima seleção)
    for group_sec in content.find_all("section", recursive=False):
        for team_sec in group_sec.find_all("section", recursive=False):
            # Nome: h3 dentro do div.mw-heading3
            heading = team_sec.find("div", class_="mw-heading3")
            h3 = heading.find("h3") if heading else team_sec.find("h3")
            if not h3:
                continue
            nome_selecao = h3.get_text(strip=True)

            if nome_selecao not in COUNTRY_PT_TO_FM:
                secoes_ignoradas.append(nome_selecao)
                continue

            table = team_sec.find("table")
            if table is None:
                log.warning(f"[{nome_selecao}] Secao sem tabela — ignorada.")
                continue

            jogadores = _parsear_tabela(nome_selecao, table)
            n = len(jogadores)
            audit_rows.append({"selecao": nome_selecao, "jogadores": n})

            if not (23 <= n <= 26):
                log.warning(
                    f"[AVISO] {nome_selecao}: {n} jogadores extraídos "
                    f"(esperado 23-26). Verifique parsing."
                )

            todos_convocados.extend(jogadores)

    if secoes_ignoradas:
        log.warning(
            f"Secoes com tabela mas sem mapeamento PT->FM (ignoradas): "
            f"{secoes_ignoradas}"
        )

    # ── Tabela de auditoria ────────────────────────────────────────────────────
    print("\n─── Auditoria por seleção ──────────────────────────────────────────")
    df_audit = pd.DataFrame(audit_rows).sort_values("selecao").reset_index(drop=True)
    print(df_audit.to_string(index=False))
    total_jogadores = sum(r["jogadores"] for r in audit_rows)
    print(
        f"\nTotal seleções  : {len(audit_rows)}\n"
        f"Total jogadores : {total_jogadores}"
    )

    if not (800 <= total_jogadores <= 850):
        log.warning(
            f"ATENÇÃO: {total_jogadores} jogadores — esperado 800–850!"
        )
    else:
        log.info(f"Total dentro do intervalo esperado (800–850) ✓")

    df = pd.DataFrame(todos_convocados)
    df.to_csv(OUT_CONV, index=False, encoding="utf-8-sig")
    log.info(f"Salvo: '{OUT_CONV}' ({len(df)} linhas).")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — Match convocados ↔ FM2023
# ══════════════════════════════════════════════════════════════════════════════

def carregar_fm() -> pd.DataFrame:
    log.info(f"\n=== FASE 2: Carregando '{FM_PATH}' ===")
    df = pd.read_csv(FM_PATH, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()

    df["dob_iso"]   = df["DoB"].apply(parse_dob_fm)
    df["dob_date"]  = pd.to_datetime(df["dob_iso"], errors="coerce")
    df["name_norm"] = df["Name"].apply(norm_nome)
    df["Pac"]       = pd.to_numeric(df.get("Pac", 0), errors="coerce").fillna(0)

    log.info(f"FM2023: {len(df):,} jogadores, {df['Nat'].nunique()} nacionalidades.")
    return df


def _melhor_match(subset: pd.DataFrame, nome_wiki: str) -> Optional[pd.Series]:
    """Melhor linha do subset para nome_wiki.
    Prioridade: match exato → substring (palavra em comum) → maior Pac."""
    if subset.empty:
        return None

    nome_norm = norm_nome(nome_wiki)
    palavras  = set(nome_norm.split())

    exatos = subset[subset["name_norm"] == nome_norm]
    if not exatos.empty:
        return exatos.sort_values("Pac", ascending=False).iloc[0]

    if palavras:
        mask = subset["name_norm"].apply(lambda n: bool(palavras & set(n.split())))
        subs = subset[mask]
        if not subs.empty:
            return subs.sort_values("Pac", ascending=False).iloc[0]

    return None


def _mascara_dob(df: pd.DataFrame, dob_wiki: str) -> pd.Series:
    """Máscara booleana: dob_date do FM a no máximo TOL_DIAS dias de dob_wiki."""
    try:
        ref = pd.Timestamp(dob_wiki)
        return (df["dob_date"] - ref).abs() <= pd.Timedelta(days=TOL_DIAS)
    except Exception:
        return pd.Series(False, index=df.index)


def fase2_match(df_conv: pd.DataFrame, df_fm: pd.DataFrame) -> pd.DataFrame:
    log.info("Iniciando matching em cascata...")

    fm_por_nat = {nat: grp for nat, grp in df_fm.groupby("Nat")}

    resultados     = []
    nao_encontrados = []
    niveis         = {1: 0, 2: 0, 3: 0}
    nats_sem_mapa  = set()

    for _, conv in df_conv.iterrows():
        nat_wiki = conv["nacionalidade"]
        nat_fm   = COUNTRY_PT_TO_FM.get(nat_wiki)
        dob_wiki = conv["data_nascimento"]

        if nat_fm is None and nat_wiki not in nats_sem_mapa:
            log.warning(f"Nacionalidade não mapeada: '{nat_wiki}'")
            nats_sem_mapa.add(nat_wiki)

        linha_fm = None
        nivel    = None

        # ── Nível 1: Nat + DoB ────────────────────────────────────────────────
        if nat_fm and dob_wiki:
            sub = fm_por_nat.get(nat_fm, pd.DataFrame())
            if not sub.empty:
                mask = _mascara_dob(sub, dob_wiki)
                linha_fm = _melhor_match(sub[mask], conv["nome"])
                if linha_fm is not None:
                    nivel = 1

        # ── Nível 2: só DoB (fallback para naturalizados) ─────────────────────
        if linha_fm is None and dob_wiki:
            mask = _mascara_dob(df_fm, dob_wiki)
            linha_fm = _melhor_match(df_fm[mask], conv["nome"])
            if linha_fm is not None:
                nivel = 2

        # ── Nível 3: Nat + nome (sem restrição de DoB) ────────────────────────
        if linha_fm is None and nat_fm:
            sub = fm_por_nat.get(nat_fm, pd.DataFrame())
            linha_fm = _melhor_match(sub, conv["nome"])
            if linha_fm is not None:
                nivel = 3

        if linha_fm is not None:
            niveis[nivel] += 1
            resultados.append({
                "nome_wiki":          conv["nome"],
                "dob_wiki":           dob_wiki,
                "nacionalidade_wiki": nat_wiki,
                "nivel_match":        nivel,
                **linha_fm.to_dict(),
            })
        else:
            nao_encontrados.append({
                "nome":          conv["nome"],
                "nacionalidade": nat_wiki,
                "dob":           dob_wiki,
            })

    df_result = pd.DataFrame(resultados)
    total_conv  = len(df_conv)
    total_match = len(df_result)
    taxa_geral  = total_match / total_conv * 100 if total_conv else 0

    # ── Relatório de qualidade ─────────────────────────────────────────────────
    print("\n─── Relatório de qualidade do match ────────────────────────────────")
    print(f"Convocados total    : {total_conv}")
    print(f"Encontrados total   : {total_match}  ({taxa_geral:.1f}%)")
    print(f"  Nível 1 (Nat+DoB) : {niveis[1]}")
    print(f"  Nível 2 (só DoB)  : {niveis[2]}")
    print(f"  Nível 3 (Nat+nome): {niveis[3]}")
    print(f"Não encontrados     : {len(nao_encontrados)}")

    if taxa_geral < 90:
        log.warning(f"ATENÇÃO: taxa geral de {taxa_geral:.1f}% está abaixo de 90%!")

    # Taxa por seleção
    if not df_result.empty:
        print("\n─── Taxa de match por seleção ──────────────────────────────────────")
        totais  = df_conv.groupby("nacionalidade").size().rename("total")
        matches = df_result.groupby("nacionalidade_wiki").size().rename("matched")
        df_taxa = (
            totais.to_frame()
            .join(matches, how="left")
            .fillna({"matched": 0})
        )
        df_taxa["matched"] = df_taxa["matched"].astype(int)
        df_taxa["taxa%"]   = (df_taxa["matched"] / df_taxa["total"] * 100).round(1)
        df_taxa = df_taxa.sort_values("taxa%")
        print(df_taxa.to_string())

        problemas = df_taxa[df_taxa["taxa%"] < 70]
        if not problemas.empty:
            log.warning(
                f"\nATENÇÃO — seleções com taxa < 70% "
                f"(verifique mapeamento de nacionalidade):\n{problemas}"
            )

    # Lista de não encontrados
    if nao_encontrados:
        print("\n─── Convocados não encontrados ─────────────────────────────────────")
        print(pd.DataFrame(nao_encontrados).to_string(index=False))

    if not df_result.empty:
        df_result.to_csv(OUT_FM, index=False, encoding="utf-8-sig")
        log.info(f"\nSalvo: '{OUT_FM}' ({len(df_result):,} linhas).")
    else:
        log.warning("Nenhum match encontrado — fm2023_copa_2022.csv não foi gerado.")

    return df_result


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    html         = baixar_ou_carregar_html()
    df_conv      = fase1_scraper(html)
    df_fm        = carregar_fm()
    df_resultado = fase2_match(df_conv, df_fm)
    print("\n=== Concluído ===")
