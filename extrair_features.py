# -*- coding: utf-8 -*-
"""Extrai features (espaciais 360 + físicas FM) para as competições ★★★ + World Cup
e unifica todos os contra-ataques num único dataset (xcounter_features_unificado.csv).

Cada competição usa a edição do FM mais próxima na data:
  World Cup 2022 -> FM2023 | Euro 2020 -> FM2021
  La Liga 20/21  -> FM2021 | Ligue 1 21/22 -> FM2022 | Ligue 1 22/23 -> FM2023

Matching FM: alias por UID -> fuzzy (SequenceMatcher) no pool da NACIONALIDADE DO
JOGADOR (via country_name do lineup; funciona para seleções e clubes).

Requer: data/fm2021.csv, data/fm2022.csv, data/fm2023.csv
"""
import os, re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from mplsoccer import Sbopen
from unidecode import unidecode

# (competition_id, season_id, nome, edição_FM)
COMPS = [
    (43, 106, 'World Cup 2022',  '2023'),
    (55, 43,  'Euro 2020',       '2021'),
    (11, 90,  'La Liga 2020/21', '2021'),
    (7, 108,  'Ligue 1 2021/22', '2022'),
    (7, 235,  'Ligue 1 2022/23', '2023'),
]

FM_ATTRS = ['Pac', 'Acc', 'Sta']          # suficientes para as features de saída
PLAY_PATTERN_ALVO = 'From Counter'
LIMIAR_FUZZY = 0.60
GOAL_X, GOAL_Y = 120.0, 40.0
config = {'criterio_sucesso': 'xg_threshold', 'xg_threshold': 0.10}

POSICOES_DEFENSIVAS = [
    'Centre Back', 'Center Back', 'Left Back', 'Right Back',
    'Left Centre Back', 'Left Center Back', 'Right Centre Back', 'Right Center Back',
    'Left Wing Back', 'Right Wing Back', 'Sweeper',
]
FEATS_ESP = ['dist_gol_defensor_proximo', 'dist_gol_atacante_avancado',
             'diferencial_profundidade', 'num_defensores_entre_bola_gol',
             'largura_linha_defensiva', 'dist_bola_gol', 'superioridade_terco_ofensivo']

# Aliases manuais p/ falsos negativos (nome normalizado StatsBomb -> UID FM, estável
# entre edições). Resolvem nomes civis longos vs. nome curto do FM e nacionalidade
# divergente (ex.: Saidy Janko é SUI no FM, não GAM).
ALIASES_FM = {
    # World Cup 2022 (FM2023)
    'lionel andres messi cuccittini':         7458500,   # Lionel Messi
    'joao mario naval da costa eduardo':       55022391,  # João Mário (1993)
    'otavio edmilson da silva monteiro':       19186901,  # Otávio
    'ngoran suiru fai collins':                13112672,  # Collins Fai
    'firas tariq nasser al albirakan':         23414228,  # Firas Al-Buraikan
    # Euro 2020 / La Liga 2020-21 / Ligue 1 (FM2021/2022)
    'jorge luiz frello filho':                 43036641,  # Jorginho
    'ruben santos gato alves dias':            55070299,  # Rúben Dias
    'domingos sousa coutinho meneses duarte':  55063355,  # Domingos Duarte
    'ronald federico araujo da silva':         78085068,  # Ronald Araújo
    'felipe augusto de almeida monteiro':      19162578,  # Felipe
    'yuri berchiche izeta':                    67035000,  # Yuri Berchiche
    'saidy janko':                             98030722,  # Saidy Janko (FM: SUI)
}

# StatsBomb country_name -> código FIFA do FM (87 validados; \xa0 normalizado p/ espaço)
COUNTRY_TO_FM = {
    'Algeria': 'ALG', 'Angola': 'ANG', 'Argentina': 'ARG', 'Australia': 'AUS',
    'Austria': 'AUT', 'Belgium': 'BEL', 'Benin': 'BEN', 'Bosnia and Herzegovina': 'BIH',
    'Brazil': 'BRA', 'Burkina Faso': 'BFA', 'Burundi': 'BDI', 'Cameroon': 'CMR',
    'Canada': 'CAN', 'Cape Verde': 'CPV', 'Central African Republic': 'CTA',
    'Chile': 'CHI', 'Colombia': 'COL', 'Comoros': 'COM', 'Congo, (Kinshasa)': 'COD',
    'Congo (Brazzaville)': 'CGO', 'Costa Rica': 'CRC', 'Croatia': 'CRO',
    'Czech Republic': 'CZE', "Côte d'Ivoire": 'CIV', 'Denmark': 'DEN',
    'Dominican Republic': 'DOM', 'Ecuador': 'ECU', 'Egypt': 'EGY', 'England': 'ENG',
    'Equatorial Guinea': 'EQG', 'Finland': 'FIN', 'France': 'FRA', 'Gabon': 'GAB',
    'Gambia': 'GAM', 'Georgia': 'GEO', 'Germany': 'GER', 'Ghana': 'GHA', 'Greece': 'GRE',
    'Guadeloupe': 'GLP', 'Guinea': 'GUI', 'Guinea-Bissau': 'GNB', 'Haiti': 'HAI',
    'Honduras': 'HON', 'Hungary': 'HUN', 'Iran, Islamic Republic of': 'IRN',
    'Israel': 'ISR', 'Italy': 'ITA', 'Japan': 'JPN', 'Korea (South)': 'KOR',
    'Kosovo': 'KOS', 'Macedonia, Republic of': 'MKD', 'Madagascar': 'MAD', 'Mali': 'MLI',
    'Martinique': 'MTQ', 'Mexico': 'MEX', 'Montenegro': 'MNE', 'Morocco': 'MAR',
    'Mozambique': 'MOZ', 'Netherlands': 'NED', 'Nigeria': 'NGA', 'Norway': 'NOR',
    'Paraguay': 'PAR', 'Peru': 'PER', 'Poland': 'POL', 'Portugal': 'POR', 'Qatar': 'QAT',
    'Romania': 'ROU', 'Russia': 'RUS', 'Saudi Arabia': 'KSA', 'Scotland': 'SCO',
    'Senegal': 'SEN', 'Serbia': 'SRB', 'Slovakia': 'SVK', 'Slovenia': 'SVN',
    'South Africa': 'RSA', 'Spain': 'ESP', 'Sweden': 'SWE', 'Switzerland': 'SUI',
    'Togo': 'TOG', 'Tunisia': 'TUN', 'Turkey': 'TUR', 'Ukraine': 'UKR',
    'United States of America': 'USA', 'Uruguay': 'URU',
    'Venezuela (Bolivarian Republic)': 'VEN', 'Wales': 'WAL', 'Zimbabwe': 'ZIM',
}


def norm_nome(nome):
    nome = unidecode(str(nome)).lower()
    nome = re.sub(r'[^a-z\s]', '', nome)
    nome = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', nome)
    return re.sub(r'\s+', ' ', nome).strip()


def country_to_fm(cn):
    return COUNTRY_TO_FM.get(str(cn).replace('\xa0', ' ').strip())


def load_fm(yy):
    """Carrega uma edição do FM. fm2023 tem layout limpo; fm2020/21/22 têm vírgulas
    dentro de valores (posições), então parseamos pelas pontas (id à esquerda,
    atributos à direita: Pac=-17, Acc=-18, Sta=-8)."""
    path = f'data/fm{yy}.csv'
    if yy == '2023':
        df = pd.read_csv(path, encoding='utf-8')
        df.columns = [c.strip() for c in df.columns]
        df = df[['UID', 'Name', 'DoB', 'Nat'] + FM_ATTRS].copy()
    else:
        rows = []
        with open(path, encoding='utf-8', errors='replace') as f:
            f.readline()
            for line in f:
                p = [x.strip() for x in line.rstrip('\n').split(',')]
                if len(p) < 60:
                    continue
                try:
                    uid = int(p[0])
                except ValueError:
                    continue
                rows.append((uid, p[1], p[2], p[3], p[-17], p[-18], p[-8]))
        df = pd.DataFrame(rows, columns=['UID', 'Name', 'DoB', 'Nat'] + FM_ATTRS)
    for c in FM_ATTRS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Name'] = df['Name'].astype(str).str.strip()
    df['Nat'] = df['Nat'].astype(str).str.strip()
    df['name_norm'] = df['Name'].apply(norm_nome)
    return {
        'df': df,
        'pools': {nat: sub for nat, sub in df.groupby('Nat')},
        'uid': df.drop_duplicates('UID').set_index('UID'),
    }


def match_fuzzy(player_name, nat_fm, fm, limiar=LIMIAR_FUZZY):
    alvo = norm_nome(player_name)
    if not alvo:
        return None, player_name, 0.0
    uid = ALIASES_FM.get(alvo)
    if uid is not None and uid in fm['uid'].index:
        row = fm['uid'].loc[uid]
        return {a: row.get(a, np.nan) for a in FM_ATTRS}, row['Name'], 1.0
    pool = fm['pools'].get(nat_fm)
    if pool is None or pool.empty:
        pool = fm['df']
    sims = pool['name_norm'].apply(lambda n: SequenceMatcher(None, alvo, n).ratio())
    idx, melhor = sims.idxmax(), float(sims.max())
    if melhor < limiar:
        return None, player_name, round(melhor, 3)
    row = pool.loc[idx]
    return {a: row.get(a, np.nan) for a in FM_ATTRS}, row['Name'], round(melhor, 3)


def identificar(df_event, cfg):
    df = df_event.sort_values(['period', 'minute', 'second', 'index']).reset_index(drop=True)
    eh = df['play_pattern_name'] == PLAY_PATTERN_ALVO
    grupo = (eh != eh.shift()).cumsum()
    xg_thr = cfg.get('xg_threshold', 0.10)
    out = []
    for _, cadeia in df[eh].groupby(grupo):
        primeiro, ultimo = cadeia.iloc[0], cadeia.iloc[-1]
        xv = cadeia['x'].dropna()
        x_ini = float(xv.iloc[0]) if len(xv) else np.nan
        x_fim = ultimo['end_x']
        if pd.isna(x_fim):
            x_fim = ultimo['x']
        x_fim = float(x_fim) if pd.notna(x_fim) else x_ini
        prog = (x_fim - x_ini) if pd.notna(x_ini) else np.nan
        t0 = primeiro['minute'] * 60 + primeiro['second']
        t1 = ultimo['minute'] * 60 + ultimo['second']
        chutes = cadeia[cadeia['type_name'] == 'Shot']
        tem_chute = len(chutes) > 0
        xg_final = float(chutes['shot_statsbomb_xg'].fillna(0).max()) if tem_chute else 0.0
        crit = cfg['criterio_sucesso']
        if crit == 'qualquer_chute':
            suc = tem_chute
        elif crit == 'gol':
            suc = bool((chutes['outcome_name'].astype(str).str.lower() == 'goal').any())
        else:
            suc = bool(tem_chute and xg_final >= xg_thr)
        out.append({
            'id_inicio': primeiro['id'], 'player_name': str(primeiro.get('player_name', '')),
            'player_id': primeiro.get('player_id'), 'team': primeiro['team_name'],
            'minute': int(primeiro['minute']), 'second': int(primeiro['second']),
            'tipo_inicio': primeiro['type_name'], 'num_eventos': len(cadeia),
            'progressao_x': prog, 'duracao_s': float(t1 - t0),
            'terminou_chute': tem_chute, 'xg_final': xg_final, 'sucesso': suc,
        })
    return out


def dist_eucl(x, y, x0=GOAL_X, y0=GOAL_Y):
    return float(np.sqrt((x - x0) ** 2 + (y - y0) ** 2))


def extrair_features_espaciais(event_id, df_event, df_360):
    ev = df_event[df_event['id'] == event_id]
    if ev.empty:
        return None
    ev = ev.iloc[0]
    frame = df_360[df_360['id'] == event_id]
    if frame.empty:
        return None
    bola_x, bola_y = ev.get('x'), ev.get('y')
    if bola_x is None or pd.isna(bola_x):
        return None
    bola_x, bola_y = float(bola_x), float(bola_y)
    tm = frame[frame['teammate'] == True].copy()
    op = frame[frame['teammate'] == False].copy()
    if len(op) < 1 or len(tm) < 1:
        return None
    x_def_av = float(op['x'].max()); y_def_av = float(op.loc[op['x'].idxmax(), 'y'])
    x_atac_av = float(tm['x'].max()); y_atac_av = float(tm.loc[tm['x'].idxmax(), 'y'])
    n_def_entre = int((op['x'] > bola_x).sum())
    top3 = op.nlargest(min(3, len(op)), 'x')
    largura = float(top3['y'].max() - top3['y'].min()) if len(top3) >= 2 else 0.0
    sup_of = int((tm['x'] > 80).sum()) - int((op['x'] > 80).sum())
    return {
        'dist_gol_defensor_proximo':     dist_eucl(x_def_av, y_def_av),
        'dist_gol_atacante_avancado':    dist_eucl(x_atac_av, y_atac_av),
        'diferencial_profundidade':      x_atac_av - x_def_av,
        'num_defensores_entre_bola_gol': n_def_entre,
        'largura_linha_defensiva':       largura,
        'dist_bola_gol':                 dist_eucl(bola_x, bola_y),
        'superioridade_terco_ofensivo':  sup_of,
    }


def defender_means(df_event, df_tactics, pid2nat, fm):
    xi = (df_event[df_event['type_name'] == 'Starting XI'][['id', 'team_name']].drop_duplicates())
    tact = df_tactics.merge(xi, on='id', how='inner')
    res = {}
    for team, grp in tact.groupby('team_name'):
        defs = grp[grp['position_name'].isin(POSICOES_DEFENSIVAS)]
        if defs.empty:
            defs = grp[grp['position_name'] != 'Goalkeeper']
        linhas = []
        for _, pl in defs.iterrows():
            nat = pid2nat.get(pl['player_id'])
            attrs, _, _ = match_fuzzy(pl['player_name'], nat, fm)
            if attrs is not None:
                linhas.append(attrs)
        if linhas:
            dd = pd.DataFrame(linhas)
            res[team] = {a: float(dd[a].mean()) for a in FM_ATTRS}
        else:
            res[team] = {a: np.nan for a in FM_ATTRS}
    return res


def main():
    parser = Sbopen()
    print('Carregando edições do FM necessárias...', flush=True)
    FM_CACHE = {}
    for yy in sorted({c[3] for c in COMPS}):
        FM_CACHE[yy] = load_fm(yy)
        print(f'  fm{yy}: {len(FM_CACHE[yy]["df"]):,} jogadores', flush=True)

    todas_linhas, falhas_atac = [], []
    nats_nao_mapeadas = {}
    _dif = lambda a, b: (a - b) if not (pd.isna(a) or pd.isna(b)) else np.nan

    for cid, sid, nome_comp, yy in COMPS:
        fm = FM_CACHE[yy]
        matches = parser.match(competition_id=cid, season_id=sid)
        print(f'\n=== {nome_comp} (FM{yy}) — {len(matches)} partidas ===', flush=True)
        n_ca_comp = 0
        for k, (_, mrow) in enumerate(matches.iterrows(), 1):
            mid = mrow['match_id']
            try:
                df_event, _, _, df_tactics = parser.event(mid)
                df_360, _ = parser.frame(mid)
                df_lineup = parser.lineup(mid)
                pid2nat = {}
                for _, r in df_lineup.iterrows():
                    code = country_to_fm(r['country_name'])
                    pid2nat[r['player_id']] = code
                    if code is None:
                        cn = str(r['country_name']).replace('\xa0', ' ').strip()
                        nats_nao_mapeadas[cn] = nats_nao_mapeadas.get(cn, 0) + 1

                cas = identificar(df_event, config)
                n_ca_comp += len(cas)
                if not cas:
                    continue

                defm = defender_means(df_event, df_tactics, pid2nat, fm)
                equipes = list(defm.keys())
                for ca in cas:
                    atac_team = ca['team']
                    def_team = next((e for e in equipes if e != atac_team), atac_team)
                    nat = pid2nat.get(ca['player_id'])
                    attrs, nome, sim = match_fuzzy(ca['player_name'], nat, fm)
                    falhou = attrs is None
                    if falhou:
                        attrs = {a: np.nan for a in FM_ATTRS}
                        falhas_atac.append({'competicao': nome_comp, 'match_id': mid,
                                            'player_name': ca['player_name'], 'nat_fm': nat,
                                            'melhor_sim': sim})
                    dinfo = defm.get(def_team, {})
                    pac_a, acc_a = attrs.get('Pac', np.nan), attrs.get('Acc', np.nan)
                    pac_d, acc_d = dinfo.get('Pac', np.nan), dinfo.get('Acc', np.nan)

                    esp = extrair_features_espaciais(ca['id_inicio'], df_event, df_360)
                    if esp is None:
                        esp = {f: np.nan for f in FEATS_ESP}

                    linha = dict(ca)
                    linha.pop('player_id', None)
                    linha.update(esp)
                    linha.update({
                        'competicao': nome_comp, 'fm_edicao': yy, 'match_id': mid,
                        'atac_fm_nome': nome, 'atac_fm_sim': sim, 'atac_match_ok': not falhou,
                        'pace_atacante': pac_a, 'acceleration_atacante': acc_a,
                        'stamina_atacante': attrs.get('Sta', np.nan),
                        'pace_defensor': pac_d, 'acceleration_defensor': acc_d,
                        'diferencial_pace': _dif(pac_a, pac_d),
                        'diferencial_acceleration': _dif(acc_a, acc_d),
                    })
                    todas_linhas.append(linha)
            except Exception as e:
                print(f'  [ERRO] match {mid}: {e}', flush=True)
            if k % 10 == 0:
                print(f'  ... {k}/{len(matches)} | CAs acumulados={len(todas_linhas)}', flush=True)
        print(f'  {nome_comp}: {n_ca_comp} contra-ataques', flush=True)

    df = pd.DataFrame(todas_linhas)
    os.makedirs('dados_processados', exist_ok=True)
    df.to_csv('dados_processados/xcounter_features_unificado.csv', index=False, encoding='utf-8')
    if falhas_atac:
        pd.DataFrame(falhas_atac).to_csv('dados_processados/xcounter_falhas_unificado.csv',
                                         index=False, encoding='utf-8')

    n, nf = len(df), len(falhas_atac)
    print('\n' + '=' * 64)
    print('RESUMO — DATASET UNIFICADO')
    print('=' * 64)
    print(f'Contra-ataques (linhas)   : {n}')
    print('Por competição:')
    print(df.groupby('competicao').agg(n=('sucesso', 'size'), sucessos=('sucesso', 'sum'),
                                       com_esp=('diferencial_profundidade', lambda s: s.notna().sum()),
                                       match_ok=('atac_match_ok', 'sum')).to_string())
    print(f'\nCom features espaciais 360 : {int(df["diferencial_profundidade"].notna().sum())}/{n}')
    print(f'Iniciadores SEM match FM   : {nf}  ({(nf/n*100 if n else 0):.1f}%)')
    if nf:
        print(pd.DataFrame(falhas_atac)[['competicao', 'player_name', 'nat_fm', 'melhor_sim']].to_string(index=False))
    if nats_nao_mapeadas:
        print(f'Países sem mapeamento FM (freq): {dict(sorted(nats_nao_mapeadas.items(), key=lambda x:-x[1]))}')
    print('\nArquivos salvos em dados_processados/: xcounter_features_unificado.csv'
          + (' + xcounter_falhas_unificado.csv' if nf else ''))


if __name__ == '__main__':
    main()
