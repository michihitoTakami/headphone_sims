"""Rebuilt report: 4-model acoustic comparison (Z1R / LCD / DX10000CL / DCA-AMTS)."""

import base64
import json
from pathlib import Path

summary = json.load(open("runs/final_summary.json"))
# Robustness study (issue #3): multi-subject comb variance + re-seat sensitivity
comb = json.load(open("runs/subject_comb_summary.json"))
variance = json.load(open("runs/subject_variance_summary.json"))
reseat = json.load(open("runs/reseat_summary.json"))
preservation = json.load(open("runs/preservation_summary.json"))
verify = {r["subject"]: r for r in json.load(open("runs/subjects_verify.json"))["results"]}
PANEL = variance["subjects"]


def b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


imgs = {k: b64(p) for k, p in {
    "radar": "runs/final_radar.png",
    "lateral": "runs/final_lateral.png",
    "levels": "runs/final_levels.png",
    "tf": "runs/final_pinna_tf.png",
    "coupling": "runs/coupling_comb.png",
    "dca_cmp": "runs/dca_real_vs_ideal.png",
    "fr_overlay": "runs/probe_fr_overlay.png",
    "local_uni": "runs/local_uniformity.png",
    "geo_z1r": "runs/final_geo_Z1R.png",
    "geo_lcd": "runs/final_geo_LCD.png",
    "geo_dx": "runs/final_geo_DX.png",
    "geo_dca": "runs/final_geo_DCA.png",
    "subjects": "runs/subjects_probe_layouts.png",
    "subj_comb": "runs/subject_comb.png",
    "subj_var": "runs/subject_variance.png",
    "preservation": "runs/preservation.png",
    "reseat": "runs/reseat.png",
}.items()}

ORDER = ["Z1R", "LCD", "DX", "DCA"]
ROWS = [
    ("波形一致・振幅込み(平均 / 最悪点)", "similarity_mean", +1,
     lambda s: f"{s['similarity_mean']:.3f} / {s['similarity_min']:.3f}"),
    ("波形形状のみ(旧指標、平均)", "shape_similarity_mean", +1,
     lambda s: f"{s['shape_similarity_mean']:.3f}"),
    ("レベルむら σ / 最悪ドロップ [dB]", "level_spread_db", -1,
     lambda s: f"{s['level_spread_db']:.1f} / {s['level_min_db']:.1f}"),
    ("入射角ばらつき [°]", "incidence_spread_deg", -1,
     lambda s: f"{s['incidence_spread_deg']:.1f}"),
    ("拡散度(平均)", "diffuseness_mean", -1,
     lambda s: f"{s['diffuseness_mean']:.3f}"),
    ("到達時間スプレッド・オンセット基準 [µs]", "arrival_spread_ms", -1,
     lambda s: f"{s['arrival_spread_ms']*1e3:.0f}"),
]


def table(phase: str) -> str:
    rows = []
    for label, key, sign, fmt in ROWS:
        vals = {k: summary[phase][k2][key] for k, k2 in zip(ORDER, ORDER)}
        vals = {k: summary[phase][k][key] for k in ORDER}
        best = max(ORDER, key=lambda k: sign * vals[k])
        cells = "".join(
            f"<td class='num{' good' if k == best else ''}'>{fmt(summary[phase][k])}</td>"
            for k in ORDER
        )
        rows.append(f"<tr><td>{label}</td>{cells}</tr>")
    return "\n".join(rows)


ROBUST4 = ["z1r", "lcd", "dx", "dca2"]  # 4 models: multi-subject + re-seat studies


def comb_table() -> str:
    """Per-model inter-subject comb stats (auto from subject_comb_summary)."""
    rows = []
    def swings(m):
        return [r["swing_db"] for r in comb[m]["per_subject"]]
    cells = {
        "コーム振れ幅 平均±σ [dB]": (
            lambda m: f"{comb[m]['swing_mean_db']:.1f} ± {comb[m]['swing_std_db']:.1f}",
            min, lambda m: comb[m]["swing_mean_db"]),
        "同・被験者レンジ [dB]": (
            lambda m: f"{min(swings(m)):.1f} 〜 {max(swings(m)):.1f}", None, None),
        "最深ノッチ周波数のレンジ [kHz]": (
            lambda m: f"{comb[m]['notch_hz_min']/1e3:.1f} 〜 {comb[m]['notch_hz_max']/1e3:.1f}",
            None, None),
        "最深ノッチ深さ 平均 [dB]": (
            lambda m: f"{comb[m]['notch_depth_mean_db']:.1f}",
            max, lambda m: comb[m]["notch_depth_mean_db"]),
        "ピンナノッチ衝突 / HWノッチ総数": (
            lambda m: f"{comb[m]['collisions_total']} / "
                      f"{comb[m]['collisions_total'] + comb[m]['false_cues_total']}",
            None, None),
    }
    for label, (fmt, agg, keyf) in cells.items():
        best = agg(ROBUST4, key=keyf) if agg else None
        rows.append("<tr><td>" + label + "</td>" + "".join(
            f"<td class='num{' good' if m == best else ''}'>{fmt(m)}</td>" for m in ROBUST4
        ) + "</tr>")
    return "\n".join(rows)


def variance_table() -> str:
    """Per-model inter-subject spread of core post-pinna metrics."""
    keys = [
        ("波形一致・核心帯 平均±σ", "similarity_mean", 1.0, "{:.3f}", max),
        ("レベルばらつき σ [dB] 平均±σ", "level_spread_db", 1.0, "{:.2f}", min),
        ("到達時間ばらつき [µs] 平均±σ", "arrival_spread_ms", 1e3, "{:.0f}", min),
        ("入射角ばらつき [°] 平均±σ", "incidence_spread_deg", 1.0, "{:.1f}", min),
    ]
    rows = []
    for label, key, scale, fmt, agg in keys:
        def val(m, stat, key=key, scale=scale):
            return variance["models"][m]["core"][key][stat] * scale
        best = agg(ROBUST4, key=lambda m: val(m, "mean"))
        best_std = min(ROBUST4, key=lambda m: val(m, "std"))
        cells = []
        for m in ROBUST4:
            mean_s = fmt.format(val(m, "mean"))
            std_s = fmt.format(val(m, "std"))
            cls = " good" if m == best else ""
            std_html = f"<strong>{std_s}</strong>" if m == best_std else std_s
            cells.append(f"<td class='num{cls}'>{mean_s} ± {std_html}</td>")
        rows.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    return "\n".join(rows)


def reseat_table() -> str:
    keys = [
        ("コーム平行移動 軸方向 [Hz/mm]", lambda e: f"{e['axial_shift_hz_per_mm']:+.0f}", min,
         lambda e: abs(e["axial_shift_hz_per_mm"])),
        ("コーム平行移動 上下方向 [Hz/mm]", lambda e: f"{e['vertical_shift_hz_per_mm']:+.0f}", min,
         lambda e: abs(e["vertical_shift_hz_per_mm"])),
        ("コーム変化 ΔC RMS 最大 [dB]", lambda e: f"{e['dC_rms_db_max']:.1f}", min,
         lambda e: e["dC_rms_db_max"]),
        ("核心帯 波形一致の変動幅", lambda e: f"{e['similarity_range']:.3f}", min,
         lambda e: e["similarity_range"]),
        ("10kHz照射マップ相関(上下±2mm 最小)", lambda e: f"{min(e['conditions'][c]['map10k_corr'] for c in ('vym2', 'vyp2')):.2f}", max,
         lambda e: min(e["conditions"][c]["map10k_corr"] for c in ("vym2", "vyp2"))),
    ]
    rows = []
    for label, fmt, agg, keyf in keys:
        best = agg(ROBUST4, key=lambda m: keyf(reseat[m]))
        rows.append("<tr><td>" + label + "</td>" + "".join(
            f"<td class='num{' good' if m == best else ''}'>{fmt(reseat[m])}</td>"
            for m in ROBUST4
        ) + "</tr>")
    return "\n".join(rows)


def preservation_table() -> str:
    """Per-model preservation decomposition vs the transparent reference."""
    keys = [
        ("ドライバ由来 P_drv = corr(裸, 透明基準)", "pinna", "P_drv"),
        ("構造由来 P_str = corr(構造あり, 裸)", "pinna", "P_str"),
        ("製品全体 P_tot = corr(構造あり, 透明基準)", "pinna", "P_tot"),
        ("個人署名の保存 P_sig_tot(4〜12.5kHz)", "pinna", "P_sig_tot"),
        ("個人署名の保存 P_sig_tot(核心帯 5〜10kHz)", "core", "P_sig_tot"),
    ]
    rows = []
    for label, band, key in keys:
        def val(m, stat, band=band, key=key):
            return preservation["models"][m][band][key][stat]
        best = max(ROBUST4, key=lambda m: val(m, "mean"))
        rows.append("<tr><td>" + label + "</td>" + "".join(
            f"<td class='num{' good' if m == best else ''}'>"
            f"{val(m, 'mean'):.3f} ± {val(m, 'std'):.3f}</td>"
            for m in ROBUST4
        ) + "</tr>")
    return "\n".join(rows)


def panel_table() -> str:
    head = "".join(f"<th>pp{s}</th>" for s in PANEL)
    feats = [
        ("外耳道の奥行き(先端から)[mm]", "recession_mm", "{:.0f}"),
        ("コンカ深さ [mm]", "concha_depth_mm", "{:.0f}"),
        ("耳介プローブ域 前後×上下 [mm]", None, None),
    ]
    rows = [f"<tr><th></th>{head}</tr>"]
    for label, key, fmt in feats:
        if key is None:
            cells = "".join(
                f"<td class='num'>{verify[s]['extent_x_mm']:.0f}×{verify[s]['extent_y_mm']:.0f}</td>"
                for s in PANEL)
        else:
            cells = "".join(f"<td class='num'>{fmt.format(verify[s][key])}</td>" for s in PANEL)
        rows.append(f"<tr><td>{label}</td>{cells}</tr>")
    return "\n".join(rows)


html = f"""<title>ヘッドホン4モデル音響比較</title>
<style>
:root {{
  --ground:#F6F8F9; --surface:#FFFFFF; --ink:#1B2530; --muted:#5C6B7A;
  --accent:#C05B21; --line:#DDE4E9; --good:#2E7D51;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#10161C; --surface:#1A222B; --ink:#E7ECF0; --muted:#93A1AE;
    --accent:#E68A4F; --line:#2A3540; --good:#5BBB8A;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#10161C; --surface:#1A222B; --ink:#E7ECF0; --muted:#93A1AE;
  --accent:#E68A4F; --line:#2A3540; --good:#5BBB8A;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--ground); color:var(--ink); margin:0;
  font-family:"Hiragino Sans","Noto Sans JP","Yu Gothic UI",system-ui,sans-serif;
  font-size:15.5px; line-height:1.75; }}
main {{ max-width:980px; margin:0 auto; padding:2.5rem 1.4rem 4rem; }}
.eyebrow {{ font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--accent); font-weight:700; margin:0 0 .4rem; }}
h1 {{ font-size:1.75rem; font-weight:800; margin:.1rem 0 .4rem; text-wrap:balance;
  letter-spacing:-.01em; }}
h2 {{ font-size:1.15rem; font-weight:700; margin:3rem 0 .8rem; padding-top:1.2rem;
  border-top:1px solid var(--line); }}
p {{ max-width:46em; margin:.5rem 0; }}
.lede {{ color:var(--muted); max-width:46em; }}
figure {{ margin:1.2rem 0; }}
figure img {{ max-width:100%; border:1px solid var(--line); border-radius:6px;
  background:#fff; display:block; }}
figcaption {{ font-size:.82rem; color:var(--muted); margin-top:.45rem; max-width:46em; }}
table {{ border-collapse:collapse; width:100%; font-size:.88rem; margin:1rem 0; }}
td, th {{ border-bottom:1px solid var(--line); padding:.45rem .8rem .45rem 0;
  text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-weight:600; }}
td.num {{ font-family:ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums; }}
td.good {{ color:var(--good); font-weight:700; }}
.callout {{ border-left:3px solid var(--accent); background:var(--surface);
  padding:.9rem 1.2rem; border-radius:0 8px 8px 0; margin:1.4rem 0; max-width:46em; }}
.callout strong {{ color:var(--accent); }}
code {{ font-family:ui-monospace,Menlo,monospace; font-size:.85em; background:var(--surface);
  border:1px solid var(--line); border-radius:4px; padding:.05em .35em; }}
.method {{ background:var(--surface); border:1px solid var(--line); border-radius:8px;
  padding:1rem 1.2rem; font-size:.86rem; margin:1.2rem 0; }}
footer {{ margin-top:3.5rem; padding-top:1.2rem; border-top:1px solid var(--line);
  font-size:.8rem; color:var(--muted); }}
</style>
<main>
<p class="eyebrow">headphone_sims — model comparison</p>
<h1>ヘッドホン4モデル音響比較</h1>
<p class="lede">MDR-Z1R型・LCD型・DX10000CL型・DCA型(AMTS)の実機構造モデルを、実測ピンナ(HUTUBS被験者1)に対する
3D FDTDシミュレーションで比較。「ピンナ全体に同一の波形・入射角・音量が届いているか」を、
音源の責任範囲(入射波面)とピンナ散乱(空間キュー生成)に分離して評価した。
さらに<strong>8被験者パネルでの個人差ロバスト性</strong>(セクション6)と
<strong>±1〜2mm再装着ズレへの感度</strong>(セクション7)を定量した。</p>

<div class="method"><strong>方法:</strong> 音響FDTD(0.5mm格子、20kHzで34点/波長、GPU)。
ピンナ表面プローブ約260点(ドライバから見通せる面のみ、外耳道基準)。傾き0°、
<strong>各モデルを物理的に可能な最小距離に配置</strong>(前面構造+2mmクリアランス):
Z1R 11mm / LCD 13mm / DX 9mm / DCA 24mm(ドライバ→ピンナ先端)= 頭部表面から29 / 31 / 27 / 42mm。
Z1R/LCD/DXは実機のイヤパッド深さ(2〜3cm)の範囲、DCAは厚さ21mmの前面スタック
(マグネット+AMTSインサート)が要求する物理最小距離。
<strong>入射波面プロトコル</strong>=ピンナ固体のみ除去し到達瞬間の窓(−0.1/+0.15ms)で評価。
ドーム振動板はテーパー駆動(Z1Rは剛性エッジ: リム際5mmのみ減衰)。剛体・密閉マウントモデル。
<strong>ハウジング前面</strong>: 実機同様、グリル/マグネット前面と面一のフランジを全モデルに設置
(封止壁が裸で立つ「堀」形状は周縁に実在しない影を作るため)。
入射窓は+0.45ms(最大口径の全面からの到達を収容 — 狭い窓は大口径を過小評価する)。
<strong>帯域方針</strong>: 全時間領域指標は<strong>1〜12.5kHzに帯域制限</strong>(ゼロ位相バンドパス)。
14kHz超は聴覚感度・空間キュー寄与とも小さく、含めると高域ビーミングの差を過大評価する。
加えて空間キューの核心である<strong>5〜10kHz帯</strong>の集計を別掲。
<strong>指標(v3改定)</strong>: 波形一致は<strong>振幅込み</strong>
(2·xcorr/(Eₓ+Eᵧ)、形と音圧が両方合って1.0 — −10dBの微弱プローブは形が完璧でも0.58)。
旧「形状のみ相関」は音圧ゼロ近くでも高得点になる欠陥があり参考値に降格。
到達時間はエンベロープピークからオンセット(50%立ち上がり)基準へ変更。
<strong>ロバスト性スタディ(セクション6・7)</strong>: HUTUBS 8被験者パネル
(形状特徴のgreedy max-min選定)×4モデル、および pp1 での再装着摂動
(駆動距離−1/+2mm、上下±2mm — 上下ズレはピンナ・外耳道・プローブごと移動しドライバは固定)×4モデル。
各条件で構造あり/裸ソース×incident/pinnaの4ランを回し、C(f)と核心帯指標を基準と同一プロトコルで算出。</div>

<h2>1. 比較した4モデル</h2>
<table>
<tr><th></th><th>MDR-Z1R型</th><th>LCD型</th><th>DX10000CL型</th><th>DCA型(AMTS)</th></tr>
<tr><td>振動板</td><td>70mm = 30mm中央ドーム(頂点6mm)+幅20mmの丸ドーナツ型エッジ(クレスト4.5mm)。剛性エッジ駆動: リム際5mmのみクランプ減衰(実効放射面積86%)</td>
<td>90×65mm 長方形平面</td><td>40mm フルドーム(34mm+ロールエッジ)、テーパー駆動</td>
<td>72×45mm 長方形平面(Expanse/E3級の角丸台形を矩形近似)</td></tr>
<tr><td>前面構造</td><td><strong>平面</strong>フィボナッチグリル(8+13螺旋、開口率71%、standoff 8.5mm)</td>
<td>マグネットバー: スロット11本/90mm(写真実測: ピッチ8.2mm、スロット2.6mm、開口率32%)+Fazor: 矩形5mm+おにぎり型5mm=厚さ10mm</td>
<td>ヘックスグリル(開口率52%、空洞6.5mm)</td>
<td>マグネットバー(スロット2.5mm/ピッチ6.5mm、厚さ4mm)+<strong>AMTSメタマテリアル・インサート(写真実測)</strong>: 耳側トップが3→15mmへ傾斜、ヘックス格子(実測ピッチ3.2mm、r=1.1mm)268セル。うち54セル(20%)が実機写真から抽出した<strong>潰しセル</strong>=上蓋付きλ/4側枝レゾネータ(ドライバ側に開口、c/4L≈5.7〜29kHz)、残りは開放チューブ。潰しは厚端側(チューニングが帯域内に入る場所)に集中</td></tr>
<tr><td>配置(→ピンナ先端 / 頭部表面から)</td><td class='num'>11mm / 29mm</td>
<td class='num'>13mm / 31mm</td><td class='num'>9mm / 27mm</td><td class='num'>24mm / 42mm</td></tr>
</table>
<figure><img src="data:image/png;base64,{imgs['geo_z1r']}" alt="Z1R断面">
<figcaption>MDR-Z1R型: ドーム+テーパーエッジ振動板と、それに沿う湾曲フィボナッチグリル。</figcaption></figure>
<figure><img src="data:image/png;base64,{imgs['geo_lcd']}" alt="LCD断面">
<figcaption>LCD型: 長方形平面振動板と縦バーのマグネットアレイ(Fazor面取り)。</figcaption></figure>
<figure><img src="data:image/png;base64,{imgs['geo_dx']}" alt="DX断面">
<figcaption>DX10000CL型: 40mmフルドームと高開口率ヘックスグリル。</figcaption></figure>
<figure><img src="data:image/png;base64,{imgs['geo_dca']}" alt="DCA断面">
<figcaption>DCA型: 72×45mm平面振動板 → マグネットバー → 傾斜AMTSインサート(実測潰しマップ)の3層スタック。</figcaption></figure>

<h3>AMTSの実機潰しパターン — 写真からの抽出と、理想化モデルとの差</h3>
<p>実機のAMTSインサート写真では一部のセルが白い材で「潰して」ある。これをセル単位で抽出した
(暗/明ブロブ検出 → 傾斜天面が平面であることを利用したホモグラフィ格子フィット →
セルごとの開/閉分類 → 帯状拡大画像で目視検証。境界の視射角が浅い2行±数セルの不確実性が残る)。
判明した設計: 潰しセルは死んだ穴ではなく<strong>上蓋付きλ/4側枝レゾネータ</strong>
(ドライバ側に開口)で、λ/4チューニングが空間キュー帯に入る<strong>厚端側に密に、
薄端側にはほぼ置かない</strong>散点配置になっている。物理的に、隣接セルと壁を共有する直管では
「長さのシャッフル」は不可能だが、「開/閉のシャッフル」は外形を乱さず可能 —
実機はこの合法な自由度で吸収の帯域重み付けと透過の空間均一化を同時にやっている。</p>
<figure><img src="data:image/png;base64,{imgs['dca_cmp']}" alt="実測vs理想化">
<figcaption>理想化(セル交互配置)vs 実測潰しマップ。実測マップは6〜10kHzのレベルむらを
一貫して低減(10kHz帯 σ5.5→4.8dB、8kHz帯 4.0→3.4dB)。</figcaption></figure>
<table>
<tr><th>指標</th><th>交互配置(理想化)</th><th>実測潰しマップ</th></tr>
<tr><td>カップリングコーム 5-10kHz振れ幅</td><td class='num'>18.4 dB</td><td class='num good'>11.9 dB(4モデル最小)</td></tr>
<tr><td>ピンナ後 波形一致(5-10kHz)</td><td class='num'>0.395</td><td class='num good'>0.515(DX/LCDと同着圏)</td></tr>
<tr><td>ピンナ後 到達スプレッド</td><td class='num'>52 µs</td><td class='num good'>25 µs</td></tr>
<tr><td>入射10kHz帯レベルむら σ</td><td class='num'>5.5 dB</td><td class='num good'>4.8 dB</td></tr>
</table>
<p>ただし回収されたのは<strong>総量指標</strong>(コーム・総合波形一致・到達整列)であって、
質的な署名は残存する。帯域別マップを見ると実測版でも<strong>8kHzは傾斜の薄端側に集中し、
10kHzは両端(厚端の管共鳴スカート+薄端の広帯域透過)に分布</strong>する —
「どの周波数がどの場所を照らすか」の系統的マッピングは楔構造そのものの性質で、
潰しでは消えない。これを直接測る指標(帯域間レベル差の空間ばらつき)では:</p>
<table>
<tr><th>スペクトル形状の空間ばらつき</th><th>Z1R</th><th>LCD</th><th>DX</th><th>DCA実測</th><th>DCA交互</th></tr>
<tr><td>σ_probe(10kHz帯 − 8kHz帯) [dB]</td><td class='num'>1.1</td><td class='num'>1.2</td><td class='num'>1.1</td><td class='num'><strong>4.1</strong></td><td class='num'>4.5</td></tr>
<tr><td>8k/10kマップ相関 r</td><td class='num'>0.98</td><td class='num'>0.87</td><td class='num'>0.98</td><td class='num'><strong>0.55</strong></td><td class='num'>0.59</td></tr>
</table>
<p>DCAはピンナ各所に届くスペクトルの「形」が他モデルの約4倍ばらつき、実測潰しでもほぼ改善しない。
振幅込み波形一致(広帯域窓相関)はこの帯域再配分に鈍感なため0.515まで回復して見えるが、
ピンナキュー(帯域ごとの部位重み付け)にとってはこの残存構造がDCA固有のコストであり続ける。
なお改善分も格子ピッチ4.5→3.2mm・開口率増と同時更新の複合効果。以下の比較は実測マップ版を用いる。</p>

<h2>2. 入射波面の品質(音源の責任範囲)</h2>
<figure><img src="data:image/png;base64,{imgs['radar']}" alt="レーダーチャート"></figure>
<table>
<tr><th>入射波面(1〜12.5kHz)</th><th>Z1R</th><th>LCD</th><th>DX10000CL</th><th>DCA(AMTS)</th></tr>
{table("incident")}
</table>
<table>
<tr><th>入射波面・空間キュー核心帯(5〜10kHz)</th><th>Z1R</th><th>LCD</th><th>DX10000CL</th><th>DCA(AMTS)</th></tr>
{table("incident_core")}
</table>
<p><strong>帯域制限後の構図(1〜12.5kHz):</strong>
<strong>LCD 0.889 と Z1R 0.882 がほぼ同着首位</strong>、DX 0.803。
16kHz帯を含めた旧集計ではZ1Rが0.783まで沈んでいたが、そのペナルティ
(2稜線干渉の高域ビーミング)は空間聴覚がほぼ使わない帯域に集中していた。
<strong>空間キュー核心の5〜10kHzでは入射はZ1Rが首位(0.904; LCD 0.894、レベルσはLCD 1.9が最良)</strong>
— 大振動板の面が本来効く帯域で効いている。DXは時間整列(11µs)最強だが周縁ドロップ(−14.3dB)が重い。
<strong>DCA(AMTS実測)は0.812(核心帯0.827)</strong>: AMTSインサートは往路の波面にも作用し、
波形形状一致は最下位(0.881)・入射拡散度は最大(0.074)。一方オンセット整列は12µsでDX(11µs)に並ぶ —
先頭波面は綺麗に揃うが、後続にレゾネータ散乱の尾を引く、という構図。
検証: Z1Rの周縁レベル低下は理想テーパーピストンのRayleigh積分予測と1.5dB以内で一致
(ピストン縁の古典的−6dB効果 = 有限開口の物理)。</p>
<figure><img src="data:image/png;base64,{imgs['lateral']}" alt="横方向プロファイル">
<figcaption>ピンナ中心(外耳道軸)から周縁への入射波形類似度。中心付近は全モデルほぼ1.0、
周縁での保ち方に音源の個性が出る。</figcaption></figure>

<h2>3. 音量の空間分布</h2>
<figure><img src="data:image/png;base64,{imgs['levels']}" alt="音量分布マップ">
<figcaption>帯域別レベル分布(各モデル平均比、±9dBスケール)。
<strong>LCD</strong>は広帯域で最も均一(σ1.8dB、8kHz帯σ1.4dB)だが、
開口短辺の外にあたる後方ピンナ領域に−6〜−9dBの段差。
<strong>Z1R</strong>も8kHz帯σ1.5dBと均一で、高域は中央やや優勢。
<strong>DX10000CL</strong>は全帯域で緩やかな中央集中(10kHz帯σ4.9dB)。
<strong>DCA(AMTS実測)</strong>は広帯域では均一(σ3.1dB)だが、傾斜レゾネータ列が
位置依存の周波数フィルタとして働く: <strong>8kHz帯は薄端側に集中(厚端はλ/4潰しの吸収帯 5.7〜7.7kHzのスカート
+管共鳴11.4kHzの谷間で沈む)、10kHz帯は両端に分布(厚端=11.4kHz共鳴スカート、薄端=短管の広帯域透過)して
中央が沈む</strong>。σは4.8dB(交互配置5.5dB)と僅かに改善するのみで、
帯域が上がるほどむらが増す唯一のモデルであることは変わらない。</figcaption>
</figure>

<h3>全プローブの周波数応答の重ね描き — 「揃い方」の3類型</h3>
<figure><img src="data:image/png;base64,{imgs['fr_overlay']}" alt="周波数応答重ね描き">
<figcaption>全プローブの入射周波数応答(直接波窓、1/6oct平滑、色=駆動軸からの横距離)。
<strong>上段</strong>はレベル込みの偏差、<strong>下段</strong>は各プローブの平均レベルを除いた
「スペクトル形状」だけの偏差。</figcaption></figure>
<p>3つの類型がはっきり分かれる。<strong>Z1R/LCD</strong>は上下段とも束が細い
(形状σ 0.8 / 1.5dB) — 同じ波形が僅かなレベル差で届く。
<strong>DX</strong>は上段で±7dBに大きく扇形に開くが、色順(中心→周縁)に整然と並び、
下段ではσ1.3dBまで潰れる — <strong>拡がりはほぼ純粋なレベル差で、スペクトル形状は保存</strong>
されている(小口径ビームの幾何減衰。相似波形が上下に分布する、という描像そのもの)。
<strong>DCA</strong>は上段のσ4.1dBがDX(4.4dB)と同程度に見えるが中身が違う:
色順に並ばず曲線同士が交差し、<strong>下段でもσ2.7dBと束にならない — 形そのものが場所ごとに違う</strong>。
レベル差は頭が「距離・方向の手がかり」として解釈しうる自然な勾配だが、
形の違いは場所ごとに別のイコライザを通した状態であり、ピンナキューの帯域重み付けを
直接乱す。スペクトル形状の空間ばらつき(前節)と同じ結論を、生の応答の束で確認できる。</p>

<h3>局所均一性の2軸分解 — 「系統的な勾配」と「乱れ」</h3>
<p>各プローブの近傍7mmで場を平面フィットし、<strong>フィットされた勾配の大きさ
(=近傍との系統的な差分の量。滑らかな減衰なら大きくても規則的)</strong>と
<strong>フィット残差(=平面で説明できない局所的な乱れ)</strong>に分解した:</p>
<figure><img src="data:image/png;base64,{imgs['local_uni']}" alt="局所均一性2軸">
<figcaption>左: 広帯域レベル。右: スペクトル形状(5〜10kHz、プローブごとのレベルオフセット除去後)。
右下=理想(差分も乱れも小)。</figcaption></figure>
<p>レベル(左)ではDXとDCAが同じ位置(勾配0.41dB/mm)に来るが、
<strong>形状(右)で二者は分離する</strong>: DXは形状勾配0.14dB/mm・残差0.05dBと
Z1Rと同じ「原点近く」に戻る — DXの急峻さは<strong>形を保った系統的レベル減衰</strong>であり、
近傍同士は常に相似波形。LCDはレベルこそ最も平坦(0.25dB/mm)だが形状の乱れは
Z1R/DXの2倍(残差0.10dB) — 開口短辺の影の縁で形が変わる。
DCAは形状の勾配0.48dB/mm・残差0.14dBとも突出し、<strong>系統的にも局所的にも
「隣と違うスペクトル」</strong>が届いている。まとめ: Z1R=両方小、DX=レベル差はあるが形は揃う、
LCD=レベル平坦だが形に局所乱れ、DCA=両方大。</p>

<h2>4. ピンナ反射 — 空間キューの生成確認</h2>
<figure><img src="data:image/png;base64,{imgs['tf']}" alt="ピンナ伝達関数">
<figcaption>ピンナ伝達関数(ピンナ有り/入射波面。同一プローブ・同一音源なので前面構造の影響は相殺)。
<strong>コンカ共鳴ピーク +20〜30dB @ 5kHz前後</strong>と<strong>8〜17kHzのノッチ群</strong>が
3モデルとも明確に形成される。8kHz以下は音源によらずほぼ共通(キューは頑健)、
10kHz超のノッチの深さ・位置は音源の照らし方で変調される —
音源設計が空間キューへ影響するのはこの帯域。</figcaption>
</figure>

<h3>前面構造⇄ピンナのカップリング反射(コーム)— 分離定量</h3>
<p>モデル間でTFの極(6〜8kHzのピーク/ノッチ)が食い違う理由を、
<strong>前面構造を全て取り除いた「裸ソース」対照ラン</strong>との比 C(f) = 構造ありTF/裸TF
で分離した。C(f)が「前面反射だけが作るコーム」そのもの:</p>
<figure><img src="data:image/png;base64,{imgs['coupling']}" alt="カップリングコーム">
<figcaption>外耳道でのカップリング成分。空間キュー帯(5〜10kHz)での振れ幅:
<strong>LCD 22.2dB(最深ノッチ−7.9dB@6.4kHz)</strong> > Z1R 15.9dB(@5.6kHz) >
DX 14.6dB(@9.6kHz) > <strong>DCA 11.9dB(@7.2kHz、最小)</strong> —
DCA以外はおおむね前面の固体率×近接度の順。
ただしこの序列は<strong>pp1固有</strong>: 8被験者平均では
Z1R 20.1 ≈ DCA 19.9 > DX 17.8 > LCD 16.5 dBとほぼ丸ごと入れ替わる(セクション6)。</figcaption>
</figure>
<p><strong>2つの寄与の分離結果</strong>:
(1) <strong>入射幾何</strong>(裸でも残る差): 平面波(LCD)と曲面波(Z1R/DX)では
コンカの励起が違い、共鳴の鋭さ・ノッチ位置が元々ずれる — 実耳で音源方向により
ノッチが動くのと同じ物理。
(2) <strong>カップリング反射</strong>: ピンナ⇄前面の往復干渉(往復30〜60mm→櫛周期6〜11kHz)が
キュー帯域に<strong>15〜22dBの変調</strong>を追加する。開口率32%のLCDマグネット面が最強の
反射板で、6.4kHzノッチはこれが作る。
<strong>AMTSの検証</strong>: DCAの前面はピンナ先端からわずか3mmと4モデル中最接近にも関わらず、
コームは<strong>11.9dBで4モデル最小</strong>(LCD比−10.3dB) — λ/4側枝の吸収と散点配置による
反射の非コヒーレント化という、AMTSの狙いがそのまま数字に出ている。しかも本モデルは剛体で
熱粘性損失を含まないため、<strong>実機の吸収はこれよりさらに強いはず</strong>(本結果は保守側)。振れ幅がピンナ固有のノッチ構造と同程度に大きい —
つまり<strong>ヘッドホンの空間キューは「ピンナ半分・前面反射半分」で形成される</strong>。
実機でも起きる「ヘッドホン・耳カップリング」現象(ヘッドホンFRが自由音場HRTFと一致しない
既知の理由)であり、前面の開口率・距離は音色だけでなく空間キューの設計変数でもある。</p>

<h2>5. ピンナ通過後(実聴取条件)の総合指標</h2>
<table>
<tr><th>ピンナ有り(1〜12.5kHz)</th><th>Z1R</th><th>LCD</th><th>DX10000CL</th><th>DCA(AMTS)</th></tr>
{table("pinna")}
</table>
<table>
<tr><th>ピンナ有り・空間キュー核心帯(5〜10kHz)</th><th>Z1R</th><th>LCD</th><th>DX10000CL</th><th>DCA(AMTS)</th></tr>
{table("pinna_core")}
</table>
<p>ピンナ散乱が支配的になり差は圧縮される。1〜12.5kHzでは
<strong>DX 0.506 > LCD 0.473 ≈ Z1R 0.460 > DCA 0.438</strong>、
5〜10kHz核心帯では<strong>DX 0.524 ≈ LCD 0.517 ≈ DCA 0.515 > Z1R 0.492</strong> —
核心帯の上位は数字上は三つ巴。ただしDCAの0.515は割引いて読む必要がある:
広帯域窓相関ベースの波形一致は帯域再配分(スペクトル形状の空間ばらつき σ(10k−8k)=4.1dB、
他モデルの約4倍)に鈍感で、DCA固有の「場所ごとに違うスペクトル」のコストを
十分に罰していない。方向純度はDX(0.439)とDCA核心帯(0.440)が並び、時間整列はDX(20µs)首位。
理想化交互配置(核心帯0.395)→実測マップ(0.515)の+0.12は、
前面設計の細部(どのセルを塞ぐか)が総量指標を大きく動かすことの実証。</p>

<h2>6. 個人差ロバスト性 — 8被験者パネル(4モデル)</h2>
<p>ここまでの比較は全てHUTUBS被験者1(pp1)の耳に対するもの。しかしカップリングコームの
ノッチ周波数は往復経路差の物理(f ≈ c(2n+1)/2Δ)で決まり、Δは「ピンナ各点⇄前面」の距離 —
コンカ深さ・耳介突出量の個人差はまさにこのmmオーダーで分布する。そこで
<strong>HUTUBSから追加13被験者の頭部メッシュを取得し、形状が最もばらける7名+pp1の
8被験者パネル</strong>で4モデル(Z1R/LCD/DX/DCA実測)×構造あり/裸ソース×incident/pinnaの全128ランを回した
(パネル選定は外耳道奥行き・コンカ深さ・耳介寸法の正規化特徴空間でのgreedy max-min)。</p>
<p><strong>パイプラインの頑健性(前提検証):</strong> 外耳道検出(干渉軸最近傍)は
検証した14被験者全員で軸から0.5mm以内に収まり、プローブ選択(突出マスク+コンカ円板+
視線判定)も全員で245〜273点を確保 — pp1向けに調整した手法は修正なしで全被験者に成立した。</p>
<table>{panel_table()}</table>
<figure><img src="data:image/png;base64,{imgs['subjects']}" alt="被験者パネル">
<figcaption>検証した14被験者のプローブ配置(外耳道基準)。+ = 外耳道入口、色 = 先端からの奥行き。
外耳道の奥行きは19〜36mmと約2倍の開きがある。</figcaption></figure>

<h3>コームは「ヘッドホン×耳」の共同プロパティ — 周波数も、強度も</h3>
<figure><img src="data:image/png;base64,{imgs['subj_comb']}" alt="コーム個人間分散">
<figcaption>同一モデルのC(f)を8被験者で重ね描き。ノッチ周波数は人ごとに5.6〜10kHzへ散り、
「同じヘッドホンでも人ごとに別のコーム」になることが直接確認できる。</figcaption></figure>
<table>
<tr><th>コーム(5〜10kHz、8被験者)</th><th>Z1R(開口71%)</th><th>LCD(開口32%)</th><th>DX(開口52%)</th><th>DCA(AMTS実測)</th></tr>
{comb_table()}
</table>
<p><strong>予想通りだった点:</strong> ノッチ周波数は被験者間で5.6〜10kHzに分散し
(モデル内で最大4kHz幅)、コーム<strong>周波数</strong>が個人の耳形状で決まることは完全に確認された。
ハードウェアノッチが本人のピンナノッチと衝突(1/12オクターブ以内=キュー消失リスク)する率は
22〜38%で、残りはピンナノッチの隙間に落ちて<strong>偽キュー</strong>になる —
どのモデルも「誰の耳でも安全」ではなく、当たり外れは確率的に起きる。</p>
<p><strong>予想を覆した点:</strong> コーム<strong>強度</strong>は「前面固体率×近接度で
ヘッドホン側が決める」と仮定していたが、これは成り立たない。
pp1でのLCD 22.2dBは<strong>8被験者中の最大値</strong>(レンジ12.2〜22.2dB)、
逆にpp1でのDCA 11.9dBは<strong>ほぼ最小値</strong>(レンジ11.7〜37.0dB)で、
被験者平均では<strong>Z1R 20.1±3.8 ≈ DCA 19.9±7.9 > DX 17.8±5.0 > LCD 16.5±3.2 dB</strong>と
pp1の序列(LCD > Z1R > DX > DCA)がほぼ丸ごと入れ替わった —
<strong>pp1はたまたまLCDに最悪・DCAに最良の耳だった</strong>。モデル内の被験者間σ(3.2〜7.9dB)は
モデル間の平均差(3.6dB)を上回り、<strong>強度もまた共同プロパティ</strong>。
とりわけ<strong>DCAのσ7.9dBは4モデル最大</strong>で、pp82(外耳道奥行き36mmのパネル最深耳)では
振れ幅37dB・最深ノッチ−28.9dB@8kHzという壊滅的なコームになる。
物理的に筋は通っている: AMTSのλ/4側枝・散点配置による反射抑制は<strong>位相設計された
干渉ベースの機構</strong>であり、想定の幾何(反射経路)から外れた耳では抑制が外れて逆に共振する。
抵抗性の吸収(実機に加わる熱粘性損失)は位相に依存しないため、実機ではこの外れ値は
剛体モデルより緩和されるはず — 剛体近似はDCAの個人間分散を過大評価する側に働く。</p>

<h3>ポスト・ピンナ指標の被験者間分散 — ロバスト性ランキング</h3>
<figure><img src="data:image/png;base64,{imgs['subj_var']}" alt="指標の被験者間分散"></figure>
<table>
<tr><th>ピンナ有り・核心帯(5〜10kHz、8被験者)</th><th>Z1R</th><th>LCD</th><th>DX10000CL</th><th>DCA(AMTS)</th></tr>
{variance_table()}
</table>
<p>被験者平均で見ると<strong>LCDの核心帯波形一致0.541±0.029が明確な首位</strong>
(8人全員が0.505以上 — pp1比較での「DX/DCAと同着」はpp1がLCDに不利な耳だった)。
<strong>品質の一貫性(被験者間σの小ささ)ではDXが首位</strong>(σ0.022、
到達時間ばらつきも27±3µsと圧倒的に一貫)— 小口径ドームの中央集中ビームは
レベルむらという固定費を払う代わりに、誰の耳に対しても同じように振る舞う。
<strong>Z1R(0.487±0.037)は平均最下位</strong>、<strong>DCA(0.515±0.047)はσ最大</strong>:
DCAは耳との相性で0.470〜0.620まで振れる「当たり外れ」の大きいモデルで、
コームの個人間分散(σ7.9dB)がそのまま総合指標に波及している。</p>
<p><strong>仮説検証(「コームが強い設計ほど個人差に敏感」):</strong> 支持が強まった。
コーム振れ幅の上位2モデル(Z1R 20.1 / DCA 19.9dB)がそのままσの上位2
(DCA 0.047 / Z1R 0.037)を占める(スピアマンρ=0.6、n=4)。
中位のLCD/DXは入れ替わるので単調な予測子とまでは言えないが、
「強い(または大きくばらつく)コームを持ち込む設計は、届く品質も人次第になる」
という因果は4モデルで一貫している。</p>

<h3>個人ピンナ署名の保存 — 透明ドライバ基準との比較</h3>
<p>ここで解釈上の注意: 上の被験者間σは「品質の一貫性」であって、応答そのものは
<strong>耳ごとに違うのが正しい状態</strong>(むしろ全員同じ応答になる設計は個人キューを洗い流している)。
「正しく個人化されているか」を測るには基準が要る。そこで<strong>透明ドライバ基準 TF_ideal</strong> =
全モデル共通の小型ソース(10mmピストン、バッフル・構造なし、固定距離20mm)による各被験者の
ピンナ応答を追加ランで取得し、これに対する忠実度を測った。
ただし小口径の実ドライバ(DX)は口径だけで透明基準に似てしまう交絡があるため、<strong>分解</strong>する:
<strong>P_drv</strong> = corr(裸ドライバ, 透明基準) にドライバ形状・距離の寄与を隔離し、
<strong>P_str</strong> = corr(構造あり, 裸) は同一ドライバ同士の比較なので<strong>口径交絡なしの
構造だけの効果</strong>、<strong>P_tot</strong> = corr(構造あり, 透明基準) が製品全体。
さらに各TFからパネル平均を引いた<strong>個人署名</strong>(その耳をその耳たらしめている部分 —
コンカ共鳴のような万人共通成分は生TFの相関を底上げするが個人情報を運ばない)同士の相関
<strong>P_sig_tot</strong> = corr(署名 構造あり, 署名 透明基準) を本命指標とする:
<strong>透明ドライバなら届いていたはずの個人由来ピーク・ノッチを、その製品は届けるか</strong>。</p>
<figure><img src="data:image/png;base64,{imgs['preservation']}" alt="個人ピンナ署名の保存">
<figcaption>左上: 保存の3分解(薄=ドライバ由来、中=構造由来、濃=製品全体)。
右上: 個人署名の保存 P_sig_tot(点=被験者)。下段: 署名の実例 —
破線(透明基準=その人固有のピンナ特徴)を実線(製品)がどれだけなぞれているか。</figcaption></figure>
<table>
<tr><th>保存係数(8被験者 平均±σ)</th><th>Z1R</th><th>LCD</th><th>DX10000CL</th><th>DCA(AMTS)</th></tr>
{preservation_table()}
</table>
<p>分解した各層で順位が入れ替わる。
(1) <strong>構造由来(口径交絡なし)</strong>: P_str は LCD 0.747 > DX 0.732 > Z1R 0.652 > DCA 0.627、
署名版 P_sig_str は <strong>LCD/DX 0.697 ≫ Z1R 0.563 > DCA 0.448</strong> —
前節の通り耳ごとに別物になるAMTSのコームは、<strong>個人署名の保存では構造として最下位</strong>。
(2) <strong>ドライバ由来</strong>: P_drv は DCA 0.772 が最大だが、これは面形状そのものより
21mmスタックが強制する<strong>遠距離配置(24mm — 透明基準の20mmに最も近い)</strong>の寄与が大きい。
基準距離を近く(例: DXの9mm相当)に置けばDXが有利になる類の項であり、
この距離・口径依存性を構造の項に混ぜないために分解した。
(3) <strong>製品全体の個人署名保存 P_sig_tot</strong>:
<strong>DCA 0.469 > DX 0.377 > LCD 0.361 > Z1R 0.307</strong> —
構造で最も失い、配置で最も得るDCAが総合では首位、という皮肉な構図
(「遠くに置ける」こと自体が音響的な資産)。
帯域を空間キュー核心(5〜10kHz — コームがまさに住む帯域)に絞ると順位はまた動き、
<strong>LCD 0.382 > DCA 0.335 > DX 0.292 ≫ Z1R 0.137</strong>: Z1Rの署名破壊はこの帯域に集中している。
ただし絶対値は重い: <strong>4モデルとも0.5未満、被験者σ±0.23〜0.30</strong>。
どの製品も「透明ドライバなら届いていたはずの個人由来ピーク・ノッチ」の半分程度しか
再現できておらず、Z1Rでは署名が<strong>反転</strong>する被験者(r=−0.29)すらある。
ヘッドホンの近接場では個人のピンナ署名は前面反射と照射幾何でここまで書き換えられる —
空間オーディオで個人HRTF測定・個人化EQが効く理由を、そのまま裏づける数字でもある。</p>

<h2>7. 再装着感度 — 駆動距離±・上下±のズレ(pp1、4モデル)</h2>
<p class="lede">「掛け直すと音像が変わる」を定量する: pp1に対し、駆動距離−1mm(パッド圧縮の内側限界)/
+2mm(緩い装着)と、上下±2mm(高め/低めの装着 — ピンナ・外耳道・プローブごと移動、ドライバ固定)の
4摂動×4モデル、各摂動で構造あり/裸×incident/pinnaの計64ランを基準セットと同一プロトコルで実行した。
DCAはAMTSの傾斜軸が上下(y)なので、上下ズレは透過マップの並進も同時に検査する。</p>
<figure><img src="data:image/png;base64,{imgs['reseat']}" alt="再装着感度"></figure>
<table>
<tr><th>再装着摂動(基準比)</th><th>Z1R</th><th>LCD</th><th>DX10000CL</th><th>DCA(AMTS)</th></tr>
{reseat_table()}
</table>
<p><strong>コームは軸方向ズレで数百Hz/mm動く。</strong> 最深ノッチの追跡はコームの歯の間を
飛び移る(tooth-hopping)ため、C(f)全体のlog-f相互相関でパターンの平行移動を推定した:
近づくと上方、離れると下方に移動し、レートは<strong>LCD 70 / Z1R 145 / DX 177 /
DCA 283 Hz/mm</strong>(軸+2mmではDCAは−347Hz/mm) — 往復経路差の物理(f·ΔΔ/Δ ≈ 数百Hz/mm)の
予測通りで、±1〜2mmの再装着でノッチが数百Hz〜1kHz級動く。
「掛け直すたびに音が違う」経験則のスケールと整合する。
一方<strong>上下ズレはコームを平行移動させない</strong>(≤71Hz/mm) —
代わりにコームの形そのものを変える(Z1Rは上下+2mmで振れ幅15.9→24.3dBに増大)。
経路差の集合が一様にでなく非対称に組み変わるためで、軸ズレ=移調、上下ズレ=再編成、と役割が分かれる。</p>
<p><strong>DCA最大変動仮説は棄却。</strong> 事前予想は「コーム移動+AMTS透過マップ並進の
二重機構でDCAの変動が4モデル最大」だったが、結果は逆:
コーム<strong>移動レート</strong>こそDCAが最大(283Hz/mm)で、軸+2mmのΔC RMS(5.8dB)もZ1R(5.7dB)と
最大タイだが、<strong>核心帯指標の変動幅はDCAが4モデル最小</strong>(0.023 vs DX 0.058)。
理由は2つ: (1) DCAのコームは元々浅い(11.9dB)ので、動いても外耳道スペクトルへの絶対的影響が小さい。
(2) 恐れられた透過マップ並進は±2mmでは実質検出されない — 10kHz照射マップの基準⇔ズレ相関は
全モデル・全条件でr≥0.99。傾斜による局所通過帯の移動(予測130〜200Hz/mm)は実在しても、
マップ自体が2mmスケールでは滑らかなため相関を壊さない。
逆に<strong>指標変動が最大なのはDX</strong>(0.058): 小口径ビームの中央集中は距離に敏感で、
軸+2mmで核心帯波形一致が0.524→0.478に落ちる。
まとめると、再装着で動くのは主に<strong>カップリングコーム(全モデルでΔC RMS 1〜6dB)</strong>であり、
その帯域は空間キュー帯そのもの — 装着のたびに違うコームに脳が再適応する、という描像が
4モデル共通に成り立つ。設計側の含意: コームを浅くする(DCA型のAMTS)ことは
個人差だけでなく再装着変動の絶対量も抑える。</p>

<h2>8. 結論 — 単独の勝者はいない、目的で選ぶ</h2>
<div class="callout">
<p><strong>LCD型(大型平面磁界)</strong>: 「音圧ごと同じ波形をピンナ全体に」という目標に最も近い
(入射・振幅込み0.889、レベルσ1.8dB、入射角7.9°)。実寸の厚いマグネット列は深いスロットが
コリメータとして働く。代償は開口率32%(写真実測)+厚さ10mm構造の音色影響 —
外耳道でのカップリング変調がpp1では5〜10kHzで<strong>22.2dBと4モデル中最大</strong>
(最深ノッチ−7.9dB@6.4kHz、櫛はヘルムホルツ共鳴ではなくピンナ⇄マグネット面の往復干渉が主因)—
と、ピンナ後の方向純度の低さ(拡散度0.579)。
ただし8被験者で均すとコームは平均16.5dBと3モデル中最小(pp1はLCDに最悪の耳だった)で、
<strong>核心帯ポスト・ピンナ0.541±0.029は被験者平均の明確な首位</strong>(セクション6)—
個人差込みで見るとLCDの評価はむしろ上がる。</p>
<p><strong>DX10000CL型(小径フルドーム)</strong>: 時間整列(11-20µs)・ピンナ後の方向純度(0.439)は最強。
ただし小口径ビームの中央集中により周縁の音圧が最も深く落ち(−14.3dB)、振幅込みでは3位。
「音像の輪郭・定位の鋭さ」を最優先し、レベルむらを受容する選択。
<strong>個人差ロバスト性は3モデル中首位</strong>(核心帯σ0.022、到達27±3µs、セクション6)—
中央集中ビームは誰の耳に対しても同じように振る舞う。</p>
<p><strong>DCA型(平面磁界+AMTS、実測潰しマップ)</strong>: メタマテリアルの狙いのうち
「反射の抑制」は定量的に確認できた — 前面がピンナに最接近(3mm)なのに<strong>カップリングコームは
11.9dBで4モデル最小</strong>(固定コーム重畳が最も少ない。剛体=無損失での値なので実機はさらに良い側)。
一方、楔構造固有の<strong>「帯域→場所」の系統的マッピングは実測潰しでも残る</strong>:
8kHzは薄端側、10kHzは両端、とピンナの違う場所を違う周波数が照らし、スペクトル形状の
空間ばらつきは他モデルの約4倍(σ(10k−8k)=4.1dB、8k/10kマップ相関0.55)。
帯域ごとの部位重み付けで成立するピンナキューにとって、これは総量指標に表れないコストであり、
21mm厚スタックの強制する距離42mmと合わせてDCA固有の弱点。
一方、懸念していた装着ズレ感度は<strong>むしろ4モデル最小</strong>と判明(±1〜2mm再装着での
核心帯指標変動0.023、透過マップ並進はr≥0.99で実質不検出 — セクション7):
浅いコームは「動いても被害が小さい」。
ただし「コーム最小」は<strong>pp1固有</strong>だった: 8被験者ではコームは11.7〜37.0dBと
4モデル中最大の個人間分散を示し(pp82では−28.9dBの壊滅的ノッチ)、位相設計された
干渉ベースの反射抑制は<strong>耳形状との相性が出る</strong>(セクション6。
実機の熱粘性損失は位相非依存なので外れ値は緩和されるはず — 剛体近似はDCAに厳しい側)。
一方で<strong>個人署名の総合保存 P_sig_tot 0.469 は4モデル首位</strong> —
構造(コーム)で最も失うのに、21mmスタックが強制する遠距離配置(24mm)が最も自然な照射を与えて
取り返す(「遠くに置ける」こと自体が資産)。
要するに<strong>装着再現性と署名の総合保存は最良、しかし個人間では最大の「当たり外れ」を持ち、
照射の帯域均一性(キュー寄り)は最下位</strong>という、良くも悪くもチューニングの効いたモデル。</p>
<p><strong>MDR-Z1R型(30mmドーム+ドーナツエッジ)</strong>: 空間キュー核心帯(5〜10kHz)の
<strong>入射品質で単独首位(0.904)</strong>、レベル均一・最悪ドロップも最良 —
大振動板は「効く帯域」で確かに効いている。2稜線干渉の代償は主に12.5kHz超に現れ、
空間聴覚への実害は小さい。ピンナ後の総合はLCD/DXに一歩譲る。
弱点は<strong>個人差への敏感さ</strong>(コーム平均20.1dB、核心帯平均も最下位0.487、セクション6)と
<strong>個人署名の保存の弱さ</strong>(P_sig_tot 0.307で4モデル最下位、署名が反転する被験者すらある)—
pp1で見えた入射品質の優位は、耳が変わると保てない。</p>
</div>
<p class="lede"><strong>ピンナ剛体近似の検証:</strong> 皮膚・軟骨の音響インピーダンス(≈1.5MRayl)は
空気の3600倍で反射率|R|≈0.999(吸収~0.1%/反射)— 剛体境界BEMのHRTF計算が実測と±1dBで一致する
のはこのため。感度実験として実皮膚の約60倍の吸音層(6%/反射)をピンナ表面に付けても
指標変化は類似度+0.05・入射角−5.7°・TF平均−1.5dB程度 → 実皮膚相当ではその1/60で無視可能。
ただし高Q共鳴ピーク(コンカ+25dB)は剛体でやや過大になる(毛髪・衣服の減衰も未モデル)。
その他の限界: セクション1〜5の詳細解剖は単一被験者(HUTUBS pp1)・左耳
(個人差はセクション6で8被験者×4モデルに拡張済み)。
全ラン最小距離・傾き0°、イヤパッド側壁なし(横方向は無響)。
振動板は理想運動(分割振動・歪みなし)。指標と知覚閾値の対応付けは未較正。
前後方向のドライバ角度付けは未検証。</p>

<h2>9. 再現</h2>
<p><code>configs/hutubs_70mm_z1r_v2.yaml</code> /
<code>configs/hutubs_90mm_planar_v2.yaml</code> /
<code>configs/hutubs_40mm_dome_v2.yaml</code> /
<code>configs/hutubs_dca_amts_real.yaml</code>(理想化交互配置: <code>hutubs_dca_amts.yaml</code>)。
入射波面プロトコルは <code>build_scene(config, incident_only=True)</code>。
ロバスト性スタディ(セクション6・7)は <code>scripts/robustness/</code>:
<code>verify_subjects</code> → <code>select_subjects</code> → <code>run_subjects</code> /
<code>run_reseat</code> → <code>fig_subject_comb</code> / <code>fig_subject_variance</code> /
<code>fig_reseat</code>。上下ズレは <code>PinnaSpec.offset_y</code>。</p>
<footer>headphone_sims — スタガード格子 圧力-速度 FDTD(PyTorch/GPU)。
検証: 自由音場1/r・バッフル付きピストン指向性(解析解±0.10)・気密性・エネルギー有界性。
ピンナ: HUTUBS (Brinkmann et al. 2019, CC BY 4.0)。</footer>
</main>
"""
out = Path("runs/matched_filter_results.html")
out.write_text(html, encoding="utf-8")
print(out, f"{out.stat().st_size/1e6:.1f} MB")
