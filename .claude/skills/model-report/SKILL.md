---
name: model-report
description: 4モデル音響比較レポート(Z1R/LCD/DX10000CL/DCA-AMTS)の再生成・更新・Artifact発行の手順。モデル追加やメトリクス変更後にレポートを最新化するときに使う。
---

# 4モデル音響比較レポートの生成と発行

実機構造モデル(MDR-Z1R / LCD / DX10000CL / DCA-AMTS実測)をHUTUBS pp1ピンナに対して
比較するレポート `runs/matched_filter_results.html` を再生成し、既存Artifactを更新する。

## パイプライン(全て**リポジトリルートから**実行 — 相対パス `runs/` 前提)

```bash
uv run python scripts/report/run_models.py   # 1) シミュレーション(GPU、既存npzはスキップ)
uv run python scripts/report/make_all.py     # 2) 全図表 + summary + レポートHTML
```

`run_models.py` は構造ありラン(batch3_*、速度場込み)と裸ソース対照ラン
(bare_*、カップリング分離用。フィルタ・バッフルなし、ドライバのみ依存)を生成。
強制再実行はnpzを削除する。`make_all.py` は fig_main → fig_coupling →
fig_fr_overlay → fig_local_uniformity → make_report の順に実行する。

## Artifact発行

**既存レポートの更新は必ず同じURLに対して行う**(新規作成しない):

- URL: `https://claude.ai/code/artifact/72b0a2b0-bae5-4698-9370-6084e508dc7a`
- Artifactツールで `file_path=runs/matched_filter_results.html` を発行。
  このURLを発行した会話以外からは **`url` パラメータに上記URLを渡す**(渡さないと別Artifactが出来る)
- favicon は `🔬` を維持(変更するとユーザーがタブを見失う)
- `label` に短い版名(例: "fr-overlay")を付ける

## 規約(レポートの一貫性を守るもの)

- **帯域方針**: 時間領域指標は1〜12.5kHz帯域制限(ゼロ位相)、空間キュー核心帯5〜10kHzを別掲
- **入射プロトコル**: `build_scene(incident_only=True)`(ピンナ固体のみ除去)+
  プローブごとの直接波窓 −0.15/+0.45ms。狭い窓は大口径を過小評価するので +0.45ms を変えない
- **指標v3**: 波形一致は振幅込み(2·xcorr/(Eₓ+Eᵧ))、到達はオンセット基準。
  広帯域窓相関は**帯域再配分に鈍感**なので、スペクトル形状の空間ばらつき
  (σ_probe(帯域間レベル差)、fig_local_uniformity の平面フィット分解)を併記する
- **カップリング分離**: C(f) = 構造ありTF ÷ 裸ソースTF(外耳道基準プローブ)。
  DCAは構造あり=`batch3_dca2`(実測潰しマップ)、裸=`bare_dca` を使う
  (fig_coupling.py の RUNKEY/BAREKEY 参照)

## モデルの追加・変更手順

1. `configs/` にシーンconfigを追加(既存v2 configの規約: tilt 0°、物理最小距離)
2. `scripts/report/run_models.py` の STRUCTURED / BARE にエントリ追加
3. `scripts/report/fig_main.py` の MODELS と CFGS、`fig_coupling.py` の
   MODELS/RUNKEY/BAREKEY、`fig_fr_overlay.py`・`fig_local_uniformity.py` の MODELS に追加
4. `scripts/report/make_report.py`: `ORDER` にキー追加、各テーブルヘッダに列追加、
   セクション1のモデル表に列追加、`imgs` に `geo_*` 追加、結論callountに段落追加
5. **本文の数値はスクリプトが自動反映しない**(テーブルは final_summary.json 由来で自動、
   地の文・結論・figcaptionの数値は手書き)。数値が変わったら本文も必ず更新する。
   スクリプトの出力する数値(コンソール)と本文を突き合わせること

## 落とし穴

- スクリプトを scratchpad 等から実行すると cwd が変わり `runs/` が解決できず落ちる —
  必ずリポジトリルートから
- 裸ラン(bare_*)には速度場がない。圧力のみの指標に限定するか、ゼロ埋めする
- レポートは自己完結HTML(画像はbase64インライン)。外部URL参照は artifact のCSPで死ぬ
- コード変更を伴う場合は main 直接pushせずブランチ+PR(CLAUDE.md)
