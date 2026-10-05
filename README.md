# mafuyu-2d

七瀬真冬の Live2D モデル作り。1枚のイラスト（`4_4.png`）からパーツを分けて、Cubism Editor で組み立てる。

| フォルダ | 内容 |
| --- | --- |
| `parts/` | 切り分けたパーツ画像（髪・目・口・腕など） |
| `tools/` | パーツの切り出しと PSD の書き出し（`python tools/build_all.py`） |
| `ai/` | 欠けた部分の補完（LaMa）や PSD 作成のスクリプト。`ai/tha3/`（talking-head-anime-3）と `ai/venv/` は大きいのでリポジトリには入れていない |
| `dist/` | Cubism 用の PSD と手順書（`Cubismセットアップ手順.md`） |
| `Cubism/` | Cubism Editor のモデルファイル（`.cmo3`） |
| `export/` | 書き出した moc3 一式（VTube Studio 用） |
| `web/` | ブラウザで動かす簡易プレビュー |

作業の経過は `作業メモ.md` にまとめている。
