# setlog-join

setlog の動画を日付順に結合するスクリプトです．
人物の顔が映っているクリップ（2 秒の動画）を検出して，ぼかしを加えたり，削除したりすることもできます．

## セットアップ

ffmpeg がインストールされた環境が必要です（macOS の場合 `brew install ffmpeg`）．

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## コマンド

```bash
# assets/setlog-YYMM.mp4 を読み込んで，output.mp4 を生成
python main.py
```

| オプション | 説明 |
| --- | --- |
| （オプションなし） | 各ファイルを結合する |
| `--blur` | `face.txt` に記載のクリップにぼかしを加える．結合完了後，ぼかし境界の前後 ±2 フレーム（計 5 枚）を `boundaries/` に出力する |
| `--clip` | `face.txt` に記載のクリップを削除する |
| `--test` | 各セグメントに `filename/clipindex` ラベルを重畳する |
| `--face` | YuNet で各セグメントの顔を検出して `face.txt` を生成する |

## 使用方法

1. setlog アプリから日付ごとの動画を書き出す
2. assets/setlog-YYMM.mp4 に動画を配置する
3. スクリプトを実行する

### 顔検出してぼかしを加える場合

1. `--face` で顔検出して `face.txt` を作成する
2. 必要に応じて手動で `face.txt` を編集する
3. `--blur` オプション付きでスクリプトを実行する
4. `boundaries/` に境界フレームが出力されるので、適切にぼかしが掛かっているかを目視確認する

### 顔検出してクリップを加える場合

1. `--face` で顔検出して `face.txt` を作成する
2. 必要に応じて手動で `face.txt` を編集する
3. `--clip` オプション付きでスクリプトを実行する
