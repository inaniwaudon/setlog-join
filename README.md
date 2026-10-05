# setlog

`assets/setlog-MMDD.mp4` を日付順に結合して `output.mp4` を作る。

## コマンド

| コマンド | 説明 |
| --- | --- |
| `python main.py` | クリップを結合して `output.mp4` を書き出す。 |
| `python main.py --blur` | `face.txt` に記載のセグメントを `boxblur` でぼかして結合。 |
| `python main.py --blur --test` | 各セグメントに `clip/n` ラベルを重畳（境界確認用）。 |
| `python main.py --face` | YuNet で各セグメントの顔を検出し `face.txt` を生成。 |
| `python main.py --boundary` | `face.txt` のぼかし対象セグメントについて、元クリップから境界フレーム±2 を横並び PNG で `boundaries/` に書き出す。シーン切替検出がフレーム単位で正しいかの目視確認用。 |

## 流れ

1. `--face` で顔検出して `face.txt` を作成
2. 必要に応じて手動で `face.txt` を編集
3. `--blur` で `output.mp4` を生成
4. `--boundary` で境界フレーム（±2）を書き出し、シーン切替位置がフレーム単位で正しいか目視確認。
