# `sim_core` v3 成对编舞偏好采集工具 (`scratch/simv3/prefs_tool/`)

本工具用于采集真实人类对“同一首歌、同一段 4–8 小节”两份不同谱面（官谱 vs 不同生成版本）的成对偏好标注，输出格式直接兼容 `sim_core.vocab_and_rarity.load_interest_weights()`。

## 1. 启动方式

### A. 本地网页版（含同步音频播放 + Simai 文本 + 双手/躯干 2.5D 轨迹动画 + 盲测模式）
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONPATH="."
.\.venv\Scripts\python.exe scratch/simv3/prefs_tool/server.py --port 8765
```
随后在浏览器打开 `http://127.0.0.1:8765`：
- 选择曲目、起始小节、窗口长度（4 / 6 / 8 小节）及候选来源 A/B（或直接按 `R` 随机抽取下一组）；
- 按 `Space` 播放/暂停当前 4–8 小节对应的 `track.mp3` 音频片段，左右画布实时同步渲染两份谱面的音符与模拟器双手/身体动作；
- 按 `1`（选 A）、`2`（一样/平局）、`3`（选 B）投票，结果自动追加写入 `scratch/simv3/prefs/user_prefs.jsonl`。

### B. 命令行交互版
```powershell
.\.venv\Scripts\python.exe scratch/simv3/prefs_tool/server.py --cli
```

## 2. 输出格式与权重自动拟合
每次投票写入 `scratch/simv3/prefs/user_prefs.jsonl`（每行一条 JSON，含 `chosen` 与 `rejected` 的 6 维有趣度特征字典及 `weight`）。当 `scratch/simv3/prefs/*.jsonl` 存在标注记录时，`sim_core.vocab_and_rarity.load_interest_weights()` 会自动使用正则化 Bradley-Terry 逻辑回归拟合并更新 6 项特征权重。
