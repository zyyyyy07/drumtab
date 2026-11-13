# 鼓谱

给鼓手用的本地自动扒谱工具。上传一首歌，自动分离出各件乐器，每件乐器各出一份分谱，另有总谱；播放原曲时当前小节高亮、谱面自动滚动；所有谱子都能按 A4 打印或存成 PDF。

鼓谱是重点，其他乐器的谱子目前只是能看的初版。

## 能做什么

- **分轨**：鼓、贝斯、吉他、键盘、人声、其他乐器
- **出谱**：鼓谱（打击乐五线谱）、贝斯四线谱、吉他六线谱、键盘大谱表、人声和其他乐器五线谱，外加总谱
- **跟练**：小节高亮、自动滚动、点谱面跳转、减速播放（50%–100%，不变调）、可切换听原曲 / 去鼓伴奏 / 单件乐器
- **导出**：A4 打印 / PDF、Guitar Pro（.gp）、MIDI
- **校正**：小节线挪一拍、改拍号、速度减半或翻倍记谱、调鼓识别灵敏度，几秒出新谱，不用重新识别

## 安装与使用（Windows）

需要：Python 3.12、git、NVIDIA 显卡（没有显卡也能跑，只是慢很多）。

1. 双击 `安装.bat`：在本文件夹里建运行环境，并下载鼓识别模型。约 5 GB。
2. 双击 `启动.bat`：浏览器会打开 http://127.0.0.1:8765 。把歌拖进网页即可。
3. 用的时候不要关掉那个黑色窗口，用完再关。

第一次处理歌曲时会自动下载分轨和节拍模型（约 450 MB），存到本文件夹的 `models/` 里。之后所有文件都只写在本文件夹内。

一首 5 分钟的歌在 RTX 5060 笔记本上大约 1 分多钟处理完。

## 准确率

鼓识别在 MDB Drums 数据集（23 首带人工标注的真实录音）上实测，命中窗 ±50 毫秒：

| 底鼓 | 军鼓 | 踩镲 | 镲片 | 通鼓 | 整体 |
|---|---|---|---|---|---|
| 93% | 78% | 87% | 85% | 50% | 85% |

摇滚、流行、放克、金属类大多在 90% 以上；爵士（刷子、大量鬼音）明显偏低。军鼓失分主要在鬼音，通鼓是最弱的一类。

小节线、速度偶尔会判错（整体差一拍、按半速或倍速记），用网页上的“校正”修。

## 工作原理

| 步骤 | 使用的模型 / 方法 |
|---|---|
| 分轨 | Demucs `htdemucs_6s`，鼓轨再用 `htdemucs_ft` 的鼓专用模型提纯 |
| 节拍与小节线 | Beat This!（CPJKU，ISMIR 2024） |
| 鼓识别 | ADTOF Frame-RNN（PyTorch 版），分轨鼓轨与原曲的识别结果融合 |
| 鼓件细分 | 自写信号分析：开/闭镲、吊镲/叮叮镲、高/中/落地通鼓、鬼音 |
| 音高乐器 | Spotify basic-pitch（NeuralNote 用的同一个模型） |
| 记谱与显示 | 自写量化与记谱，alphaTab 渲染和播放同步 |

## 目录

```
app/        识别与记谱（Python）
web/        网页界面
dev/        评测与调试脚本（测试数据集需自行下载，见下）
启动.bat     启动
安装.bat     安装运行环境
```

以下目录不在仓库里：`.venv/`（运行环境）、`models/`（模型权重，首次运行自动下载）、`vendor/`（ADTOF，安装时自动下载）、`projects/`（你上传的歌和结果）。

## 开发与评测

`dev/` 里的评测脚本需要 [MDB Drums](https://github.com/CarlSouthall/MDBDrums) 数据集（CC BY-NC-SA 4.0），克隆到 `dev/testdata/MDBDrums`，然后运行：

```
.venv\Scripts\python.exe dev\t_cache.py     # 先跑一遍模型并缓存（约 2–3 分钟）
.venv\Scripts\python.exe dev\t_detect.py    # 5 类鼓件的准确率
.venv\Scripts\python.exe dev\t_refine.py    # 开/闭镲、吊镲/叮叮镲、通鼓、鬼音的细分混淆矩阵
.venv\Scripts\python.exe dev\t_grid.py      # 小节线与标注强拍的吻合度
```

## 第三方组件与许可

- [alphaTab](https://github.com/CoderLine/alphaTab)：MPL-2.0，`web/alphatab/` 内附说明
- Bravura 字体：SIL Open Font License 1.1
- [Demucs](https://github.com/facebookresearch/demucs)：MIT
- [Beat This!](https://github.com/CPJKU/beat_this)：MIT
- [basic-pitch](https://github.com/spotify/basic-pitch)：Apache-2.0
- [ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch)：未声明许可，因此不随本仓库分发，安装时从原作者仓库下载
