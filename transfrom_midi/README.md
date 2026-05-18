# Guitar Audio To MIDI

这个目录里放的是一个轻量级的 `纯吉他音频 -> MIDI` 工具，目标是先把最小可用流程跑通。

它更适合下面这种输入：

- 干净的独奏吉他
- 单音旋律、分解和弦、简单句子
- 背景噪声不重

它不擅长：

- 多乐器混音
- 密集和弦的精确多音转写
- 大量滑音、击勾弦、强混响

## 运行方式

### WAV 输入

```powershell
.venv\Scripts\python.exe guitar\transcribe_to_midi.py --input path\to\guitar.wav
```

### MP3 输入

```powershell
.venv\Scripts\python.exe guitar\transcribe_to_midi.py --input path\to\guitar.mp3
```

默认会在输入文件旁边输出同名 `.mid` 文件。你也可以手动指定输出路径：

```powershell
.venv\Scripts\python.exe guitar\transcribe_to_midi.py `
  --input path\to\guitar.wav `
  --output guitar\outputs\demo.mid
```

## 可调参数

- `--sample-rate`：默认 `22050`
- `--frame-time`：默认 `0.02`
- `--min-note-duration`：默认 `0.08`
- `--bpm`：默认 `120`

例如：

```powershell
.venv\Scripts\python.exe guitar\transcribe_to_midi.py `
  --input path\to\guitar.wav `
  --min-note-duration 0.12 `
  --bpm 100
```

## MP3 支持说明

`wav` 会直接读取。`mp3` 会优先使用系统里的 `ffmpeg` 解码；如果本机环境已经具备可用的 TorchAudio 解码链路，也会尝试直接读取。

如果你运行 `mp3` 时看到“MP3 decoding is not available”，通常就是本机没有可用的解码器。这时最稳的做法是先安装 `ffmpeg`，或者先把 `mp3` 转成 `wav` 再跑。
