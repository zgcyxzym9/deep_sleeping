# Guitar Audio To MIDI

这个目录里放的是一个轻量级的 `单乐器音频 -> MIDI` 工具，以及一个从.mid文件生成指定乐器音频的程序。可选乐器列表见文件末尾。

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

可选乐器名（General MIDI）：

```
acoustic_grand_piano
bright_acoustic_piano
electric_grand_piano
honky-tonk_piano
electric_piano_1
electric_piano_2
harpsichord
clavinet
celesta
glockenspiel
music_box
vibraphone
marimba
xylophone
tubular_bells
dulcimer
drawbar_organ
percussive_organ
rock_organ
church_organ
reed_organ
accordion
harmonica
tango_accordion
acoustic_guitar_nylon
acoustic_guitar_steel
electric_guitar_jazz
electric_guitar_clean
electric_guitar_muted
overdriven_guitar
distortion_guitar
guitar_harmonics
acoustic_bass
electric_bass_finger
electric_bass_pick
fretless_bass
slap_bass_1
slap_bass_2
synth_bass_1
synth_bass_2
violin
viola
cello
contrabass
tremolo_strings
pizzicato_strings
orchestral_harp
timpani
string_ensemble_1
string_ensemble_2
synth_strings_1
synth_strings_2
choir_aahs
voice_oohs
synth_voice
orchestra_hit
trumpet
trombone
tuba
muted_trumpet
french_horn
brass_section
synth_brass_1
synth_brass_2
soprano_sax
alto_sax
tenor_sax
baritone_sax
oboe
english_horn
bassoon
clarinet
piccolo
flute
recorder
pan_flute
blown_bottle
shakuhachi
whistle
ocarina
lead_1_square
lead_2_sawtooth
lead_3_calliope
lead_4_chiff
lead_5_charang
lead_6_voice
lead_7_fifths
lead_8_bass__lead
pad_1_new_age
pad_2_warm
pad_3_polysynth
pad_4_choir
pad_5_bowed
pad_6_metallic
pad_7_halo
pad_8_sweep
fx_1_rain
fx_2_soundtrack
fx_3_crystal
fx_4_atmosphere
fx_5_brightness
fx_6_goblins
fx_7_echoes
fx_8_sci-fi
sitar
banjo
shamisen
koto
kalimba
bagpipe
fiddle
shanai
tinkle_bell
agogo
steel_drums
woodblock
taiko_drum
melodic_tom
synth_drum
reverse_cymbal
guitar_fret_noise
breath_noise
seashore
bird_tweet
telephone_ring
helicopter
applause
gunshot
```
