# AutoAcoustics Pro

面向座椅电机与旋转机械问题分析的 **Windows 声学桌面程序**。

直接调用电脑麦克风、声卡或 USB 音频输入，观察波形、频谱与语谱图；停止录制后试听、选段，并结合转速、齿数及传动参数核对转频、阶次与候选机理。支持文件分析、校准、批处理、比较和报告。

**当前版本：0.3.1 · GPL-3.0-only · Python ≥3.12（已验证 3.12）**

0.3.1 改善设备刷新与重连的输入选择、连续录音试听、暂停恢复和播放错误提示；修复历史结果试听来源错配，以及诊断报告重复导出的线程清理问题。固定秒数录制在开始前核对实际采样率，不一致时提示调整后重试。

## 功能与边界

| 功能 | 当前实现 |
| --- | --- |
| 现场采集 | 真实输入设备/驱动枚举，采样率与通道选择，实时波形/频谱/滚动语谱图，WAV/HDF 保存，削波/丢样提示 |
| 文件导入 | WAV/FLAC、MP3/M4A/AAC/OGG、MP4/AVI/MKV 音轨、CSV/TXT/DAT、MAT、HDF5、TDMS |
| 声学分析 | FFT、Welch PSD、STFT、1/3 倍频程、A 计权、LAeq、LAFmax/LASmax、Zwicker 稳态/时变响度 |
| 转频与阶次 | 手动恒速参考，实测 `time_s,rpm` CSV，角域分析，1X/2X/啮合频率/电机相关频率参考与候选机理 |
| 工程工作流 | 校准来源、事件/工况、试听、A/B 比较、批处理、CSV/DOCX 报告、项目保存及源文件哈希核对 |
| 本地知识 | 本地资料/案例导入、检索、来源/版本记录；可选兼容 OpenAI Chat Completions 的问答配置 |
| NI | NI-DAQmx 后端；需要本机驱动及实际设备，真机验收待完成 |
| HEAD | 已停止录制文件的交接；HDF 解析支持已验证的 4/6、单通道、Intel FLOAT32、等速时间轴子集；官方受控采集待 SDK/许可与真机接入 |

电脑麦克风原始量为 **FS**，实时电平与频谱显示 **dBFS**。绝对 SPL 和响度需要对应当前麦克风、增益及录音设置的有效校准；未校准时不输出伪造的 Pa/sone。手动 RPM 是假设，候选机理需要改变转速、负载或测点复测。

不包含 EoL、ODS、PHM、企业管理、自动合格判定或电机运动控制。当前未宣称整机 IEC 等级、完整 ISO 认证、全部 HEAD 格式兼容或 NI/HEAD 真机计量链验收。

## 安装与启动（Windows）

安装 [Python 3.12](https://www.python.org/downloads/)，然后在 PowerShell 执行：

```powershell
git clone https://github.com/Shun1989/AutoAcoustics-Pro.git
cd AutoAcoustics-Pro
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -X utf8 launcher.py
```

`requirements.lock` 是已验证 Windows 环境的版本清单，包含开发依赖。其他平台或 Python 版本尚未作为正式交付验证。

WAV/FLAC 和电脑声卡不需要外部 FFmpeg。导入压缩音频或视频、运行全部格式测试以及本机打包时，需另行安装 [FFmpeg 与 ffprobe](https://ffmpeg.org/download.html)，并将其目录加入 `PATH`：

```powershell
ffmpeg -version
ffprobe -version
```

声卡依赖由 `sounddevice` 安装。请在 Windows 隐私设置中允许桌面应用使用麦克风。NI-DAQmx 驱动和 HEAD 软件/SDK 需从各自厂商取得。

## 第一次使用

1. 打开程序，点击“麦克风 / 声卡”，选择真实输入设备、接口、采样率和通道。点击“开始录制”，观察波形与语谱图。
2. 停止录制并等待落盘，点击“送入分析”。可试听、拖动选择启动/运行/停止等事件，填写负载、方向、测点与供电。
3. 分析前确认单位与校准来源。使用独立校准录音建立配置；没有校准时先做相对频谱分析。
4. 在“转频 / 阶次”填写恒速参考，或导入与音频同时间轴的 `time_s,rpm`。检查候选频率与观测证据，按建议复测。
5. 导出 Word/CSV 报告，或保存项目。移动录音会话时保留整个目录；移动项目时同时保留详情缓存。

也可以直接“导入录音”。多通道、科学数据变量和多音轨需要明确选择；程序不自动混音。

## 测试与本机打包

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest -q -ra
.\.venv\Scripts\python.exe -X utf8 tools/build_windows.py --name AutoAcousticsProLive
```

生成目录为 `dist/AutoAcousticsProLive/`，运行时保留整个目录。打包脚本收集依赖并生成许可库存；生成本机可运行包不等于完成公开二进制再分发条件。

仓库包含自行生成的格式/信号夹具及其生成脚本。真实录音、客户报告、工程资料原文/扫描页、私有校准配置及外部标准参考附件不随源码发布。缺少这些材料的测试会标记跳过，**不计为通过**。合法取得并恢复材料后，可用 `--require-external-data` 严格检查。

公共 Windows CI 检查安装、源码回归和源码 wheel。[公开源码验证记录](VALIDATION.md)登记本机结果和跳过范围。真实麦克风及硬件验收仍需在有设备的 Windows 机器上完成；现有本机验收与公共 CI 分开记录。

## 工程资料工具

使用自己的资料时，可安装可选依赖和 Poppler 的 `pdftoppm`，运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[corpus]"
.\.venv\Scripts\python.exe tools/build_knowledge_corpus.py --input-dir "D:\MyEngineeringLibrary" --output-dir output/knowledge_corpus
```

文本提取与页图保全会记录缺失及审阅状态，不自动代表全部内容已理解。原文件留在本机，问答只在用户点击“发送”后调用配置的 API。API 密钥保存为系统凭据或环境变量引用。

## 规格、许可与贡献

[原产品规格](../../REWRITE_PRODUCT_SPEC.md)保留作为历史基准，包含本版本未实现及明确排除的模块；当前功能范围以本文和 [架构与计算说明](ARCHITECTURE.md)为准。

项目原始源码采用 [GPL-3.0-only](../../LICENSE)。响度编排中的 MoSQITo 派生部分保留 Apache-2.0 归属，见 [NOTICE](../../NOTICE)与[第三方说明](../../THIRD_PARTY_NOTICES.md)。公开 Release 当前提供源码；Qt/FFmpeg 等依赖二进制的对应源码及发行材料尚未完整整理，因此不提供现有便携 EXE 的公开下载。

欢迎提交 [Issue](https://github.com/Shun1989/AutoAcoustics-Pro/issues) 或 Pull Request，详见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。
