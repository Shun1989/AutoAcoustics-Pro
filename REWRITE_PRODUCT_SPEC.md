# AutoAcoustics Pro — 产品开发说明书（复刻规格）

> **文档性质**：这是一份「完整可复刻」的产品+技术规格书。目标读者是**第三方大模型 / 开发团队**——拿到本文档 + 少量真实样例音频，即可重建出功能等价的 AutoAcoustics Pro。
>
> **版本基准**：v0.28.0
> **文档作者**：基于对现有源码（369 个 .py 文件）的全量逆向调研整理
> **撰写日期**：2026-10-03
>
> **使用方式建议**：不要试图让模型"一口气写完"。建议按本文档的章节把任务切分为：①工程骨架 → ②声学算法核心 → ③数据层 → ④UI 框架 → ⑤各功能模块 → ⑥许可证/构建，逐块喂给模型实现并单测。

---

## 目录

1. 产品概述
2. 技术栈与依赖
3. 系统架构与目录结构
4. 启动流程（10 步）
5. 配置系统
6. 功能总览（5 大模式 + 15+ 子模块）
7. **声学核心算法规格**（最重要，含完整公式与校准参数）
8. 数据导入与校准规格
9. 各功能模块详细规格
10. 数据模型（数据库 Schema）
11. UI / UX 规格
12. 许可证与权限系统
13. AI 能力（LLM RAG / 向量库 / 诊断）
14. 构建、打包与部署
15. 外部系统依赖
16. 实现指引与降级策略
17. 附录（校准参数表 / 标准索引 / 文件清单）

---

## 1. 产品概述

### 1.1 定位
**AutoAcoustics Pro** 是一款面向 **NVH（噪声、振动与声振粗糙度）分析** 的 Windows 桌面专业软件，服务电机 / 电驱 / 传动系统的**故障诊断、声品质评估、健康预测与产线检测**。典型用户是声学工程师、EOL（End-of-Line）产线质检、故障诊断工程师。

### 1.2 核心价值
- **声学客观量计算**：SPL、1/3 倍频程、A 计权、Zwicker 响度、声品质五维指标（尖锐度/粗糙度/波动强度/突出率/SQI）
- **故障诊断**：不平衡等故障的规则诊断 + 基于声纹向量库的相似案例检索 + LLM RAG 专家报告
- **健康预测（PHM）**：健康指数融合 + 剩余寿命（RUL）曲线外推
- **工作变形分析（ODS）**：3D 电机几何的振型动画
- **产线自动化（EoL）**：Pass/Fail 判定、班次看板、批量处理
- **企业能力**：数据湖、多用户 RBAC、许可证分层、插件、开放 API

### 1.3 商业形态
采用**节点锁定许可证** + **四档付费分层**（Standard / Advanced / Custom / Ultimate），试用版 30 天全功能。硬件指纹绑定，支持离线激活。

---

## 2. 技术栈与依赖

| 类别 | 选型 | 说明 |
|---|---|---|
| 语言 | Python **≥ 3.10**（实测 3.13） | 项目基于 3.13 开发 |
| GUI | **PyQt6 ≥ 6.5** | 桌面应用主体 |
| 绘图 | **pyqtgraph**（主）+ matplotlib（报告内嵌） | FFT/语谱图/时间历程 |
| 3D | pyqtgraph.opengl（GLViewWidget） | ODS 3D 视图 |
| 数值 | numpy、scipy、**numba**（JIT 加速，arm64 除外）、pandas | 声学计算 |
| 音频 IO | **sounddevice**（实时采集/回放）、**SoundFile**、原生 `wave` 模块 | |
| 媒体转码 | **ffmpeg**（系统 PATH） | 视频/非 WAV 音频 → WAV |
| 数据 | **SQLAlchemy**（ORM）+ sqlite3；h5py（HDF5） | |
| ML/AI | **torch**（CPU 版）、**onnxruntime**、openai（LLM 客户端） | 全部可降级 |
| Web/API | FastAPI + uvicorn + pydantic | 开放 API（Ultimate） |
| 文档 | python-docx（Word 报告）、openpyxl | |
| 其他 | sentry-sdk（遥测）、pluggy（插件）、cryptography（模型加密） | |
| 打包 | **PyInstaller**（onedir）+ **Inno Setup 6** | |

**pyproject.toml 结构**（`pyproject.toml:39-137`）：
- `requires-python = ">=3.10"`
- 核心依赖见上表；`numba/torch/onnxruntime/openai` 在 arm64 平台被条件排除
- `[project.optional-dependencies]`：
  - `dev` = pytest 系列 / ruff / mypy / pyinstaller / mkdocs
  - `daq` = nidaqmx-python
  - `enterprise` = psycopg2-binary / ldap3 / redis
- `[project.scripts]`：`autoacoustics = launcher:main`
- 包发现：`where = ["src"]`
- `setuptools_scm`：`write_to` 生成 `_version.py`，`fallback_version = "0.28.0"`

**requirements.txt 与 pyproject 的差异**（复刻时以 pyproject 为准，另外补 requirements 独有项）：requirements 额外列了 `mosqito`（声品质备选库）、`vispy`（3D 备选）、`openpyxl`、`scikit-learn`、`reportlab`、`flask`。

---

## 3. 系统架构与目录结构

### 3.1 架构总览
```
launcher.py（生产入口，566 行）
   │
   ├─ sys.path 自举 → Platform 适配 → 崩溃钩子 → 日志 → 遥测
   ├─ create_application() → 主题 → OOBE 首启向导
   ├─ AppConfigManager（配置）→ initialize_factories()（DB/Auth/License）
   ├─ 认证（standalone 自动登录 / enterprise 登录框）
   ├─ 许可证校验 → 模型加密初始化
   └─ MainWindow(config).show() → app.exec()
```

### 3.2 顶层目录
```
项目根/
├─ launcher.py                  # 生产入口（566 行）
├─ config.yaml                  # 开发模板配置（130 行）
├─ pyproject.toml / requirements.txt
├─ installer.iss                # Inno Setup 打包脚本
├─ AutoAcoustics_Pro.spec       # PyInstaller 配置
├─ build_installer.py / .bat    # 打包脚本
├─ docs/  Reference/  Final/  For OpenCode/   # 文档
├─ plugins/                     # 插件（示例插件）
├─ tools/                       # 构建/加密/许可证签发工具
├─ tests/                       # 测试（393 KB）
└─ src/autoacoustics/           # 源码主包
```

### 3.3 `src/autoacoustics/` 子包职责

| 子包 | 职责 | 关键文件 |
|---|---|---|
| `core/` | 全部业务逻辑（约 45 个模块） | `dsp.py`、`psychoacoustics.py`、`sound_quality.py`、`dual_mode_engine.py`、`expert_system.py`、`case_database.py`、`phm_engine.py`、`ods_engine.py`、`data_lake_engine.py`、`license_engine.py`、`license_manager.py`、`auth_manager.py`、`factories.py`、`app_config.py`、`platform_compat.py`、`telemetry.py`、`model_encryption.py`、`plugin_manager.py`、`api_server.py`、`ota_updater.py`；子包 `acquisition/`（`realtime_recorder.py`、`file_importer.py`、`sensor_config.py`） |
| `ui/` | 全部 PyQt6 界面（28 个文件） | `main_window.py`（4503 行）、`cyber_industrial_theme.py`、`widgets.py`（1909 行）等 |
| `models/` | AI 模型注册/加载抽象（经 model_encryption 解密 .enc） | `__init__.py`（`get_model`/`list_available_models`） |
| `order_analysis/` | 阶次跟踪（转速脉冲→RPM→阶次切片） | `engine.py`、`dashboard.py` |
| `sensor_fusion/` | 多通道（声/振/电流）融合诊断+投票 | `engine.py`、`dashboard.py` |
| `plugins/` | 微内核插件系统 | `__init__.py` |
| `v2_engine/` | V2 物理信息 DSP：TSA 时域同步平均 + 传动链运动学 | `tsa_engine.py`、`kinematic_model.py` |
| `v2_expert_system/` | LLM RAG 专家报告 | `llm_rag_engine.py`、`report_generator.py` |
| `v2_knowledge/` | 声纹向量库 | `vector_db_manager.py`、`embedding_engine.py` |
| `utils/` | 通用 IO/日志/配置 | `io.py`（`read_audio`）、`logger.py`、`config.py` |

**依赖方向**：`ui → core → utils`；`v2_expert_system → v2_knowledge`；`v2_*` / `order_analysis` / `sensor_fusion` 均由 `ui.main_window` 直接 import，子包间基本不互相依赖（叶子包）。

---

## 4. 启动流程（10 步）

`launcher.py` 的 `main()`（L485-561）按序执行，复刻时必须保证顺序与降级策略：

1. **sys.path 自举**（L32-37）：把项目根 + `src/` 插入 `sys.path[0]`，未安装也能 `import autoacoustics`。**这是"运行不需要 pip install"的关键**。
2. **平台层**（L39-48）：在任何 PyQt import 前调用 `Platform.setup_high_dpi_scaling()` 和 `setup_qt_platform_plugin()`。
3. **崩溃钩子**（L53-62）：`core.app_logger.CrashHandler.install()`。
4. **日志**（L67-79）：`init_logging()`。
5. **遥测**（L329-379）：Sentry，DSN 取 `AA_SENTRY_DSN` 环境变量或 config，无 DSN 则本地模式。
6. **create_application()**（L94-114）：设 AppName="AutoAcoustics"/Org/版本，强制 PassThrough DPI 取整。
7. **主题**：`ui.cyber_industrial_theme.CyberThemeManager.apply(app)`（L503）。
8. **OOBE 首启向导**（L119-162）：`cfg.is_first_run` 时弹 `ui.oobe_wizard.OOBESetupWizard`，失败回退 standalone。
9. **配置 + 工厂**（L173）：`AppConfigManager.instance()` → `initialize_factories()`（DB/Auth/License 工厂）。
10. **认证 → 许可证 → 主窗口**（L190-305）：
    - 认证：standalone 自动登录 admin；enterprise 弹 `ui.login_dialog.LoginDialog`
    - 许可证：standalone 节点锁定（`lic_mgr.check_out`），硬失败关键词 expired/corrupt/invalid machine/blocked 才拦截；`AUTOACOUSTICS_DEMO` 环境变量可绕过
    - `_init_model_encryption()`（L384）：从硬件指纹+许可派生 AES-256 密钥
    - `MainWindow(config=cfg).show()` → `app.exec()` → `cleanup_on_exit`（遥测 flush、席位 check_in、登出、关工厂）

**注意**：无单实例锁、无 QSplashScreen。

---

## 5. 配置系统

### 5.1 配置搜索顺序
`core/app_config.py:162-172`：依次搜索 `Platform.get_config_dir()` → 当前目录 `.` → 上级目录 `..`。
- Windows 下 config dir = `%APPDATA%\AutoAcoustics\`（`platform_compat.py:198-211`）
- macOS = `~/Library/Application Support/AutoAcoustics/`
- Linux = `~/.local/share/autoacoustics/`

**关键**：项目根 `config.yaml` 是开发模板；**运行时实际读 `%APPDATA%\AutoAcoustics\config.yaml`**（优先级更高）。两者是两份文件，需注意。未找到配置 → `is_first_run=True`（`app_config.py:394`）触发 OOBE。

### 5.2 config.yaml 全量项
```yaml
app:           # 名称/版本/log_level
database:      # engine(sqlite/postgresql/mysql) + 连接池；密码支持 ${DB_PASSWORD} 环境变量插值
storage:       # local_path / nas_path
auth:          # local / ldap
license_server:  # standalone / floating（心跳 60s、超时 180s、三档席位 5/20/50、trial_days 30）
audit:         # 审计开关
ui:            # theme=cyber_industrial、语言 zh_CN、auto_backup 300s
daq:           # 48kHz、ring buffer 96000、backend sounddevice/nidaqmx/mock
reporting:     # docx 输出目录
```

### 5.3 平台适配（`platform_compat.py`，`Platform` 类 L46）
- OS 判定、`has_nidaqmx()` 仅 Windows（L134）
- `has_ffmpeg()` 走 PATH（L175）
- `get_user_data_dir()` L185、`get_temp_dir()`=temp/AutoAcousticsPro（L224）
- `setup_qt_platform_plugin`（cocoa/windows/xcb，L234）
- 音频转码 `get_audio_converter_command`：macOS 优先 afconvert，其余用 ffmpeg（L299-335）

---

## 6. 功能总览（5 大模式 + 15+ 子模块）

### 6.1 主窗口顶层结构
主窗口 `ui/main_window.py`（4503 行，`MainWindow(QMainWindow)` L159）顶层是 `_mode_tabs = QTabWidget()`（L312），包含 **5 个模式页**（L342-362）：

| 模式页 | 图标 | 用途 |
|---|---|---|
| 实验室研发 | 🔬 | 核心分析工作台（功能最全） |
| EoL 产线 | 🏭 | 产线自动化车间看板 |
| PHM | 📈 | 故障预测与健康管理 |
| ODS | 📊 | 工作变形分析 |
| 企业数据湖 | 🏢 | 企业级数据湖看板 |

**无传统菜单栏**（全部走面板 + 工具栏 + 快捷键）。面板可浮动（`_float_out` 弹独立 QMainWindow，L2270）。会话保存/恢复（`_save_session` L978 / `_restore_session` L1044）。

### 6.2 实验室研发页 — 右侧子页（`_right_tabs`，L657）
实验室页 `_build_lab_ui()`（L377）是左右 QSplitter，右侧 `_right_tabs` 含 **15+ 子页**（L663-818，后 5 个按 license tier 条件添加）：

| 子页 | 实现文件 | 功能 |
|---|---|---|
| 诊断报告 | `expert_panel.py` | AI 专家诊断输出 |
| 特征频率 | （main_window 内嵌 + `frequency_calculator.py`） | 轴承/电磁/齿轮特征频率标注 |
| 声品质 | （main_window + `sound_quality.py`） | 声品质五维指标 |
| SQI 雷达 | `sqi_radar.py` | SQI 雷达图 + 评分面板 |
| AI 专家 | `expert_panel.py` | 传动链建模 + AI 诊断 |
| 信号增强 | `signal_panel.py` | 降噪/滤波控制 |
| 知识库 | `knowledge_panel.py` | FMEA 知识库管理 |
| 数据采集 | `acquisition_panel.py` | 传感器采集 + 校准 |
| DAQ 采集 | `daq_panel.py` | 多通道 DAQ 硬件 + 实时示波器 |
| 故障标注 | `spectrogram_annotator.py` | 语谱图 ROI 标注 |
| 快速录入 | `quick_case_entry.py` | 一键从分析结果建案例 |
| 传动链建模 | `expert_panel.py` / `v2_engine/kinematic_model.py` | 传动链拓扑 + GMF 推导 |
| 声纹检索 | `match_display.py` | 相似案例向量匹配展示 |
| AI 专家报告 | `v2_expert_system/` | LLM RAG 五段式报告 |
| 阶次分析 | `order_analysis/` | 阶次跟踪 |
| 传感融合 | `sensor_fusion/` | 多通道融合诊断 |

### 6.3 其余模式页
- **EoL 产线**（`eol_dashboard.py` 768 行）：VerdictDisplayWidget（判定大字屏）、ShiftStatusPanel（班次状态）、HistoryListWidget（历史记录）
- **PHM**（`phm_dashboard.py` 1008 行）：TestNodeEditorDialog（测试节点编辑）、HI/RUL 曲线、健康分级看板
- **ODS**（`ods_dashboard.py` 299 行 + `ods_3d_view.py` 441 行）：测点选择、频率选择器（`ods_frequency_selector.py`）、3D 振型动画
- **企业数据湖**（`data_lake_dashboard.py` 1395 行）：KPI 卡片、分面搜索（FacetedSearchPanel）、趋势图

---

## 7. 声学核心算法规格（★ 最重要，含完整公式与校准参数）

> 这一章是复刻的灵魂。所有数值参数、阈值、校准数组都来自现有代码的**实际调优结果**，必须原样采用，否则计算结果会偏差数倍。**信号链统一约定**：单声道 float 信号，参考声压 `P_REF = 20e-6 Pa`，默认采样率 `fs = 48000 Hz`。

### 7.1 FFT / 单边幅度谱（`core/dsp.py:36-82`）
- `compute_fft(x, fs, window='hann')`：加窗后 rFFT，**幅度归一化 = `|rfft(x·w)| / Σw`**，线性平均，dB = `20·log10(amp + 1e-20)`
- 这是**幅值谱约定**（用于看图），与能量/响度计算的 Parseval 约定不同，注意区分
- `compute_stft`（:85-109）：`scipy.signal.stft`，hann 1024/512，返回 `|Zxx|`，用语谱图

### 7.2 1/3 倍频程带 RMS（三处实现：dual_mode_engine:177-217、psychoacoustics:139-189、sound_quality:80-118）
**统一标准，三处必须一致**：
- 中心频率：20 Hz – 20 kHz 共 **31 点**（IEC 61260-1）
- 截止条件：`fc > fs/2.5` 时跳过
- 带宽：`fc·2^(±1/6)`
- 流程：Hanning 窗 → rFFT → 取带内 bin → **单边 Parseval**：
  ```
  wc       = sqrt(N / Σw²)                    # 窗补偿因子
  band_rms = sqrt(2·Σ|X[k]|² / N²) · wc       # N=FFT 长度
  spl      = 20·log10(band_rms/20e-6 + 1e-20)
  ```

### 7.3 时域 SPL 与 A 计权
- 时域 SPL（`dsp.py:23`）：`Lp = 20·log10(RMS/20µPa)`，先 `remove_dc`（:31）
- **OASPL(A)**（`dual_mode_engine.py:326-335`）：1/3 倍频程带加权能量和
  ```
  OASPL(A) = 20·log10( sqrt( Σ_i (rms_i · 10^(Aw_i/20))² ) / P_REF )
  ```
  A 计权表 `A_WEIGHTING_DB`（:116-122，**IEC 61672-1**）。**仅实现 A 计权，无 C/Z**。

### 7.4 A 计权 IFFT 法（SPL 时间历程，`ui/widgets.py:1321-1402`）
对每帧信号用频域 A 计权再逆变换回时域：
1. 分帧 2048 / 跳 1024，Hanning 窗 + `wc`（:1361）
2. rFFT → 按 `_A_WEIGHTING_FREQS/DB`（:1307-1318）线性插值 → `X·10^(Aw/20)` → irfft → RMS → `p²(t)`
3. **IEC 61672 指数时间计权**：`α = 1 − exp(−dt/τ)`，Fast `τ_F=0.125s` / Slow `τ_S=1.0s`（:1353）；`y[n] = α·x[n] + (1−α)·y[n−1]`；`dBA = 10·log10(y / p₀²)`

### 7.5 Zwicker 响度（核心引擎，`core/psychoacoustics.py`）★ 全公式
这是最难复刻的部分，三要素**缺一不可**（否则结果偏差 10-100 倍）：

**① ISO 226:2003 频率相关听阈**（:99-111，31 点 20–20000 Hz）
```
freqs  = [20,25,31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,800,
          1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000,20000] Hz
TQ     = [78.5,68.7,59.5,51.1,44.0,37.5,31.5,26.5,22.1,17.9,14.4,11.4,9.0,7.0,6.0,
          5.0,4.4,4.0,3.4,2.9,2.5,2.2,2.1,2.1,2.5,3.4,5.1,6.5,9.5,13.1,28.0] dB
```
`iso226_hearing_threshold`（:114）做线性插值。**不能用固定 40dB 阈值**。

**② GF 自由场外耳传递修正**（:243-253，23 点，DIN 45646/ISO 11904）
```
f      = [20,31.5,50,80,125,200,315,500,800,1000,1250,1600,2000,2500,3150,
          4000,5000,6300,8000,10000,12500,16000,20000] Hz
GF_dB  = [0,-1,-2,-1,-0.5,0,1.5,2.5,4,6,9,12,15,17,18,17,16.5,14.5,10,7,4,-1,-5]
```
（v2 联合校准：5000-6300 Hz 区域 **+1.5 dB**，即上面数组的当前值）
`SPL_eff = SPL + GF_corr(f)`

**③ Bark 域激发与扩散**（:265-302）
- Bark 映射（Traunmüller 1990，:259）：
  `z = 13·atan(0.00076·f) + 3.5·atan((f/7500)²)`
  逆映射（dual_mode_engine:153）：`bark_to_hz(z) = 926.5·(z/(z+6.5))^1.03`
- 基底激发 = 1（听阈处）；阈上带 `E = 10^((SPL_eff − TQ)/10)`
- **非对称高斯扩散核**：下坡 `σ_low=1.5` Bark / 上坡 `σ_high=1.2` Bark（对应 ±27 dB/Bark 坡度），行归一化后卷积得到 `E_spread`

**④ Specific Loudness（Moore-Glasberg 1996，:308-311）** ★ 关键参数
```
N' = C · ( max(E_spread, 1)^α − 1 )
   C = 0.048   （v3 校准值，64 组 HEAD 数据批量优化，从 0.050 下调 4%）
   α = 0.23
```
- 插值到 **24 点均匀 Bark 网格 [0.5, 1.5, …, 23.5]**，**总响度 sone = Σ N'(z)**
- 入口：`compute_loudness`（:192）、`compute_zwicker_loudness_full`（:325）返回 `(sone, bark_grid, N')`
- 校准精度：64 组 HEAD 数据中 46 组在 ±10% 以内

### 7.6 声品质五维指标（`psychoacoustics.py`）
- **尖锐度**（Aures, DIN 45692，`compute_sharpness` :448）：
  `S = 0.11 · Σ(N'·g(z)·z) / ΣN'`，权重 `g(z) = 0.078·(0.123z/(0.123z+1))·(0.78z/(0.78z+1))·(z/(z+1))²`，单位 acum
- **粗糙度**（:492）：Hilbert 包络 → FFT → 20–300 Hz 调制带能量比 `sqrt(P_band/P_total)×2`，单位 asper
- **波动强度**（:540）：包络 4 阶 Butterworth 20 Hz 低通，`std_slow/std_total×1.5`，单位 vacil
- **突出率**（:592）：200 Hz–10 kHz 最高谱峰，邻域=临界带宽 `BWc = 25 + 75·(1+1.4(f/1000)²)^0.69` 内剔除峰 ±5%，`PR = 10·log10(E_peak/E_nb)`，单位 dB
- **SQI 五维雷达**（:695-774）：权重 `(响度0.30, 锐度0.25, 粗糙0.15, 波动0.15, 突出0.15)`；`NORM_REFS = 8 sone / 3 acum / 1.5 asper / 1.0 vacil / 20 dB`；每维归一到 0–10 分，`SQI = 100 − Σ w·penalty·100`
- 注：`sound_quality.py` 是旧简化版（固定 40dB 阈值 :143、权重 (0.4,0.25,0.15,0.1,0.1) :350），新代码以 `psychoacoustics.py` 为准

### 7.7 临界频带 / Bark 网格（`dual_mode_engine.py:168-170`）
- 24 个临界带中心：`arange(0.5, 24.5, 1)`，默认响度限值 `0.15 sone/Bark`
- `compute_loudness_mode`（:373-504）调 `compute_zwicker_loudness_full` 得 `N'(z)`，做带限/总响度/锐度判定

### 7.8 响度时间历程（`ui/widgets.py:1409-1567`）
逐帧：1/3 倍频程 SPL → GF 修正 → 激发扩散（σ=1.5/1.2）→ C=0.048/α=0.23 → 24 Bark 积分 → F/S 指数平滑（同 §7.4 的 IEC 61672 指数计权）

### 7.9 PHM 健康预测（`core/phm_engine.py`）
- **健康指数 HI 融合**（`compute_hi_score` :219-339）：
  ```
  权重 = {loudness:0.20, sharpness:0.15, roughness:0.18,
          fluctuation:0.12, prominence:0.10, sqi:0.25}
  ```
  每指标分段退化映射：`[0,warn]→0`，`[warn,crit]→线性0→0.6`，`[crit,fail]→0.6→0.95`，`>fail→≤0.98`
  - 阈值 `MetricThresholds`（:87-143）：响度 warn/crit/fail = 2.5/4.5/6.0 sone，锐度 = 1.75/2.8/4.0 等；`from_limits` 按 `1×/1.6×/2.4×` 自动生成三档
  - 健康分级：**≥70 正常 / ≥40 警告 / ≥20 临界 / <20 失效**
- **剩余寿命 RUL**（`run_phm_prediction` :410-560）：小时轴归一化 → auto 模式在**指数 `a·e^(bx)+c`**（curve_fit 智能初值 :567-609）、二次、线性中选 **R² 最优**模型；外推曲线与失效线按 20% 线性插值求交得 RUL（:658-695）；置信度：≥5 点且 R²>0.85 = High。Mock 生成器 :800-887

### 7.10 ODS 工作变形（`core/ods_engine.py` + `ui/ods_3d_view.py`）
- **传递率** `T_i(f) = P_xy/P_xx`（scipy welch/csd，nfft 4096，overlap 87.5%，:100-167）
- **ODS 振型提取**（`extract_ods_shape` :169-238）：Hanning 单段 rFFT 取目标频 bin，幅值 + 相对参考通道相位，`ODS 向量 = A·e^(jφ)`
- **模态分类**（`_classify_mode` :240）：相位差 `>0.7π` 反相 / `<0.3π` 同相，判定弯曲/扭转/椭圆化
- 峰值检测：`find_peaks(prominence=10 dB)`（:338）
- **3D 动画**（`ods_3d_view.py:105-161`）：测点瞬时位移 `drive = A_i·cos(φ_anim+φ_i)·exaggeration`（clip ±0.8·scale），沿径向，节点位移按**反平方距离加权插值**；`_anim_timer`（QTimer，interval=`1000/max(24,freq·30)` ms :391）驱动 `_animation_step`（:397）：`phase += 2π·f·(1/30)`，模 2π；`stop_animation`（:423）停 timer；`closeEvent`（:432）必须停动画防止野指针

---

## 8. 数据导入与校准规格（`core/acquisition/file_importer.py`）

### 8.1 支持格式（:103-127）
| 类别 | 格式 | 读取方式 |
|---|---|---|
| 音频 | wav / flac / mp3 / m4a / aac / ogg | WAV 原生 `wave` 模块读 8/16/24/32 bit（归一化 `/2^(bit−1)` :147-212）；其余经 ffmpeg |
| 视频 | mp4 / avi / mkv | `utils/io.py:_convert_to_wav`（:215-265）调 ffmpeg `-acodec pcm_s16le -ar 44100 -ac 1` |
| 数值 | csv / txt / dat | 分隔符自动检测、时间列推 fs（默认 48000 :257） |
| 科学 | mat | scipy.io，变量名推断 fs（:412） |
| 仪器 | HDF5 / TDMS / **HEAD .hdf** | h5py / 专用解析（见 §8.2） |

统一返回 `SignalData`（:59-98），`to_physical()` 乘灵敏度。**注**：导入路径未用 soundfile/sounddevice，后者仅用于 DAQ 实时采集。

### 8.2 HEAD HDF 解析（:431-664）★ 校准关键
HEAD acoustics 设备导出的 `.hdf`：
1. **magic 检测**：文本头含 `; Copyright` + `HEAD acoustics`（:431-438）
2. **文本头解析**：`delta value`（→fs）、`nbr of channel/scans`、`start of data`（默认 65536）、`byte order`；逐通道解析 `name str / physical quantity / physical unit / calibration / implementation type`（:441-536）
3. **二进制**：FLOAT32/64、INT16/32 等，多通道交织 reshape（:597-607）
4. **校准策略**（:611-664）：
   - `physical unit = Pa` 系 → 数据已是物理量，calibration 仅元数据
   - 电压数据 → `calibration ≥ 1` 视为 mV/Pa：`Pa = raw/(cal/1000)`；`< 1` 直接乘
   - ⚠️ **calibration 字段不一定等于灵敏度**（如某文件 calibration=83.08 但有效灵敏度=45 mV/Pa），需结合 `_SI_Sensor` 元数据
   - 传感器**有效灵敏度 = Sensitivity × CalibrationFactor**（从 `_SI_Sensor` XML 提取）

### 8.3 特征频率计算（`core/frequency_calculator.py`）
- 轴承 BPFO/BPFI/BSF/FTF（:116-147）
- 电磁阶次/槽谐波边带 `f = n·p·f_s ± k·Z·f_s`（:65-113）
- 换向纹波、蜗轮/行星轮频率，用于频谱标注

---

## 9. 各功能模块详细规格

### 9.1 SPL / 双模式评估（`dual_mode_engine.py` + `dual_mode_chart.py`）
- 双模式：SPL 评估 + 响度评估并行，`evaluate_both_modes` 同时给出 OASPL(A) 与 sone/尖锐度判定
- `dual_mode_chart.py`（443 行）：DualModeChartWidget 展示双模式结果
- 判定逻辑：与用户设定的限值（`standard_selector.py` 选标准/自定义 `CustomStandardDialog`）比较得 Pass/Fail

### 9.2 声品质 / SQI（`sqi_radar.py` 599 行）
- `_RadarCanvas`（matplotlib 雷达图）+ `_ScorePanel`（评分）+ `_MetricsTable`（五维指标表）
- 调 `psychoacoustics` 算五维 → SQI → 雷达图可视化

### 9.3 PHM（`phm_dashboard.py` 1008 行）
- `TestNodeEditorDialog` 编辑测试节点（多个时间点的测量）
- 主面板：HI 曲线 + RUL 预测曲线 + 健康分级看板（调 §7.9 `run_phm_prediction`）

### 9.4 ODS（`ods_dashboard.py` + `ods_3d_view.py` + `ods_frequency_selector.py`）
- `MotorGeometryModel` 电机几何模型，`ODS3DView`（pyqtgraph.opengl）渲染 3D
- `ODSFrequencySelector` 选目标频率；`ods_dashboard` 组织测点与计算
- 动画驱动见 §7.10

### 9.5 语谱图标注（`spectrogram_annotator.py` 527 行）
- `SpectrogramAnnotatorWidget`：基于 `widgets.py SpectrogramWidget`
- `RectROIItem` 矩形 ROI 框选故障区域，`MarkEditDialog` 编辑标注（故障类型/备注）→ 存入知识库

### 9.6 批处理（`core/batch_worker.py` + `ui/batch_panel.py` 303 行）
- `BatchWorker`（QThread）：glob 全格式（:113-121）→ 去重 → 逐文件按格式分发读取 → `remove_dc` → `evaluate_both_modes`（SPL+响度）→ `compute_psychoacoustic_full`（5 维+SQI）→ 汇总 Pass/Fail 及超标原因（:239-300）

### 9.7 报告生成（`report_generator.py` + python-docx）
- Word 报告：图表（matplotlib 内嵌）+ 指标表 + 判定结论
- 字体路径 Windows `C:/Windows/Fonts/arial.ttf`（:367,387）——跨平台需适配

### 9.8 专家诊断（`expert_panel.py` 486 行 + `core/expert_system.py`）
- `_StageGroupWidget`（分阶段组）+ `DrivetrainConfigPanel`（传动链配置）+ `ExpertDiagnosisDisplay`（诊断展示）
- 结合物理诊断 + 历史案例 + 声纹匹配输出结论

### 9.9 知识库（`knowledge_panel.py` 454 行 + `core/case_database.py`）
- FMEA 案例的增删改查（`case_editor_dialog.py` 672 行）
- 声纹向量 8 维（peak_freq_1/2, oaspl_dba, loudness, sharpness, roughness, fluctuation, sqi）+ 加权余弦相似度（频率权重最高 3.0）
- 双轨匹配：SNR>8dB 走 Track A（声纹 80%+工况 20%），否则 Track B 纯工况检索

### 9.10 声纹检索（`match_display.py` 303 行 + `v2_knowledge/`）
- `MatchResultCard` + `MatchDisplayPanel` 展示相似案例匹配结果（相似度分数、历史案例卡片）

### 9.11 传动链建模（`v2_engine/kinematic_model.py` + `topology_editor_ui.py`）
- OO 传动链：Shaft（input/intermediate/output/auxiliary）/Gear/Bearing（6 种类型含接触角）/TransmissionStage/KinematicGraph
- 由输入转速自动推导各级 GMF 与轴承特征频率；可序列化；配套拓扑编辑器 UI

### 9.12 TSA 时域同步平均（`v2_engine/tsa_engine.py`）
- 转速脉冲提取（峰值/过零）→ 三次样条瞬时转速 → 等角度重采样（默认 1024 点/转）→ 跨转系综平均（TSA）→ 输出降噪 dB 估计；NumPy 向量化

### 9.13 阶次分析（`order_analysis/`）
- 转速脉冲提取 → RPM 曲线 → 阶次切片

### 9.14 传感融合（`sensor_fusion/`）
- 多通道（声/振/电流）融合诊断 + 投票机制

### 9.15 数据采集 / DAQ（`acquisition_panel.py` 822 行 / `daq_panel.py` 1020 行）
- `acquisition_panel`：LevelMeter + AcquisitionPanel（传感器采集 + 校准，`core/acquisition/realtime_recorder.py`、`sensor_config.py`）
- `daq_panel`：CalibrationTableWidget + LiveOscilloscopeWidget（实时示波器）+ LevelBarWidget，多通道 DAQ（sounddevice/nidaqmx/mock 三种 backend）
- 录音默认存 `~/NVH_*.wav`（`daq_engine.py:989`）

### 9.16 现场记录（`field_logger_panel.py` 506 行）
- `BlindLoggerToggle`（盲测记录）+ `OverloadIndicatorBar`（过载指示）+ `WavExportDialog`（WAV 导出）

### 9.17 EoL 产线（`eol_dashboard.py` 768 行 + `core/eol_controller.py`）
- `VerdictDisplayWidget`（大字判定屏）+ `ShiftStatusPanel`（班次状态）+ `HistoryListWidget`（历史）
- 产线自动判定 OK/NG

### 9.18 数据湖看板（`data_lake_dashboard.py` 1395 行）
- KPICard + FacetedSearchPanel（分面搜索）+ TrendChartPanel（趋势图）
- 读 `nvh_data_lake.db` 的 test_records / diagnostic_results / raw_data_meta

---

## 10. 数据模型（数据库 Schema）

### 10.1 主库 `nvh_data.db`（`core/db_config.py`）— ⚠️ 占位框架
- SQLAlchemy（:55），支持 SQLite/PostgreSQL/MySQL，SQLite 启用 WAL（:239）
- 默认路径 `~/.autoacoustics/nvh_data.db`（:214）
- **表结构缺失**：`init_database()`（:354）依赖 `core/db_models.py`，**该文件不存在**，运行时打印 "No db_models module yet; skipping table creation"（:372）
- 附带 `get_nas_path`（:382）NAS 路径解析

### 10.2 数据湖 `nvh_data_lake.db`（`core/data_lake_engine.py`）— 真实实现
SQLAlchemy 优先 + 纯 sqlite3 回退（:29-42）。路径 `~/.autoacoustics/nvh_data_lake.db`（:191）。

| 表 | 关键字段 | 用途 |
|---|---|---|
| `test_records` (:52) | record_id, test_time, operator, project_code, motor_model, test_condition, oaspl_dba, loudness_sone, sharpness/roughness/fluctuation/prominence, sqi_score, overall_verdict(PASS/FAIL), exceeded_bands | NVH 测试顶层记录 |
| `diagnostic_results` (:115) | test_record_id(FK), diagnosis_type, fault_description, confidence_score, health_index(0-100), severity(高/中/低), recommendation, raw_evidence(JSON) | AI 诊断结论，1:N |
| `raw_data_meta` (:148) | test_record_id(FK), file_path, file_format, file_size_bytes, md5_hash, cloud_uri, synced_at | 原始文件索引+云同步标记 |

- `sync_to_cloud()`（:722）是**模拟**（写 `nvs://` 假 URI）；`seed_mock_data()`（:873）注入 500 条假数据（可用 `AA_DISABLE_MOCK=1` 禁用）

### 10.3 FMEA 案例库 `fmea_cases.db`（`core/case_database.py`）— 真实实现（原生 sqlite3）
- 路径 `~/.autoacoustics_pro/fmea_cases.db`（:253）——注意是 `_pro` 子目录，与其他库不一致
- 单表 `fmea_cases`（:266）：id, case_name, fault_description, solution_applied, **vector_json**（8维声纹向量JSON）, source_file, created_at + Phase9 元数据 rpm, transmission_stage, component_type, motor_type, audio_quality_score, gate_decision, denoised（含自动列迁移 :299）
- 冷启动预置 3 条演示案例（:346）

### 10.4 用户库 `users.db`（`core/auth_manager.py`）— 真实实现（原生 sqlite3）
- 单表 `users`（:388）：user_id(PK), display_name, password_hash(SHA256+固定盐 :232), role(operator/engineer/manager/admin), department, email, phone, is_active, created_at, last_login, failed_login_count, locked_until
- RBAC 权限矩阵 `PERMISSIONS`（:88）25 项权限×4 角色；5 次失败锁 15 分钟（:461）
- 审计日志写 JSONL 文件（`AuditLogger` :250，IATF 16949 格式）
- 预置 admin/admin123 + 3 个演示账号（:428）

---

## 11. UI / UX 规格

### 11.1 主题系统（`cyber_industrial_theme.py` 1150 行 + `theme.py` 226 行）
- **Cyber-Industrial 主题**：深色工业风 + 霓虹点缀。`CyberColors`（配色常量）、`GlassCard`（玻璃拟态卡片）、`NeonButton`（霓虹按钮）、`CyberThemeManager.apply(app)` 全应用注入 QSS
- `theme.py` 是另一套 dark industrial 主题（备用）

### 11.2 核心可视化组件（`widgets.py` 1909 行）
| 组件 | 功能 |
|---|---|
| `TimeWaveformWidget` | 时域波形 |
| `FFTSpectrumWidget` | FFT 频谱（pyqtgraph） |
| `SpectrogramWidget` | 语谱图 |
| compute 系列 | `compute_spl_time_history`、`compute_loudness_time_history`（见 §7.4/7.8） |

### 11.3 对话框与辅助界面
| 文件 | 类 | 功能 |
|---|---|---|
| `oobe_wizard.py`（1030 行） | `OOBESetupWizard` | 首次启动向导（OOBE） |
| `login_dialog.py`（345 行） | `LoginDialog` | 企业模式登录 |
| `upsell_dialog.py`（278 行） | `UpsellDialog` | 付费功能升级提示（`show_for_feature`） |
| `standard_selector.py`（646 行） | `StandardSelectorPanel` / `CustomStandardDialog` | 标准选择 / 自定义标准 |
| `case_editor_dialog.py`（672 行） | `CaseEditorDialog` | 故障案例编辑 |
| `audio_player.py`（126 行） | `AudioPlayer` | sounddevice 音频回放 |

### 11.4 关键交互机制
- **许可证锁 tab**（`main_window.py` upsell 事件过滤器）：锁定 tab 的 tooltip 含 `🔒`，点击时事件过滤器拦截 → 弹 `UpsellDialog.show_for_feature(feature_name)`。⚠️ 已知修复：对 `QTabBar` 必须用 `mapFromGlobal(event.globalPosition().toPoint())` + `tabAt(pos)` 定位索引再取 `tabToolTip(idx)`，不能用 `toolTip()`（会取到空串导致点击无反应）
- **会话保存/恢复**（L978/L1044）：保存当前 tab、参数、导入的文件，重启恢复
- **浮动面板**（L2168/L2270）：任意面板可 `float_out` 成独立 QMainWindow
- **快捷键**（`_init_shortcuts` L909）

---

## 12. 许可证与权限系统

### 12.1 付费分层（`core/license_engine.py`）★ 核心表
`LicenseTier`（:72）：**STANDARD / ADVANCED / CUSTOM / ULTIMATE**，累积继承；旧名别名 Base→Standard、Pro→Advanced、Enterprise→Ultimate（:85）。

**Tier→FeatureGroup 对照**（`TIER_FEATURES` :124，共 33 组，`FEATURE_CATALOG` :188 含中文名/图标/min_tier）：

| Tier | 新增 FeatureGroup |
|---|---|
| **STANDARD**（10） | core_analysis, spl_evaluation, loudness_eval, standard_config, report_export, batch_processing, signal_enhancement, basic_expert, local_knowledge_base, basic_audio_capture |
| **ADVANCED**（+8） | order_analysis, ods_3d, phm_prediction, feature_frequency, sound_quality, spectrogram_annotator, field_logging, daq_acquisition |
| **CUSTOM**（+8） | eol_mes, data_lake, ai_diagnosis, ai_evolution, expert_report, sensor_fusion, dual_mode_full, drivetrain_modeling |
| **ULTIMATE**（+7） | multi_user_auth, floating_license, open_api, plugin_system, ota_updates, audit_trail, tsa_engine |

- `has_feature`（:826）：查 `feature_name in _enabled_features`（按 tier 累积，`get_enabled_features` :302）
- **Trial 机制**（:646）：无 .lic 时读 `~/.autoacoustics/trial_state.json`（机器绑定），**30 天内给 Ultimate 全功能**，过期降为 Standard（标记 TRIAL_EXPIRED）
- 验证链：RSA-PKCS1v15 签名（公钥内嵌 :483）→ 机器绑定 → 过期 → tier

### 12.2 节点锁定适配（`core/factories.py` `NodeLockedLicenseAdapter.check_out()` :310）
- `VALID` → 放行
- `NOT_ACTIVATED` / 名字含 `trial` / `DEMO_MODE` → **也放行**（返回 "Trial/demo mode"，:319-324）——**这是开发期关键改动，避免授权弹窗锁死**
- 其余状态拒绝；license_manager 模块缺失 → 放行（demo）

### 12.3 硬件指纹（`core/license_manager.py:187-282`）
`get_hardware_fingerprint()` 组件：**主板 UUID**（macOS `ioreg` / Linux `machine-id` / Windows `wmic csproduct get UUID`）+ **MAC**（uuid.getnode）+ **CPU**（platform.processor）+ **hostname**，拼接后 SHA256，输出 `前16位-后8位` 大写格式（:282）。

### 12.4 用户 RBAC（见 §10.4）
- 4 角色 operator/engineer/manager/admin × 25 项权限
- standalone 模式自动登录 admin；enterprise 模式弹 LoginDialog

---

## 13. AI 能力（含真实性标注）

> ⚠️ **真实性声明**：全仓库**不存在任何 `.onnx/.pth/.pkl/.enc` 模型文件**（已两次全目录 Glob 确认零匹配）。所有"AI" = 规则引擎 + Mock 数据 + 可选外部服务（LLM / 预训练 ResNet）。复刻时**不要假装有真实训练模型**。

### 13.1 规则诊断（`core/diagnosis.py`）
- **纯规则引擎**，仅完整实现 `diagnose_imbalance()`（:45）：找 1X 转频（rpm/60）最近频点，按超背景 dB 数分级（>30 严重 / 20-30 中等 / 15-20 轻微），`confidence = excess/40`
- 文档声称支持 8 类故障与 "V1.2 CNN/ResNet ML 分类器"（:5-7），**但代码中不存在其余 7 类与 ML 实现**

### 13.2 LLM RAG 专家报告（`v2_expert_system/llm_rag_engine.py`）
- OpenAI 兼容客户端，默认 **Ollama**（`http://localhost:11434/v1`，模型 llama3，:117），亦支持 vLLM/LM Studio/Azure/智谱；`provider="mock"` 离线兜底
- 流程：物理诊断（PhysicsDiagnosis）+ 向量库历史案例 + 声学指标 → 组装中文 prompt（:164）→ 生成**五段式 FMEA 报告 Markdown**（故障现象/机理分析/历史案例追溯/维修建议/OK-NG 判定，模板 :132），LLM 失败时用模板降级报告（:489）
- 另有 `mock_llm_server.py` 供离线测试

### 13.3 声纹向量库（`v2_knowledge/vector_db_manager.py`）
- **双后端自动降级**：ChromaDB（PersistentClient，cosine HNSW，:65）优先 → NumPy 内存余弦 KNN 兜底（:193）
- 512 维向量 + metadata（机型/诊断标签/wav路径/rpm 等）
- **知识来源是 Mock**：`mock_data_loader.py` 生成 1000 条合成签名注入

### 13.4 Embedding（`v2_knowledge/embedding_engine.py`）
三级降级：用户 ONNX 模型 → torchvision **ResNet18**（ImageNet 预训练，去 FC 得 512 维）→ NumPy 手工特征（:94-106）。仓库内无 ONNX 权重，实际只能走 ResNet18/minimal。

### 13.5 模型加密加载（`core/model_encryption.py`）
- AES-256-GCM 解密 `.enc` 模型到内存（不落盘），密钥从**硬件指纹 + 许可证**派生
- 喂给 ONNX Runtime / PyTorch；`models/diagnosis.onnx.enc` 仅为文档示例路径，**实际文件不存在**

---

## 14. 构建、打包与部署

### 14.1 PyInstaller（`AutoAcoustics_Pro.spec` 62 行）
- 入口 `launcher.py`、`pathex=['src']`、`collect_all('autoacoustics')` + 18 个 hiddenimports、**onedir**、upx=True、console=False，macOS 追加 BUNDLE
- `AutoAcousticsPro.spec` 为 Windows 变体

### 14.2 打包链
1. `build_windows.bat`：一键（`--lite` 约 300MB / `--full` 含 AI 约 2GB，7 步）
2. `build_installer.py`：调 `.venv` 的 PyInstaller，50+ hidden-imports，`--add-data` 打 config.yaml/assets/icon.ico，onedir windowed，输出 `dist\AutoAcousticsPro`
3. `installer.iss`（Inno Setup）：AppId GUID、lzma2/max、`PrivilegesRequired=lowest`、MinVersion 10.0 x64 → `Output\AutoAcousticsPro_Setup_0.28.0.exe`

### 14.3 tools/ 工具
`build.py`（跨平台打包）、`encrypt_model.py`（模型加密）、`keygen.py` + `license_generator.py` + `keys/`（许可证签发）、`setup_cython.py`

---

## 15. 外部系统依赖

| 依赖 | 用途 | 获取方式 | 必需性 |
|---|---|---|---|
| **ffmpeg** | 视频/非 WAV 音频转码（PATH 查找，`platform_compat.py:174-178`） | `winget install ffmpeg` | 视频导入必需 |
| **VC++ Redistributable** | torch / onnxruntime 加载 | https://aka.ms/vs/17/release/vc_redist.x64.exe | AI 功能必需 |
| **Inno Setup 6** | 打包安装向导（`build_installer.py:111-122`，PATH→回退 `C:\Program Files (x86)\Inno Setup 6\iscc.exe`） | 官网下载 | 仅打包 |
| **NI-DAQmx** | DAQ 硬件采集（`nidaqmx` 仅 Windows） | NI 官网驱动 | 仅 DAQ 硬件 |
| **Ollama / LLM** | 专家报告 RAG | localhost:11434 | 可选 |

---

## 16. 实现指引与降级策略

### 16.1 推荐实现顺序（供第三方大模型/团队参考）
1. **工程骨架**：`launcher.py` 十步启动链 + `src/` 布局 + `AppConfigManager` 三级配置搜索 + `platform_compat.py` 平台适配
2. **核心 DSP**：`dsp.py`（FFT/SPL/计权）+ 1/3 倍频程 Parseval（§7.1-7.4）——**先跑通，这是所有功能地基**
3. **响度 + 声品质**：`psychoacoustics.py` Zwicker 全公式（§7.5-7.6）——**最难，校准参数原样照抄**
4. **数据导入与校准**：`file_importer.py`（§8，HEAD HDF 校准是关键）
5. **UI 框架**：主窗口 5 模式页 + cyber_industrial 主题 + `widgets.py` 三大图组件
6. **核心功能**：SPL/响度/语谱图/批处理/报告
7. **进阶功能**：PHM/ODS/专家诊断/知识库/传动链/TSA/阶次/传感融合
8. **数据层 + 许可证**：4 库 schema + LicenseTier 分层
9. **AI 能力**（可选，全降级）：LLM RAG / 向量库 / embedding
10. **打包**：PyInstaller + Inno Setup

### 16.2 降级策略（重要）
**AI 功能全部"优雅降级"**——这是现有代码的设计哲学，复刻必须保留：
- 缺 torch/onnxruntime → 核心 DSP（SPL/响度/声品质）仍完整可用
- 缺 chromadb → 向量库降级 NumPy KNN
- 缺 ONNX → embedding 降级 ResNet18 / 手工特征
- 缺 LLM → RAG 降级模板报告 / mock
- 缺 nidaqmx → DAQ 降级 mock backend
- 许可证缺失 → demo 模式放行（见 §12.2，**复刻时不建议保留"授权失败弹窗锁死"的旧行为**）

### 16.3 复刻注意事项（踩过的坑）
- **torch CPU 版必须第一个装**且带 `--index-url https://download.pytorch.org/whl/cpu`，否则 `pip install -e .` 会拉 2.5GB CUDA 版
- **运行不需要 pip install**：`launcher.py` 自举 sys.path，直接 `python launcher.py`
- 项目路径含空格时，PowerShell/CMD 命令需加引号
- **Zwicker 响度三要素缺一不可**（ISO 226 听阈 + GF 修正 + Bark 扩散），漏一个结果偏差 10-100 倍
- **calibration ≠ 灵敏度**（HEAD HDF），需结合 `_SI_Sensor` 元数据算有效灵敏度
- 配置文件有两份：项目根 `config.yaml`（模板）+ `%APPDATA%\AutoAcoustics\config.yaml`（运行时生效）
- FMEA 库在 `~/.autoacoustics_pro/`（带 `_pro`），与其他库（`~/.autoacoustics/`）路径不一致，迁移/复刻时注意

---

## 17. 附录

### 17.1 校准参数速查表 ★
| 参数 | 值 | 来源 |
|---|---|---|
| 参考声压 P_REF | `20e-6 Pa` | IEC 标准 |
| 默认采样率 | `48000 Hz` | config |
| Zwicker C 系数 | **`0.048`** | v3 校准（64 组 HEAD 数据） |
| Zwicker α | **`0.23`** | Moore-Glasberg 1996 |
| Bark 扩散 σ_low / σ_high | `1.5 / 1.2` Bark | 对应 ±27 dB/Bark |
| Bark 网格 | 24 点 `[0.5,...,23.5]` | — |
| GF 修正 | 5-6.3kHz 区域 **+1.5 dB** | v2 联合校准 |
| SQI 权重 | `(0.30,0.25,0.15,0.15,0.15)` | 响度/锐度/粗糙/波动/突出 |
| SQI NORM_REFS | `8 sone/3 acum/1.5 asper/1.0 vacil/20 dB` | — |
| PHM HI 权重 | `{loudness:.20, sharpness:.15, roughness:.18, fluctuation:.12, prominence:.10, sqi:.25}` | — |
| 时间计权 τ_F / τ_S | `0.125s / 1.0s` | IEC 61672 |
| 声纹向量 | 8 维（peak_freq_1/2, oaspl, loudness, sharpness, roughness, fluctuation, sqi），频率权重 3.0 | — |
| 试用期 | 30 天全功能（Ultimate） | — |

### 17.2 声学标准索引
| 标准 | 用于 |
|---|---|
| IEC 61672-1 | A 计权、时间计权（Fast/Slow） |
| IEC 61260-1 | 1/3 倍频程滤波器 |
| ISO 226:2003 | 等响曲线 / 听阈 |
| ISO 532-1 / DIN 45631 | Zwicker 响度 |
| DIN 45692 | Aures 尖锐度 |
| DIN 45646 / ISO 11904 | 外耳传递函数（自由场） |
| ISO 13381-1 | PHM 状态监测与诊断 |

### 17.3 核心文件清单（复刻必看）
```
launcher.py                          # 入口（566 行）
config.yaml                          # 配置模板
src/autoacoustics/
├─ core/dsp.py                       # FFT/SPL/计权/STFT
├─ core/psychoacoustics.py           # ★ Zwicker 响度 + 声品质
├─ core/dual_mode_engine.py          # 双模式评估
├─ core/sound_quality.py             # 声品质（旧简化版）
├─ core/phm_engine.py                # PHM 健康预测
├─ core/ods_engine.py                # ODS 计算
├─ core/batch_worker.py              # 批处理
├─ core/frequency_calculator.py      # 特征频率
├─ core/acquisition/file_importer.py # 数据导入 + HEAD HDF 校准
├─ core/db_config.py                 # 主库（占位）
├─ core/data_lake_engine.py          # 数据湖
├─ core/case_database.py             # FMEA 案例库
├─ core/auth_manager.py              # 用户 RBAC
├─ core/license_engine.py            # 许可证（新）
├─ core/license_manager.py           # 许可证（旧）+ 硬件指纹
├─ core/factories.py                 # 工厂 + 节点锁定适配
├─ core/diagnosis.py                 # 规则诊断
├─ core/model_encryption.py          # 模型加密
├─ core/plugin_manager.py            # 插件系统
├─ core/app_config.py                # 配置管理
├─ core/platform_compat.py           # 平台适配
├─ ui/main_window.py                 # ★ 主窗口（4503 行）
├─ ui/cyber_industrial_theme.py      # 主题
├─ ui/widgets.py                     # 三大图组件 + compute 系列
├─ v2_engine/                        # TSA + 传动链运动学
├─ v2_expert_system/                 # LLM RAG
└─ v2_knowledge/                     # 声纹向量库 + embedding
```

### 17.4 已知 Bug 修复记录（复刻时直接采用正确写法）
1. **`ods_3d_view.py` QTimer**：`self._anim_timer = QTimer()` 必须改为 `QTimer(self)`（带 parent）；必须加 `closeEvent` 停动画，否则窗口关闭后定时器在已销毁 Widget 上回调崩溃
2. **`main_window.py` upsell 事件过滤器**：对 `QTabBar` 取 tooltip 必须用 `tabAt(mapFromGlobal(...))` 定位索引，不能用 `toolTip()`（空串导致点击无反应）
3. **`factories.py` check_out**：必须把 `DEMO_MODE` 加入放行列表，否则试用期过期后程序被授权弹窗锁死
4. **macOS→Windows 移植**：`report_generator.py` 字体路径需跨平台；`app_logger.py` 内存函数 Windows 用 psutil；pyproject build-backend = `setuptools.build_meta`

---

*文档结束。本说明书基于 AutoAcoustics Pro v0.28.0 源码全量调研整理，所有公式、参数、Schema 均来自实际代码，可直接用于复刻。*
