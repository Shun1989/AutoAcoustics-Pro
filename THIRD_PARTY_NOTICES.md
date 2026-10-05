# 第三方源码与依赖

本文件适用于 **0.3.1 公开源码仓库**，不代表任何历史 Windows 二进制包的许可库存。

项目原始代码与文档采用 GPL-3.0-only，完整文本见 LICENSE。第三方代码保留自己的作者声明与许可；详见 NOTICE。

## MoSQITo 派生源码

`src/autoacoustics/analysis/_loudness_engine.py` 的滤波表、非线性与时间递推来自 MoSQITo 1.2.1（Eomys 与贡献者，Apache-2.0），修改内容为分块迭代、限制存储及保留块间状态。核心响度及 Bark 积分仍调用固定上游版本，未更改响度模型。

完整 Apache-2.0 文本：`licenses/source/Apache-2.0.mosqito.txt`。测试目录另保留原许可证副本。上游：<https://github.com/Eomys/MoSQITo/tree/v1.2.1>。

## 安装时取得的依赖

Python 包由用户从其各自发行渠道安装，本仓库不包含 wheel、DLL、EXE 或 SDK。实际版本清单见 requirements.lock；包的许可证与归属以该版本自带材料为准。

PyQt6 开源发行采用 GPLv3，Qt 本体适用自己的许可与第三方归属。项目选择 GPL-3.0-only 以配合当前 PyQt6 使用方式。官方说明：<https://www.riverbankcomputing.com/software/pyqt/intro>。

FFmpeg/ffprobe 可由用户另外安装用于媒体导入；许可取决于实际构建所启用的组件。官方说明：<https://ffmpeg.org/legal.html>。本仓库不再分发其二进制。

NI-DAQmx 驱动、HEAD SDK、硬件固件、厂商商标及文档不由项目许可证授权。通过 pip 安装 nidaqmx Python API 不包含 NI-DAQmx 驱动授权。

## 数据与二进制发行

合成夹具来自仓库内生成脚本，可与本项目源码一起使用。真实录音、知识库、厂商 PDF、扫描页、客户报告与独立标准参考附件未包含在公开发布中；资料可读取不代表获得公开再分发许可。

公开源码 Release 未附便携 EXE。若以后公开 Qt、FFmpeg 等组件的二进制包，须针对实际二进制补齐对应源码、构建信息及第三方声明，并核验适用许可条件。`tools/collect_distribution_notices.py` 只生成库存，不能替代这些工作。
