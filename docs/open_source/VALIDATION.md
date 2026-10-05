# 公开源码验证

## 0.3.1 更新（2026-10-05）

本机导出的公开源码树 `pytest -q -ra`：**524 passed、48 skipped、14 warnings、23 subtests passed，98.21 s，退出 0**。本轮新增 22 个设备选择/连续录音、试听状态/历史来源、诊断导出线程检查。48 项跳过原因与下述 0.3.0 相同，未以跳过项充当验收。

以下为 0.3.0 历史记录；对应提交的 GitHub CI 结果以 Actions 为准。

2026-10-04；Windows，本机 Python 3.12.14。验证对象是独立导出的公开源码树，不依赖开发目录的私有录音或知识库。

## 本机结果

- `pytest -q -ra`：**502 passed、48 skipped、14 warnings、23 subtests passed，61.49 s，退出码 0**。
- 48 个跳过项：19 个私有录音/校准相关项，28 个外部响度参考项，1 个受限 Windows 登录会话凭据项。跳过项不计作通过。
- `pytest --collect-only --require-external-data`：因未包含私有录音和外部参考而退出码 4，证明严格验收入口会拒绝缺失材料。
- `pip wheel --no-deps --no-build-isolation`：生成纯 Python 源码 wheel，包含 64 个产品模块、GPL/NOTICE/MoSQITo Apache 许可，没有 DLL、EXE 或其他依赖二进制。
- `tools/export_public_source.py --verify`：逐文件尺寸、SHA256 及选定凭据模式检查通过。原产品规格副本与本机原件 SHA256 相同。

14 个警告来自 npTDMS 与 NumPy 的 dtype 弃用接口，不是本轮测试失败。公共源码回归未执行真实录音或外部数值参考验收。

## GitHub CI

`Windows source validation` 在干净 Windows runner 上安装 requirements.lock、FFmpeg 与源码，检查导出清单，执行公开回归并构建 wheel。实际远端结果以仓库 Actions 的对应提交记录为准，本机通过不预先代表 CI 通过。

## 验收范围

真实麦克风、NI/HEAD 真机、独立声校准、完整标准符合性、第二台用户 Windows 的人工试用不由这些公共测试证明。公开源码 Release 不包含历史便携 EXE 或私人资料。
