# 运维手册

GIGNN 是一个 Python 库包而非托管服务。本手册覆盖本地验证、包检查与常见故障。

## 运行模型

| 领域 | 当前状态 |
| --- | --- |
| 运行时服务 | 无。未定义 HTTP 服务器或 worker 进程。 |
| API 端点 | 无。没有路由或 OpenAPI 文件。 |
| 运行时环境变量 | 包代码不要求。 |
| 部署目标 | Python wheel 与 sdist 产物。 |
| 发布 | 仅手动。无自动 PyPI 上传。 |

## 标准验证

使用项目虚拟环境或另一个已装依赖的 Python `>=3.10` 环境。

```bash
.venv/bin/python -m pytest
.venv/bin/python examples/quickstart.py
.venv/bin/python -m build
```

纯文档验证：按 `docs/COMMANDS.md` 运行本地 Markdown 链接检查，并将 `llms.txt`、`llms-full.txt`、`docs/AGENTS.md`、`docs/RAG_INDEX.md` 与 README 导航及当前公开 API 边界比对。

格式检查：

```bash
.venv/bin/python -m black --check gignn tests examples
```

## 干净 wheel 冒烟

```bash
rm -rf /tmp/gignn-clean-venv
python3.11 -m venv /tmp/gignn-clean-venv
/tmp/gignn-clean-venv/bin/python -m pip install --upgrade pip
/tmp/gignn-clean-venv/bin/python -m pip install dist/gignn-*.whl
cd /tmp
/tmp/gignn-clean-venv/bin/python - <<'PY'
from gignn import GIGNNAutoClassifier, GIGNNClassifier
print(GIGNNClassifier.__name__, GIGNNAutoClassifier.__name__)
PY
```

## 产物检查

| 检查 | 预期结果 |
| --- | --- |
| wheel 存在 | `dist/gignn-<version>-py3-none-any.whl` |
| sdist 存在 | `dist/gignn-<version>.tar.gz` |
| wheel 内许可 | `gignn-<version>.dist-info/licenses/LICENSE` 存在 |
| 运行时包导入 | `from gignn import GIGNNAutoClassifier, GIGNNClassifier` 成功。 |
| 内部任务文件 | `.claude/`、`.spec/`、`tmp/` 不进入 wheel。 |
| 公开元数据 | 无私有本地路径、内部原始 URL 或仅 notebook 的路径。 |
| LLM 文档链接 | `llms.txt`、`llms-full.txt`、`docs/AGENTS.md`、`docs/RAG_INDEX.md` 存在且本地 Markdown 链接可解析。 |

## 常见问题

| 症状 | 可能原因 | 修复 |
| --- | --- | --- |
| `command not found: python` | shell 无 `python` 垫片。 | 使用 `.venv/bin/python` 或 `python3.11`。 |
| Python 版本是 3.9 | 用了 macOS 系统 Python。 | 使用 `.venv/bin/python` 或创建 Python `>=3.10` venv。 |
| `No module named sklearn` | 依赖未装进当前解释器。 | 运行 `.venv/bin/python -m pip install -e '.[dev]'`。 |
| `No module named torch` | PyTorch 缺失或装在别的环境。 | 先装官方 PyTorch wheel 组，再重装 GIGNN。 |
| `No module named torch_geometric` | PyG 缺失。 | 为当前 Torch/Python 平台安装官方 PyG wheel 组。 |
| 图构建没有边 | 几何策略或阈值太稀疏。 | 增大 `k`、增大 `threshold`，或提供显式 `edge_index`。 |
| 自动工作流报告全部图候选失败 | 每个候选都缺可用的训练/验证/测试切分边。 | 用更密的图、更大的分区、更宽的阈值，或与分区兼容的显式边。 |
| 预测拒绝特征列 | 预测列与训练列不一致。 | 重排/选择列以匹配 `feature_names_in_`。 |
| 检查点加载被拒 | 模式/版本、文件/成员/解压尺寸、数组哈希/形状/dtype、张量数、类数或特征数违反守卫。 | 从可信的兼容模型重建检查点；不要绕过校验。`GIGNNClassifier` 用 `.pt`，`GIGNNAutoClassifier` 用 `.gignn`。 |

## 恢复

这是库包。回退即还原源码变更或重装上一个产物。

```bash
git revert <commit>
python -m pip install <previous-wheel-path>
```

不要在包回退过程中删除用户数据或外部环境。

## 上报升级

| 情形 | 动作 |
| --- | --- |
| 构建产物含内部文件 | 停止发布，检查 manifest/源码树，排除泄漏后重建。 |
| 元数据暴露私有路径或 URL | 停止发布，移除元数据泄漏，重建并复检。 |
| 干净安装因 GIGNN 导入/API 错误失败 | 视为发布阻断。 |
| 干净安装仅因 PyTorch/PyG wheel 解析失败 | 记录平台特定的依赖安装路径后重试。 |
| 检查点安全守卫意外失败 | 先检查检查点来源与模式，再考虑任何绕过。 |
