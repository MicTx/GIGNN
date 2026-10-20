# 命令参考

在仓库根目录使用这些命令。存在项目虚拟环境时优先使用它。

## 环境搭建

| 命令 | 用途 |
| --- | --- |
| `python3.11 -m venv .venv` | 创建本地开发虚拟环境。 |
| `. .venv/bin/activate` | 在 POSIX shell 中激活虚拟环境。 |
| `python -m pip install --upgrade pip` | 在当前环境内升级 pip。 |
| `python -m pip install -e '.[dev]'` | 以可编辑模式安装 GIGNN 并带开发工具。 |
| `python -m pip install -e '.[gis,dev]'` | 安装开发工具与可选 GeoPandas/Shapely 辅助组件。 |

## 开发检查

| 命令 | 用途 | 来源 |
| --- | --- | --- |
| `.venv/bin/python -m pytest` | 运行 `tests/` 下全部测试。 | `pyproject.toml` pytest 配置 |
| `.venv/bin/python -m black --check gignn tests examples` | 只检查格式不改文件。 | `.[dev]` extra 含 Black |
| `.venv/bin/python examples/quickstart.py` | 运行内置工作流冒烟示例。 | `examples/quickstart.py` |

## 文档检查

纯文档变更后与发布准备前使用本检查。

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import re

root = Path('.')
docs = [
    root / 'README.md',
    root / 'llms.txt',
    root / 'llms-full.txt',
    *sorted((root / 'docs').glob('*.md')),
]
missing = []
for path in docs:
    text = path.read_text()
    for match in re.finditer(r'\[[^\]]+\]\(([^)]+)\)', text):
        target = match.group(1)
        if '://' in target or target.startswith('#'):
            continue
        rel = target.split('#', 1)[0]
        if rel and not (path.parent / rel).resolve().exists():
            missing.append(f'{path} -> {target}')
if missing:
    raise SystemExit('\n'.join(missing))
print('All local markdown links resolve')
PY
```

同时检查机器可读入口与公开文档保持对齐：

| 文件 | 必须与以下保持对齐 |
| --- | --- |
| `llms.txt` | `README.md`、`docs/FEATURE_OVERVIEW.md`、`docs/API.md`、`docs/DATA_CONTRACT.md` |
| `llms-full.txt` | `docs/API.md`、`docs/DATA_CONTRACT.md`、`docs/REPORTS.md`、`docs/PLOTTING.md`、`docs/SECURITY.md`、`docs/COMMANDS.md` |
| `docs/AGENTS.md` | `docs/COMMANDS.md`、`docs/SECURITY.md`、当前测试地图 |
| `docs/RAG_INDEX.md` | README 文档表列出的全部文件 |

## 构建命令

| 命令 | 用途 | 来源 |
| --- | --- | --- |
| `.venv/bin/python -m build` | 构建 wheel 与 sdist 到 `dist/`。 | `pyproject.toml` 构建后端 |
| `.venv/bin/python -m build --sdist --wheel --outdir /tmp/gignn-build` | 在仓库外构建产物用于验证。 | Python build 前端 |

## 干净产物验证

用仓库外的虚拟环境做干净 wheel 验证。

```bash
python3.11 -m venv /tmp/gignn-clean-venv
/tmp/gignn-clean-venv/bin/python -m pip install --upgrade pip
/tmp/gignn-clean-venv/bin/python -m pip install dist/gignn-*.whl
cd /tmp
/tmp/gignn-clean-venv/bin/python - <<'PY'
from gignn import GIGNNAutoClassifier, GIGNNClassifier
print(GIGNNClassifier.__name__, GIGNNAutoClassifier.__name__)
PY
```

## 包事实

| 条目 | 值 |
| --- | --- |
| 包名 | `gignn` |
| 版本 | `0.1.0` |
| Python 要求 | `>=3.10` |
| 许可 | `MIT` |
| 构建后端 | `setuptools.build_meta` |
| 测试路径 | `tests` |
| pytest addopts | `-q` |
| 运行时环境变量 | 无要求 |

## 备注

- macOS 系统 `python3` 可能是 Python 3.9，不满足本项目要求。
- PyTorch 与 PyTorch Geometric 是核心依赖；依赖解析失败时先安装官方平台 wheel。
- 不要提交本地产物如 `.venv/`、`dist/`、`build/`、`*.egg-info/`、`__pycache__/`、`.pytest_cache/`，除非发布流程显式要求生成产物。
