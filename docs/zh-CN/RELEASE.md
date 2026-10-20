# 发布

本文档描述 GIGNN 的手动发布准备流程。仓库不自动发布到 PyPI。

## 发布前置条件

| 检查 | 要求 |
| --- | --- |
| 工作树 | 除预期的发布变更外干净。 |
| 版本 | `pyproject.toml` 版本与目标发布一致。 |
| 文档 | README 与 docs 反映当前公开 API 与行为。 |
| 测试 | Python `>=3.10` 环境下全量测试通过。 |
| 构建 | wheel 与 sdist 构建成功。 |
| 产物 | 上传前检查 wheel 与 sdist。 |

## 版本来源

包版本定义在 `pyproject.toml`：

```toml
[project]
version = "0.1.0"
```

升级版本时更新提及确切版本的文档示例。

## 构建产物

构建：

```bash
.venv/bin/python -m build
```

预期产物名遵循以下模式：

```text
dist/gignn-<version>-py3-none-any.whl
dist/gignn-<version>.tar.gz
```

不影响仓库 `dist/` 的临时验证：

```bash
.venv/bin/python -m build --sdist --wheel --outdir /tmp/gignn-release-build
```

## 发布检查清单

1. 确认 `pyproject.toml` 版本。
2. 运行测试：

```bash
.venv/bin/python -m pytest
```

3. 运行 `docs/COMMANDS.md` 的文档链接检查。
4. 确认 `llms.txt`、`llms-full.txt`、`docs/AGENTS.md`、`docs/RAG_INDEX.md` 仍与 README 文档表及当前公开 API 边界一致。
5. 运行格式检查：

```bash
.venv/bin/python -m black --check gignn tests examples
```

6. 运行示例冒烟：

```bash
.venv/bin/python examples/quickstart.py
```

7. 构建产物：

```bash
.venv/bin/python -m build
```

8. 从仓库外验证 wheel 导入：

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

9. 检查产物是否意外含内部文件或私有元数据。
10. 仅在显式人工批准后打标签并发布。

## 产物检查

建议检查：

```bash
python -m zipfile -l dist/gignn-<version>-py3-none-any.whl
python -m tarfile -l dist/gignn-<version>.tar.gz
```

关注：

- wheel 元数据中的许可文件。
- `gignn/` 包文件在位。
- wheel 内没有 `.claude/`、`.spec/`、`.venv/`、`tmp/` 与 notebook 草稿文件。
- 元数据中无私有本地路径或仅内部 URL。

## 仓库双态

本仓库有两个状态：

- **私有 Gitea（真源态）。** 私有 `origin` 远端跟踪全部内容，包括 `patent/` 专利申请材料与 `.spec/` 内部任务记录。不要在这里停止跟踪或 gitignore 它们。
- **公开 GitHub（发布态）。** 公开镜像绝不能包含 `patent/`、`.spec/` 或 `tmp/`。不要把本仓库历史直接推到 GitHub —— 专利文件会从每个提交中仍可恢复。

发布清单通过后，按以下全新历史导出流程发布到 GitHub：

```bash
mkdir -p /tmp/gignn-public-src
git -C "$(pwd)" archive --format=tar HEAD \
  | tar -x -C /tmp/gignn-public-src
rm -rf /tmp/gignn-public-src/patent /tmp/gignn-public-src/.spec /tmp/gignn-public-src/tmp
cd /tmp/gignn-public-src
git init && git add -A && git commit -m "GIGNN <version>"
git remote add origin <github-remote>
git push -u origin main
```

`git archive` 只导出被跟踪的文件，构建产物、`.venv/`、`dist/` 会自动排除；三条 `rm -rf` 处理仅限内部的被跟踪目录。

## 发布策略

发布是手动的。除非显式要求发布目标且产物检查通过，不上传 PyPI 或 TestPyPI。

若使用 Twine，仅在复查后执行上传：

```bash
python -m twine check dist/*
python -m twine upload dist/*
```

`twine` 不在当前 `.[dev]` extra 内；如需要仅在发布环境中安装。

## 回退

发布提交有误时还原该提交：

```bash
git revert <commit>
```

安装的包有误时重装上一个已复查的 wheel：

```bash
python -m pip install <previous-wheel-path>
```

公开包上传有误时，按包索引策略执行撤回（yank）或替换。不要假设可以删除。
