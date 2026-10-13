# Windows 本机环境准备验证记录

日期：2026-09-25（Asia/Shanghai）。代码基线：`8cbd4d6b94eb9713952d3bd8f050da176accdca1`，另有本次未提交的 Windows 文件锁兼容改动。

## 阶段失败与处理

- 本机 `python` 指向 3.14.6，不满足 `pyproject.toml` 的 `>=3.11,<3.14`；改用已有的 `E:\Python313\python.exe`（3.13.1）。
- 首次 `npm ci` 无进展；调试输出显示 `package-lock.json` 的包地址指向当前不可达的 `registry.anpm.alibaba-inc.com`，请求反复 `ECONNRESET`。停止该次安装后，通过 `--registry https://registry.npmjs.org/ --replace-registry-host=always` 取包，未改锁文件。
- Windows 启动后端首先报 `ModuleNotFoundError: No module named 'fcntl'`。将 JSON 偏好存储的文件锁改为 `portalocker`，并在项目依赖及锁文件中显式声明。
- Docker Desktop CLI 存在，但 Docker daemon 未运行；WSL 只有 `docker-desktop` 发行版。本次采用原生 Windows 本机模式。

## 本次验证的原始终端结果摘录

| 检查 | 实际结果 |
| --- | --- |
| Python 锁文件安装 | `uv sync --frozen` 安装 110 个包；随后离线检查显示 `Checked 110 packages` / `Would make no changes`。 |
| 前端依赖 | 使用官方 npm 仓库成功安装 257 个包。 |
| 前端构建 | `npm run build`：TypeScript 与 Vite 均成功，`603 modules transformed`。 |
| 前端本机 HTTP | Vite 监听 `127.0.0.1:5173`；`GET /` 返回 200，HTML 包含根节点。 |
| 文件锁回归 | `python -m pytest -q tests/test_memory_persistence.py`：`15 passed in 0.86s`。 |
| 后端隔离启动 | 在独立 `DATA_DIR`，以临时假密钥和 `127.0.0.1:9` 模型/embedding 端点启动 `uvicorn`；日志出现 `Application startup complete`。 |
| 后端本机 HTTP | `GET /health` 返回 200，`database=sqlite`、`trade_database=sqlite`；`GET /docs` 返回 200。 |

隔离启动中的向量建库和品类知识库请求因假 embedding 端点连接失败，日志记录降级；这符合预期，不能视为真实模型、向量检索或完整买家选购通过。两台临时服务已停止，未占用 `5173` 或 `18000` 端口。隔离数据在工作区 `.tooling/globex-smoke-data`，未写入项目原有 `data/`。项目 `.env` 的 `LLM_API_KEY` 保持空值。

## 后续真实链路

在被 Git 忽略的根目录 `.env` 中填写实际可用的 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL` 与 `EMBEDDING_MODEL`；如 embedding 使用独立网关，再填写 `EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY`。先验证账户支持工具调用、流式输出及 embeddings，再按 README 启动单 worker 后端和前端，完成页面选购。`RERANKER_BASE_URL` 留空时不会创建专用精排器，不能把该状态记为精排已验证。
