# 北辰智造企业知识库 RAG Agent

面向企业内部知识库的问答 Agent。知识库为北辰智造科技（197 人，工业软件 / MES）的 273 份文档（Markdown、DOCX、PDF、CSV、XLSX、PPTX、EML 等），覆盖制度、HR、财务、销售、IT、产品与项目资料。

项目的完成标准不是"能跑通 demo"，而是**可量化评估**：121 题开发集 + 20 题留出集 + 10 题同义改写 + 7 题提示注入对抗集，全部带可复现的回答指纹与判定指纹。

## 要解决的问题

企业知识库不是"把文档向量化再问模型"这么简单。本项目针对以下真实陷阱做设计：

| 陷阱 | 例子 |
|---|---|
| 版本冲突 | 同主题新旧版本同时标为 `active`（HR-001 与 HR-023、FN-001 与 FN-012），需结合生效日期、正文废止声明与修订记录判定权威版本 |
| 权限边界 | 薪资、预算、合同等机密文档只能进有权限角色的上下文，不能靠提示词"请拒答" |
| 多格式解析 | 扫描 PDF 需要 OCR，表格散落在 CSV / XLSX / Markdown 中 |
| 结构化问题 | 合计、占比、对比类问题必须走 SQL / pandas，不能靠向量检索硬答 |
| 不可信文档 | 检索到的文档正文可能包含覆盖指令、角色劫持或提示词泄漏诱导 |
| 引用可核验 | 每个关键事实必须标注真实 `doc_id`，不能把 A 文档的话标成 B 文档 |

## 核心设计取舍

1. **权限在检索层，不在提示词**
   检索前按角色过滤可见密级，检索后再按 `can_access(role, meta)` 校验；未授权内容不会进入模型上下文。角色由调用方传入（当前无登录鉴权，见「已知局限」）。

2. **版本权威性优先于"最新版本"**
   综合 `status`、`effective_date`、正文废止声明与 `PO-008` 修订记录判断权威版本；草稿、已废止版本只能作为补充说明。问题点名某文件/版本时，以该文件原文作答，其他文件只能放在补充说明。

3. **混合检索 + 元数据过滤**
   Chroma 向量与 BM25 关键词检索经 RRF 融合；默认 `top_k=12`，聚合题在无表格时放宽到 16。rerank 作为可选开关，默认关闭（见消融）。

4. **结构化问题走工具**
   路由到 `structured` 节点后用 DuckDB / pandas 查询 CSV / XLSX，回答中说明统计口径；工具结果异常时回落生成。

5. **引用强制与自检**
   回答必须用 `[DOC-ID]` 标注来源；结束后检查引用真实性、系统提示词泄漏与不安全标记。

6. **间接提示注入防护**
   检索上下文用 `<document trust="untrusted">` 隔离；文档内覆盖指令按整句屏蔽；输出侧检测提示词逐字泄漏并替换为安全拒答。

7. **评估优先 + 回答指纹**
   代码、索引或 `RAG_*` 配置任一变化都会改变回答指纹，旧回答自动失效，避免混用不同版本的分数。评估支持 `--ids` 增量、`--merge-partial` 并入、`--judge-stability` 复用回答复判。

## 架构

```text
用户 / Web Demo
      │  HTTP / SSE
      ▼
FastAPI（并发限流、超时、内存会话、SSE 流式）
      │
      ▼
LangGraph（只负责路由与状态流转）
  prepare → retrieve → guard ──► finalize（无权 / 无资料拒答）
                         │
                         └→ route ──► structured（DuckDB / pandas）──► self_check
                                      │                                      ▲
                                      └────────────► generate ───────────────┘
                                                                             │
                                                                         finalize

检索层：Chroma 向量 + BM25 → RRF → 密级/角色过滤 → 可选 rerank（默认关）
索引层：build/corpus（解析 + OCR）→ build/index（chunks / Chroma / BM25）
数据层：data/kb（273 份只读文档 + _manifest.json）
```

关键节点：

- `prepare`：多轮追问改写（有历史时），把问题改写成可独立检索的查询。
- `retrieve`：基础 `top_k` 检索；聚合题放宽；额外做一次 `top_k=1` 的受限检索用于权限判定。
- `guard`：判断是否在问无权内容；命中则直接拒答，不进入生成。
- `route`：区分 RAG 与结构化工具（SQL / 表格）。
- `self_check`：检查引用、提示词泄漏、控制符与格式问题。

## 快速开始

环境：Python 3.14（Windows + bash 或 Linux / macOS）。

```bash
# 1) 安装依赖
python -m venv .venv
source .venv/Scripts/activate        # Linux/macOS 用 .venv/bin/activate
python -m pip install -r requirements.txt

# 2) 配置密钥（项目不会自动读取 .env，需要导出到环境变量）
cp .env.example .env
# 编辑 .env，至少填写 DASHSCOPE_API_KEY
set -a; source .env; set +a

# 3) 构建语料与索引（会调用 embedding 接口；扫描件会调用 OCR）
python -m src.ingest.build_corpus            # 解析 data/kb → build/corpus
python -m src.retrieval.index --rebuild      # 分块、embedding、Chroma、BM25

# 4) 命令行问答
PYTHONIOENCODING=utf-8 python -m src.agent.ask "员工年假几天？" --role 普通员工 --debug

# 5) Web Demo
python -m uvicorn src.api.main:app --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

可选：`python -m src.ingest.build_corpus --no-ocr` 跳过扫描件 OCR（会缺少 8 份扫描件的正文）。

> 干净环境验证（2026-11-03）：Python 3.14.5 新建 venv 安装依赖后，84 个单测通过，命令行问答与 `uvicorn` 启动均正常，`/api/health` 返回 `index_ready: true`。

API 示例：

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"员工年假几天？","role":"普通员工"}'

# 流式（SSE）；多轮时把上一轮返回的 session_id 带上
curl -N -X POST http://127.0.0.1:8000/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"员工年假几天？","role":"普通员工"}'
```

角色：`普通员工`、`HR`、`财务`、`销售`、`IT`、`产品`、`管理层`。

配置项完整列表与默认值见 [`.env.example`](.env.example)。常用项：

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_CHAT_MODEL` | `qwen-plus` | 生成模型 |
| `RAG_TOP_K` | `12` | 默认检索条数 |
| `RAG_AGG_TOP_K` | `16` | 聚合题候选放宽 |
| `RAG_RERANK` | `0` | rerank 开关，默认关闭 |
| `RAG_OCR` | `1` | 扫描件 OCR 开关 |
| `RAG_TABLE_TOOL` | `1` | 结构化工具开关 |
| `RAG_MAX_TOKENS` | `900` | 生成预算，截断后逐级放宽 |

## Docker

镜像只打包依赖、`src/` 与只读的 `data/`；`build/` 中的语料和索引不烘焙进镜像，运行时挂载宿主机的 `build/`。

```bash
# 1) 宿主机先构建索引（需要 DASHSCOPE_API_KEY）
python -m src.ingest.build_corpus
python -m src.retrieval.index --rebuild

# 2) 构建镜像
docker build -t beichen-rag:dev .

# 3) 启动；配置通过 --env-file 注入，索引通过 volume 挂载
docker run --rm -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/build:/app/build" \
  beichen-rag:dev
```

Windows Git Bash 会转换 `-v` 里的路径，改用：

```bash
docker run --rm -p 8000:8000 --env-file .env \
  -v "$(cygpath -w "$(pwd)/build"):/app/build" beichen-rag:dev
```

浏览器打开 `http://127.0.0.1:8000`。容器以非 root 用户（uid 10001）运行，Linux 宿主机需保证 `build/` 对其可读。`/api/health` 在索引预热完成后返回 `index_ready: true`，镜像自带 `HEALTHCHECK`。

- 索引必须与镜像使用相同的 embedding 模型和维度；换模型后要在宿主机重建索引。
- `.env` 不要挂载进容器；`--env-file` 只支持 `KEY=VALUE` 与注释，与 `.env.example` 格式一致。
- Docker Hub 不可达时，先通过可用镜像源拉取 `python:3.14-slim` 并在本地打标签，构建时可用 `--build-arg PIP_INDEX_URL=...` 指定 PyPI 索引。这些属于环境配置，不写入项目依赖。

## 评估

### 数据集

| 集合 | 题数 | 用途 | 是否参与调参 |
|---|---|---|---|
| `eval/questions.v5.jsonl` | 121 | 开发集，日常迭代与基线 | 是 |
| `eval/questions.holdout.v1.jsonl` | 20 | 留出集，只跑一次做最终验收 | 否 |
| `eval/questions.paraphrase.v1.jsonl` | 10 | 同义改写，检验问法鲁棒性 | 否 |
| `eval/questions.injection.v1.jsonl` | 7 | 间接提示注入对抗 | 否 |

留出集与同义改写集的冻结规则、SHA256 与运行规则见 [`eval/HOLDOUT.md`](eval/HOLDOUT.md)。

### 运行

```bash
# 开发集 121 题（日常 judge×1；出稿验收用 --judge-repeats 3）
PYTHONIOENCODING=utf-8 python -m eval.run_eval
PYTHONIOENCODING=utf-8 python -m eval.run_eval --judge-repeats 3

# 留出集 / 同义改写（已冻结，不建议重跑）
PYTHONIOENCODING=utf-8 python -m eval.run_eval \
  --questions eval/questions.holdout.v1.jsonl \
  --report-stem report_holdout_v1 --results-name results_holdout_v1.jsonl \
  --judge-repeats 1

# 提示注入对抗（7 题）
PYTHONIOENCODING=utf-8 python -m eval.run_injection_eval --stem injection_v1

# 复用已存回答做 judge 稳定性（仅验证 judge 时用 --allow-stale）
PYTHONIOENCODING=utf-8 python -m eval.run_eval \
  --results build/eval/results.jsonl --judge-stability 3 --allow-stale \
  --stability-stem report_stability

# 单元测试
python -m unittest discover -s tests -v
```

报告写入 `build/eval/report*.{json,md}`；`eval/compare_reports.py` 可对比两份报告的检索指标、逐题排名与事实点变化。

### 当前基线（2026-11-03 冻结）

开发集回答指纹 `ad595828258abf82`，判定指纹以 `build/eval/report.json` 为准。三个集合用途不同，分别记录，不跨集合比较。

**开发集（121 题，judge×3，用于迭代）**

| 指标 | 结果 |
|---|---|
| 平均答案得分（0/1/2） | **1.942**（7×1 + 114×2） |
| 检索 Recall / Gold 覆盖率 | 1.000 / 1.000 |
| MRR | 0.849 |
| 引用命中率 | 0.991 |
| 忠实率 / 无依据断言 | 1.000 / 0 |
| 拒答正确率 | 1.000 |
| 权限合规 / 泄漏 | 1.000 / 0 |
| 路由 | rag 109 / sql 10 / table 2 |
| 平均延迟（检索） | 28.7 s（0.38 s） |
| 费用 | 2.8038 元 |

**留出集（20 题，judge×1，冻结后只跑一次）**

| 指标 | 结果 |
|---|---|
| 平均答案得分 | **1.85**（3×1 + 17×2） |
| 检索 Recall / Gold 覆盖率 | 1.000 / 1.000 |
| MRR | 0.9706 |
| 引用命中率 / 忠实率 | 1.000 / 0.950 |
| 拒答正确率 | 1.000 |
| 权限合规 / 泄漏 | 1.000 / 0 |
| 路由 | rag 19 / sql 1 |
| 平均延迟（检索） | 20.0 s（1.46 s） |
| 费用 | 0.2198 元 |

**同义改写（10 题，judge×1，冻结后只跑一次）**

| 指标 | 结果 |
|---|---|
| 平均答案得分 | **1.90**（1×1 + 9×2） |
| 检索 Recall / MRR | 1.000 / 0.853 |
| 引用命中率 / 忠实率 | 1.000 / 1.000 |
| 权限泄漏 | 0 |
| 费用 | 0.1168 元 |

其他验证：

- **judge 稳定性**：20 题已存回答 × 3 次判分，得分逐题一致率 1.000；121 题全量 judge×3 的得分一致率 0.9835（119/121，两题众数均为 2）。
- **提示注入**：7/7 对抗用例通过，judge 均分 2.0；121 题中未误屏蔽良性文档，检索与路由零变化。
- **单元测试**：84 个通过（节点、运行时、API、流式、会话、judge、注入防护）。

开发集分类指标：

| 类别 | 题数 | 平均分 | 检索 Recall |
|---|---|---|---|
| 单跳事实 | 32 | 1.97 | 1.000 |
| 版本冲突 | 20 | 2.00 | 1.000 |
| 多跳推理 | 14 | 2.00 | 1.000 |
| 结构化聚合 | 19 | 1.89 | 1.000 |
| 权限敏感 | 18 | 1.89 | 1.000 |
| 拒答 | 8 | 2.00 | 1.000 |
| 口语化来源 | 6 | 1.83 | 1.000 |
| 代号术语 | 4 | 1.75 | 1.000 |

### 迭代消融

| 改动 | 对比口径 | 关键结果 | 决策 |
|---|---|---|---|
| `TOP_K` 8 → 12 | 60 题；gold 覆盖率离线统计 | 覆盖率 @8 **90.3%** → @12 **95.1%**；Recall（任一命中）均为 1.000；`multi_04` / `multi_05` 修复 | 采用 `RAG_TOP_K=12`、`RAG_AGG_TOP_K=16` |
| rerank 关闭 → 开启 | 68 题，固定回答指纹 + judge×3 | MRR **0.792 → 0.947**；答案均分 2.000 → 1.985（生成抽样噪声）；检索延迟 +2.98 s/题；生成费用 +0.396 元/68 题 | **保持默认关闭**：排序收益没有转化为答案收益 |
| rerank 扩候选（30→取 12） | 68 题 | 3 题路由从 rag/table 变成 sql，答成"未查询到记录"；MRR 仍 0.947 | 只重排已召回 top-12，不扩上下文成员 |
| 间接提示注入防护 | 7 题对抗 + 121 题回归 | 7/7 通过；121 题误屏蔽 0；检索 / 路由 / 引用 / 权限无系统性回退 | 保留防护，纳入默认运行时 |
| judge×1 → judge×3 | 121 题 | 判定调用约 3 倍（judge 费用 2.1508 元）；得分一致率 0.9835 | 日常 ×1，出稿验收 ×3 |

说明：不同批次的生成结果受模型服务时段影响存在措辞漂移，跨运行比较以检索指标、路由与固定回答指纹的判定为准。

## 已知局限

1. **留出集低于开发集**：20 题留出集 1.85，低于开发集 1.942。差异主要在结构化聚合的附加说明与术语题完整性，不是检索或权限回归（Recall、引用、拒答、泄漏均达标）。
2. **judge 复合事实点假阴性**：`key_facts` 用逗号串联多个结论时，回答已分点写全仍可能被判缺失（`perm_13`、`agg_11`、留出集 `ho_agg_02`）。这是 judge 粒度问题，已记录到 `problem/`，当前冻结集不追分。
3. **`agg_10` 是真实回答错误**：XS-002 研发中心在职实为 5 人，回答称 7 人；P6 及以上口径也与 gold 不一致。演示时避让，后续修复需重新全量验证。
4. **引用命中率 0.991**：唯一来源是 `agg_13`，答案引用月度统计 `TK-002` 而非明细 `TK-001`，内容正确但归属不符。
5. **生成不可完全复现**：即使 `temperature=0`、`seed=42`，DashScope 兼容接口下仍存在措辞与结构化字段合并的随机漂移；基线是单次采样值，不保证逐字复现。
6. **延迟主要来自模型 API**：121 题平均 28.7 s，其中检索约 0.38 s；优化重点不在本地检索。
7. **生产化未完成**：会话存储为单进程内存、无登录鉴权（角色由调用方传入）、不支持可取消执行、SSE 反向代理未实测；提示注入检测以规则为主，隐蔽改写仍可能漏检。
8. **rerank 默认关闭**：当前配置下 rerank 带来 MRR 收益但无答案收益，且检索延迟与费用明显增加。结构化工具对复杂多表汇总仍有缺口（如 `agg_10` 的花名册计数）。

以上边界作为主动说明，不阻塞主流程。

## 目录结构

```text
Dockerfile      容器镜像；build/ 通过 volume 挂载，不烘焙进镜像
data/kb/        只读知识库快照（273 份文档 + 清单）
src/
  ingest/       解析、元数据、OCR
  retrieval/    分块与索引构建（index.py）、混合检索、rerank、密级过滤
  security/     角色权限、提示注入检测
  tools/        SQL / pandas 查询工具
  agent/        LangGraph 状态图、运行时与会话
  api/          FastAPI
  ui/           Web Demo
eval/           评估集、judge、跑分与报告脚本
problem/        技术问题记录（按日期命名，含【面试可讲】条目）
docs/handoff/   模块交接文档
build/          派生索引与中间产物（不入库）
```

## 相关文档

- 最新进展与下一步：`docs/handoff/` 中按日期排序的最新文档。
- 冻结评估报告导出摘要：`docs/eval/`（汇总指标 + 逐题得分，完整 JSON 在本地 `build/`）。
- 评估协议与冻结规则：`eval/HOLDOUT.md`。
- 典型问题复盘：`problem/`，包括权限绕过、rerank 改变工具路由、guard 过度拒答、间接提示注入、judge 数字/版本口径等。
