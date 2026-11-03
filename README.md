# 北辰智造企业知识库 RAG Agent

> 面向企业内部知识库的问答 Agent。在 273 份异构文档（Markdown / DOCX / PDF / CSV / XLSX / PPTX / EML）上回答制度、人事、财务、销售与项目问题，重点处理版本冲突、权限边界、结构化查询与不可信文档。
>
> 当前版本 v1.0（2026-11-03），评估指标已锁定。个人项目，知识库为虚构企业“北辰智造科技”（197 人，工业软件 / MES）的模拟语料，不含真实企业数据。

## 亮点

- **可量化评估**：121 题开发集均分 **1.94 / 2.0**（114 题满分），20 题留出集 1.85，10 题同义改写 1.90，7 题提示注入对抗全部通过。
- **权限在检索层**：按角色与密级过滤，未授权内容不进入模型上下文；权限合规率 1.000，泄漏 0。
- **版本权威判定**：同主题新旧版本可能同时标为 `active`，结合生效日期、正文废止声明与修订记录判断权威版本；版本冲突类 20 题均分 2.00。
- **结构化问题走工具**：聚合、对比类问题路由到 DuckDB / pandas，其余走向量 + BM25 混合检索；检索 Recall 1.000，MRR 0.849。
- **工程化交付**：FastAPI + SSE 流式 + 多轮会话，84 个单元测试，Docker 镜像，干净环境安装验证。

## 要解决的问题

企业知识库不是“把文档向量化再问模型”。项目针对以下六类陷阱设计：

| 陷阱 | 例子 |
|---|---|
| 版本冲突 | 同主题新旧版本同时标为 `active`（HR-001 与 HR-023、FN-001 与 FN-012），需结合生效日期、正文废止声明与修订记录判定 |
| 权限边界 | 薪资、预算、合同等机密文档只能进入有权限角色的上下文，不能靠提示词“请拒答” |
| 多格式解析 | 扫描 PDF 需要 OCR，表格散落在 CSV / XLSX / Markdown 中 |
| 结构化问题 | 合计、占比、对比类问题必须走 SQL / pandas，不能靠向量检索硬答 |
| 不可信文档 | 检索到的文档正文可能包含覆盖指令、角色劫持或提示词泄漏诱导 |
| 引用可核验 | 每个关键事实必须标注真实 `doc_id`，不能把 A 文档的话标成 B 文档 |

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

- `prepare`：多轮追问改写，把问题变成可独立检索的查询。
- `retrieve`：基础检索；聚合题放宽候选数；另做一次受限检索用于权限判定。
- `guard`：判断是否在问无权内容，命中则直接拒答，不进入生成。
- `route`：区分 RAG 与结构化工具（SQL / 表格）。
- `self_check`：检查引用、提示词泄漏、控制符与格式问题。

## 关键设计

1. **权限在检索层，不在提示词**
   检索前按角色过滤可见密级，检索后再按 `can_access(role, meta)` 校验；未授权内容不会进入模型上下文。角色由调用方传入（当前无登录鉴权，见“已知边界”）。

2. **版本权威性优先于“最新版本”**
   综合 `status`、`effective_date`、正文废止声明与修订记录判断权威版本；草稿、已废止版本只能作为补充说明。问题点名某文件或版本时，以该文件原文作答。

3. **混合检索 + 元数据过滤**
   Chroma 向量与 BM25 关键词检索经 RRF 融合；默认 `top_k=12`，聚合题在无表格时放宽到 16。rerank 是可选开关，默认关闭（见消融）。

4. **结构化问题走工具**
   路由到 `structured` 后，用 DuckDB / pandas 查询 CSV / XLSX，回答中说明统计口径；工具结果异常时回落生成。

5. **引用强制与自检**
   回答必须用 `[DOC-ID]` 标注来源；结束后检查引用真实性、系统提示词泄漏与不安全标记。

6. **间接提示注入防护**
   检索上下文用 `<document trust="untrusted">` 隔离；文档内覆盖指令按整句屏蔽；输出侧检测提示词逐字泄漏并替换为安全拒答。

7. **评估优先 + 回答指纹**
   代码、索引或 `RAG_*` 配置任一变化都会改变回答指纹，旧回答自动失效，避免混用不同版本的分数。评估支持按题增量重跑、局部结果并入与复用回答做 judge 稳定性验证。

## 评估

完成标准是可量化评估，而不是 demo 能跑通。四套集合用途不同，分别记录：

| 集合 | 题数 | 用途 | 平均分 |
|---|---|---|---|
| 开发集 | 121 | 日常迭代与基线 | **1.94 / 2.0** |
| 留出集 | 20 | 冻结后只跑一次的最终验收 | **1.85** |
| 同义改写 | 10 | 检验问法鲁棒性 | **1.90** |
| 提示注入 | 7 | 间接注入对抗 | 7/7 通过 |

答案由 LLM judge 按 0 / 1 / 2 打分。各集合的检索与合规指标：

| 集合 | Recall | MRR | 引用命中 | 忠实率 | 拒答正确 | 权限泄漏 |
|---|---|---|---|---|---|---|
| 开发集 | 1.000 | 0.849 | 0.991 | 1.000 | 1.000 | 0 |
| 留出集 | 1.000 | 0.971 | 1.000 | 0.950 | 1.000 | 0 |
| 同义改写 | 1.000 | 0.853 | 1.000 | 1.000 | — | 0 |

- 121 题平均延迟 28.7 s，其中检索约 0.38 s；主要开销在生成模型 API。judge 重复 3 次的总费用约 2.80 元。
- judge 稳定性：20 题已存回答重复判分 3 次，逐题得分完全一致；121 题全量重复 3 次的一致率 0.9835。
- 注入防护在 7 题对抗集上全部通过，121 题回归中没有误屏蔽良性文档。

### 迭代消融

| 改动 | 结果 | 决策 |
|---|---|---|
| `TOP_K` 8 → 12 | gold 覆盖率 90.3% → 95.1%，两道路径题修复 | 采用 12，聚合题放宽到 16 |
| rerank 开启 | MRR 0.792 → 0.947，答案均分基本持平，检索延迟 +2.98 s/题 | 默认关闭：排序收益未转化为答案收益 |
| rerank 扩候选 | 3 题路由从 rag/table 变为 sql，答成“未查询到记录” | 只重排已召回 top-12，不扩大上下文成员 |
| 间接注入防护 | 7/7 通过；121 题误屏蔽 0 | 保留，纳入默认运行时 |
| judge 重复 1 → 3 次 | 判定调用约 3 倍；得分一致率 0.9835 | 日常 1 次，验收 3 次 |

<details>
<summary>开发集分类结果（8 类）</summary>

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

</details>

逐题得分与分类明细见 [`docs/eval/`](docs/eval/)，评估集冻结规则见 [`eval/HOLDOUT.md`](eval/HOLDOUT.md)。

### 运行

```bash
# 开发集 121 题（日常 1 次判分；出稿验收加 --judge-repeats 3）
PYTHONIOENCODING=utf-8 python -m eval.run_eval

# 留出集 / 同义改写 / 提示注入
PYTHONIOENCODING=utf-8 python -m eval.run_eval \
  --questions eval/questions.holdout.v1.jsonl \
  --report-stem report_holdout_v1 --results-name results_holdout_v1.jsonl \
  --judge-repeats 1
PYTHONIOENCODING=utf-8 python -m eval.run_injection_eval --stem injection_v1

# 单元测试
python -m unittest discover -s tests -v
```

报告写入 `build/eval/report*.{json,md}`；`eval/compare_reports.py` 可对比两份报告的检索指标、逐题排名与事实点变化。

## 快速开始

环境：Python 3.14（Windows + bash 或 Linux / macOS）。

```bash
# 1) 安装依赖
python -m venv .venv
source .venv/Scripts/activate        # Linux / macOS 用 .venv/bin/activate
python -m pip install -r requirements.txt

# 2) 配置密钥（项目不会自动读取 .env，需要导出到环境变量）
cp .env.example .env
# 编辑 .env，至少填写 DASHSCOPE_API_KEY
set -a; source .env; set +a

# 3) 构建语料与索引（会调用 embedding 接口；扫描件会调用 OCR）
python -m src.ingest.build_corpus
python -m src.retrieval.index --rebuild

# 4) 命令行问答
PYTHONIOENCODING=utf-8 python -m src.agent.ask "员工年假几天？" --role 普通员工 --debug

# 5) Web Demo
python -m uvicorn src.api.main:app --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

角色：`普通员工`、`HR`、`财务`、`销售`、`IT`、`产品`、`管理层`。流式接口：

```bash
curl -N -X POST http://127.0.0.1:8000/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"员工年假几天？","role":"普通员工"}'
```

多轮时把上一轮返回的 `session_id` 带上。配置项完整列表见 [`.env.example`](.env.example)，常用项：

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_CHAT_MODEL` | `qwen-plus` | 生成模型 |
| `RAG_TOP_K` | `12` | 默认检索条数 |
| `RAG_AGG_TOP_K` | `16` | 聚合题候选放宽 |
| `RAG_RERANK` | `0` | rerank 开关，默认关闭 |
| `RAG_OCR` | `1` | 扫描件 OCR 开关 |
| `RAG_TABLE_TOOL` | `1` | 结构化工具开关 |

干净环境验证：Python 3.14.5 新建 venv 安装依赖后，84 个单测通过，命令行问答与 `uvicorn` 启动正常，`/api/health` 返回 `index_ready: true`。

### Docker

镜像只打包依赖、`src/` 与只读的 `data/`；`build/` 中的语料和索引不烘焙进镜像，运行时挂载宿主机的 `build/`。

```bash
# 宿主机先构建索引
python -m src.ingest.build_corpus
python -m src.retrieval.index --rebuild

docker build -t beichen-rag:dev .
docker run --rm -p 8000:8000 --env-file .env \
  -v "$(pwd)/build:/app/build" beichen-rag:dev
```

容器以非 root 用户（uid 10001）运行，镜像自带 `HEALTHCHECK`。索引必须与镜像使用相同的 embedding 模型和维度；换模型后需在宿主机重建索引。

## 已知边界与取舍

1. **留出集低于开发集**（1.85 vs 1.94）。差异来自 judge 对复合事实点和同义表述偏严，以及个别题目的补充说明不完整；检索、引用、拒答与权限指标均无回归。
2. **7 道 1 分题中有部分是真实遗漏**，例如漏答目标职级、草案状态或个别社保项。当前版本保留原始得分，没有做人工修正。
3. **有一道花名册计数题真实答错**：题目被路由到 RAG 而非结构化工具，回答列出 5 人却写成 7 人。修复需要同时调整路由规则与职级映射，并重跑全量。
4. **引用命中率 0.991**：唯一一题引用了汇总表而非明细表，内容正确但归属不符。
5. **生成不可完全复现**：即使 `temperature=0`、`seed=42`，兼容接口下仍存在措辞与结构化字段合并的漂移；基线是单次采样值。
6. **部署形态与生产仍有差距**：会话存储为单进程内存、无登录鉴权（角色由调用方传入）、不支持可取消执行、SSE 反向代理未实测；注入检测以规则为主。上生产需要补 SSO / 鉴权、共享会话存储、可观测性与取消机制。
7. **rerank 默认关闭是取舍，不是能力缺失**：当前配置下它提升 MRR 但未转化为答案收益，且增加延迟与费用。

这些边界均来自评估记录，逐题明细见 [`docs/eval/`](docs/eval/)。

## 目录结构

```text
Dockerfile      容器镜像；build/ 通过 volume 挂载，不烘焙进镜像
data/kb/        只读知识库快照（273 份文档 + 清单）
src/
  ingest/       解析、元数据、OCR
  retrieval/    分块与索引构建、混合检索、rerank、密级过滤
  security/     角色权限、提示注入检测
  tools/        SQL / pandas 查询工具
  agent/        LangGraph 状态图、运行时与会话
  api/          FastAPI
  ui/           Web Demo
eval/           评估集、judge、跑分与报告脚本
problem/        技术问题记录（现象、原因、方案、验证）
docs/handoff/   模块交接文档
build/          派生索引与中间产物（不入库）
```

## 相关文档

- [`docs/eval/`](docs/eval/)：评估报告摘要（汇总指标、分类指标、逐题得分）。
- [`eval/HOLDOUT.md`](eval/HOLDOUT.md)：留出集与同义改写集的冻结规则、SHA256 与运行约定。
- [`problem/`](problem/)：典型问题复盘，包括权限题被内部文档绕过、rerank 改变工具路由、guard 过度拒答、间接提示注入与 judge 评分校准。
- [`docs/handoff/`](docs/handoff/)：开发过程中的决策记录与模块交接。
