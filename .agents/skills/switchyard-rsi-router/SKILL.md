---
name: switchyard-rsi-router
description: RSI Router 功能实现规范。任何路由框架或网关要实现 rsi-router 能力（子任务识别、Pareto 候选选择、skill 注入、flagship 兜底、轨迹留痕）时，按本文档实现。也可用于对照检查或修改既有实现。
---

# RSI Router 实现规范

本文档定义 RSI Router 的功能契约。实现者（人或大模型）在任意路由框架、网关或代理中
实现 `rsi-router` 能力时，以本文档为准。Switchyard 的参考实现见文末。

## 功能目标

RSI Router 把一次会话中的每一步（step）路由到最合适的执行模型：

1. 一个小模型（识别模型 S）把当前步骤标注为某个子任务 ID。
2. 从子任务字典读取该子任务的 Pareto 候选模型集。
3. 过滤、排序后选出执行模型。
4. 选中模型带有 skill 时，把 skill 作为提醒消息注入请求。
5. 任何环节无法产生可靠路由时，回退到配置的兜底模型（flagship）。
6. 全过程留痕：识别调用、路由决策、执行交互三类日志。

## 核心组件

实现必须提供以下组件，接口形式不限（函数、类、插件均可）：

| 组件 | 职责 |
|---|---|
| 识别模型 S | 输入当前步骤上下文，输出子任务 ID 和置信度 |
| 子任务字典 | 存储子任务定义和每个子任务的候选模型集 |
| 候选选择器 | 过滤、排序候选并选出执行模型与 fallback 顺序 |
| skill 注入器 | 把选中候选的 skill 以消息形式追加到请求 |
| 兜底路由 | 把无法路由的请求发给 flagship 模型 |
| 轨迹留痕 | 写出识别、路由、执行三类记录 |

## 路由流程

对每个进入 rsi-router 的请求，按顺序执行：

1. 读取全部子任务定义。
2. 构造识别模型 S 的输入（见下文契约）。
3. 调用识别模型 S，解析其输出。
4. 命中以下任一情况时回退到兜底 target（配置了 `fixed_target` 时用它，否则用
   `flagship_target`），流程结束：
   - 识别调用失败或输出无法解析
   - 子任务 ID 为空
   - 置信度低于 `confidence_threshold`
   - 子任务 ID 不在字典中
5. 读取该子任务的候选模型集。
6. 过滤候选：无效行、不在执行模型池中的模型、成功率低于 `min_success_rate` 的模型。
7. 用 `selection` 规则排序，第一个为执行模型，其余为有序 fallback。
8. 若候选中存在有效 skill，注入 skill 提醒消息。
9. 把请求发给执行模型；执行失败时按 fallback 顺序重试。

注意：

- 第 4 步的低置信度回退，如果子任务在字典中，会读取候选只为取 skill；识别失败、
  空 ID、未知子任务不读取候选（减少不必要的字典查询）。
- 回退请求在能识别子任务时注入同样的"最新有效 skill"（`is_valid` 非 0、内容非空、
  `skill_version` 最大）；识别失败、空 ID、未知子任务、候选读取失败时不注入。
- 字典读取失败不导致请求失败，走回退 target 并继续服务。

## 识别模型 S 契约

### 输入

识别模型 S 的用户消息由三部分按顺序拼接，位于固定的任务指令和可用子任务列表之后：

```text
USER REQUIREMENT:
<首条用户消息内容>

RECENT STEPS:
<最近 history_steps 个轮次（assistant 轮次起）的全部消息摘要>

PREVIOUS SUBTASK PREDICTION:
<上一次预测的子任务 ID，或 (none)>
```

可用子任务列表包含每个子任务的 `subtask_id`、`subtask_name`（可省略）、`definition`。

对每段内容设置长度上限（推荐：单段约 2000 字符，总长约 8000 字符），超长截断。

### 上一次预测

- 按会话维度记录：有 session ID 用 session ID；没有 session ID 时用首条用户消息
  的哈希作为会话键。
- 只要识别模型返回了合法 JSON，就更新记录——包括低置信度、空 ID、未知子任务 ID。
- 未知子任务 ID 也必须在下一步的输入中出现，不得因为不在字典里而被丢弃。
- 识别调用失败或输出无法解析时，保留上一次预测。
- 空预测在输入中显示为 `(none)`。

### 输出

识别模型必须只输出一个 JSON 对象，不含 Markdown 或代码围栏：

```json
{"subtask_id": "<id 或空字符串>", "confidence": <0 到 1 的数值>}
```

解析规则：

- 从输出文本中提取第一个 `{` 到最后一个 `}` 之间的内容做 JSON 解析。
- `confidence` 必须是 0 到 1 的有限数值，否则视为解析失败。
- `subtask_id` 解析后做 trim。

## 子任务字典契约

字典存储两个实体（关系库表、KV、内存结构均可，字段契约如下）：

**子任务定义**（识别模型的可选范围）：

| 字段 | 类型 | 说明 |
|---|---|---|
| subtask_id | 字符串，主键 | 子任务 ID，非空 |
| subtask_name | 字符串，可空 | 展示名，可省略 |
| definition | 字符串 | 子任务职责、输入、预期产出、典型失败点 |

**候选模型集**（每个子任务的 Pareto 集）：

| 字段 | 类型 | 说明 |
|---|---|---|
| subtask_id | 字符串 | 所属子任务 |
| model_id | 字符串 | 模型 ID，必须匹配执行模型池中的某个模型 |
| cost | 数值 | 成本，`lowest_cost` 排序用 |
| latency | 数值，毫秒 | 延迟，`lowest_latency` 排序用 |
| success | 数值 | 成功率，`highest_success` 排序用 |
| skill_content | 字符串，可空 | 可注入的 skill 文本 |
| skill_version | 数值，可空 | skill 版本；注入时取数值最大的，缺失按 `0` 处理 |
| is_valid | 数值/布尔，可空 | `0` 表示该行不参与 skill 注入；缺失按有效处理 |

兼容性要求（实现建议内置，减少部署摩擦）：

- 列名别名：`latency`/`latency_ms`、`success`/`success_rate`。
- `success` 通常是 0 到 1 的比例；大于 1 且不超过 100 的值按百分比处理，除以 100。
- `cost`、`latency`、`success` 缺失或非有限数值时，该行视为无效并跳过。
- 每次路由时读取字典，不长期缓存本地副本。

## 候选选择契约

过滤顺序固定为：

1. 只保留 `model_id` 在执行模型池（`candidate_targets`）中的候选。
2. 只保留 `success_rate >= min_success_rate` 的候选。
3. 按 `selection` 排序。
4. 取第一个为执行模型，其余按序作为 fallback。

`selection` 取值：

| 值 | 排序规则 |
|---|---|
| `lowest_cost`（默认） | `cost` 升序 |
| `lowest_latency` | `latency_ms` 升序 |
| `highest_success` | `success_rate` 降序 |

`min_success_rate` 只过滤字典候选，不过滤兜底模型。全部候选被过滤时回退兜底模型，
原因码记为 `no_candidate_above_min_success_rate`。

### 固定目标 fixed_target

配置 `fixed_target` 且不为空时，跳过上面的过滤排序：所有请求都由该 target 执行。

- 设置后不再回退 `flagship_target`：识别失败、空 ID、低置信度、未知子任务、
  字典读取失败、候选读取失败、候选全被过滤时都由 `fixed_target` 执行。
- skill 仍来自字典：在 `is_valid` 非 `0` 的候选中，取 `skill_version` 最大且内容
  非空的 `skill_content`；识别不到子任务或读不到候选时不注入。`is_valid` 只影响
  skill 注入，不影响路由选择。
- 固定目标没有 fallback 模型。
- 路由日志记录 `fixed_target` 和进入选择前的 `pareto_candidates`；`selection`
  字段不出现（未参与排序）。回退到固定目标时 `decision` 记为 `fixed_target`。
- 候选读取失败仍由固定目标执行，与未配置 fixed_target 时一致。

## Skill 注入契约

候选中存在 `is_valid` 非 `0` 且 `skill_content` 非空的行时，取 `skill_version`
最大的一条，在子任务开头放一条新消息：

```text
<skill_reminder>当前为子任务id<子任务 ID>，Skill：<skill 内容></skill_reminder>
```

规则：

- 子任务第一步把 reminder 追加在请求末尾；后续步骤把它插回同一位置，让第一步的工作排在
  reminder 之后。这样模型把 skill 读成子任务的上下文，而不是本步的新指令。
- 每个请求最多一条 reminder。若传入消息中已存在完全相同的 reminder 文本，不再追加。
- OpenAI 兼容格式使用 `role: "user"`。多数 OpenAI 兼容模型（如 Qwen 系列）的
  chat template 要求 `system` 只能出现在消息列表开头。
- Anthropic 格式：在内部表示中使用 `system` 语义；编码到 wire 时渲染为 `user`
  turn（Anthropic 协议的 system 文本走独立的顶层字段，不接受 messages 内的
  system 角色）。
- 注入必须真实到达模型：如果框架有同格式请求体重放/缓存机制，注入后必须使其失效。
- 注入器只做追加，从不删除或改写已有消息。

## 回退语义

以下原因码用于日志和 evidence，实现时语义必须一致：

| 原因码 | 触发条件 |
|---|---|
| `catalog_failed` | 子任务或候选查询失败 |
| `empty_catalog` | 子任务字典为空 |
| `recognition_failed` | 识别调用失败或输出无法解析 |
| `no_subtask` | 识别输出空子任务 ID |
| `low_confidence` | 置信度低于 `confidence_threshold` |
| `unknown_subtask` | 子任务 ID 不在字典中 |
| `no_configured_candidates` | 没有候选在执行模型池中 |
| `no_candidate_above_min_success_rate` | 所有候选低于最低成功率 |

## 轨迹留痕契约

实现必须写出三类追加式 JSONL 记录，可通过配置重定向路径。每条记录的公共字段：

`request_id`（本次请求唯一 ID）、`session_id`（可选）、`step`（会话内步号，
建议取请求中 assistant 消息数 + 1）、`route_id`、`ts`（时间戳）。

**识别日志**（每次识别调用一条）：

- 识别模型 ID、完整 prompt、原始 response
- `subtask_id`、`confidence`
- `latency_ms`、可选 `error`

**路由日志**（每次路由决策一条，不含消息内容和模型输出）：

- `decision`（`subtask`、`flagship`，或配置 fixed_target 后回退到它时的 `fixed_target`）、`reason`（回退原因码）
- `subtask_id`、`confidence`、`confidence_threshold`、`min_success_rate`、
  `selection`、`skill_injected`
- `candidates`：过滤排序后的合格模型 ID 有序列表，选中者第一，其余为 fallback
- `pareto_candidates`：进入选择前的完整候选集，每项包含
  `model_id`、`cost`、`latency_ms`、`success_rate`、`configured`（是否在执行
  模型池中）、`above_min_success_rate`（是否达到阈值）、`selected`（是否被选中）
- `selected_model`、`served_model`、`result`、`usage`、`cost_usd`、
  `cost_source`、`latency_ms`

`candidates` 与 `pareto_candidates` 在候选读取成功时必须出现——包括全部被过滤
后回退的情况；候选读取失败时可以省略。

**执行日志**（每次模型交互一条）：

- `upstream_request`：实际发给执行模型的请求体（模型侧视角）
- `response`：规范化后的模型响应
- `selected_model`、`served_model`、`subtask_id`、`confidence`、`decision`、
  `reason`、`skill_injected`
- `result`、`usage`、`cost_usd`、`cost_source`、`latency_ms`

流式响应在转发的同时累积记录；终端事件之前断掉的流可以没有执行记录；失败步骤可以
没有 `upstream_request`。

`skill_injected` 语义：本次请求实际追加了 reminder 才为 true；候选无 skill 或
相同 reminder 已存在时为 false。

## 配置项契约

| 配置项 | 必填 | 默认值 | 说明 |
|---|:---:|---|---|
| `recognizer_target` | 是 | — | 识别模型 S 对应的 target |
| `flagship_target` | 是 | — | 兜底模型 target |
| `candidate_targets` | 是 | — | 执行模型池，字典候选的 model_id 必须来自其中，名称唯一 |
| `confidence_threshold` | 否 | `0.7` | 进入候选路由的最低置信度，范围 `[0, 1]` |
| `min_success_rate` | 否 | `0.0` | 候选最低成功率，范围 `[0, 1]`；`0` 表示不过滤 |
| `history_steps` | 否 | `3` | 识别模型看到的最近步数，至少 `1` |
| `selection` | 否 | `lowest_cost` | `highest_success` / `lowest_cost` / `lowest_latency` |
| `fixed_target` | 否 | 未设置 | 设置后所有请求都用该 target 执行，跳过候选过滤排序，也不再回退 flagship；有可用候选时按规则注入 skill |
| `skill_injected` | 否 | `true` | `false` 时不向请求注入 skill reminder |
| `recognizer_prompt` | 否 | 内置提示词 | 覆盖默认识别指令 |
| `max_output_tokens` | 否 | `4096` | 识别模型最大输出 token，至少 `1` |
| `mysql_url_env` | 是 | — | 存字典连接串的环境变量名 |
| `mysql_timeout_ms` | 否 | `2000` | 单次字典查询超时，至少 `1` |

配置校验要求：数值越界、空 target、重复 target、零超时都必须在启动时报错。

## 实现注意

- 字典查询失败不致命：记录原因并走兜底，服务继续。
- 兜底模型不参与 `min_success_rate` 过滤。
- step 号依赖客户端回传完整会话历史，实现时在文档中说明这一点。
- 日志只增不改，长期运行需轮转；读取接口按需重扫文件或自建索引。

## 验收清单

实现完成后应覆盖以下测试场景：

1. 正常路径：识别命中且候选合格，选中排序第一的候选，skill 注入成功。
2. 每种回退原因码各一条路径，均路由到兜底模型。
3. `min_success_rate` 剔除低成功率候选；全部剔除时回退兜底。
4. `selection` 三种排序各自生效。
5. skill reminder：OpenAI 格式为 user 角色、Anthropic 内部为 system、后续步骤保持
   子任务开头位置且不重复追加、空 skill 不注入。
6. 上一次预测：无 session 时按首条用户消息聚类；未知子任务 ID 保留并在下一步出现；
   识别失败时保留旧预测。
7. 三类日志字段完整，路由日志包含 `pareto_candidates` 与过滤标记。
8. 字典不可用时服务不崩溃，持续走兜底。

## 参考实现

Switchyard 的 Rust 实现可作为对照：

- 路由行为：`crates/libsy/src/algorithms/rsi.rs`
- 内置识别提示词：`crates/libsy/src/prompts/rsi/recognize.md`
- MySQL 字典：`crates/switchyard-runner/src/mysql_catalog.rs`
- 配置工厂：`crates/switchyard-runner/src/algorithm.rs`（`RsiRouteConfig`）
- 日志结构：`crates/switchyard-server/src/rsi_trace.rs`
- 端到端测试：`crates/switchyard-server/tests/rsi_router.rs`
- 对外文档：`docs/routing_algorithms/rsi_routing.md`
