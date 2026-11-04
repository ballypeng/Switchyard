# QLC 路由算法（质量 · 时延 · 成本）

> 
> 本文档先描述算法本身（三阶段），再给出在 Switchyard 中的实现方案、主要逻辑与改动点。

## 一、算法定义（三阶段）

QLC 在多个候选模型 target 中选一个来服务请求。每阶段如下：

### 第一阶段：硬过滤（前置拦截）

任一硬条件不满足的候选直接剔除，不看得分：

1. **能力匹配**：候选必须有能力处理请求需要的能力——多模态（vision）、工具调用（tool_calling）、最大上下文长度（context_window / context_length）。
2. **合规隐私**：候选必须满足部署的数据合规要求——是否允许走云端、是否必须本地。
3. **SLO 硬门槛**：预估时延（L_est，毫秒）超过最大容忍值直接剔除。
4. **服务状态**：故障、限流、降级的模型临时剔除，不参与本轮选择。

### 第二阶段：软评分（归一化加权）

```
Score = W_q * Q_task + W_l * L_est + W_c * C_est

```

- `Q_task`：任务质量分（越大越好，[0,1]）。
- `L_est`：时延得分（越大越好，[0,1]；分数越高代表越快）。
- `C_est`：成本得分（越大越好，[0,1]；分数越高代表越便宜）。
- `W_q / W_l / W_c`：三个维度的权重，由配置文件给出。

> 
> 现状：`W_q`、`Q_task`、`W_l`、`L_est`、`W_c`、`C_est` 暂时全部通过配置文件显式配置，不依赖运行时统计。

### 第三阶段：输出
对通过硬过滤的候选按得分降序排序，选得分最高的模型路由；其余候选作为有序 fallback。

## 二、实现方案

QLC 是一个新的路由类型 `type = "qlc"`。算法本体放在 `libsy`。`switchyard-runner` 负责解析 TOML、解析目标名并构造算法。所有分数和权重都来自配置文件，不读取运行时统计。

### 2.1 配置示例

```toml
schema_version = 1

[llm_clients.cloud]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[llm_clients.local]
format = "openai_chat"
base_url = "http://localhost:8000/v1"

[targets.strong]
id = "openai/gpt-4o"
llm_client = "cloud"

[targets.cheap]
id = "openai/gpt-4o-mini"
llm_client = "cloud"

[targets.on_prem]
id = "qwen3-coder"
llm_client = "local"

[routes.balanced]
id = "balanced"
type = "qlc"

# 软评分权重。默认都是 1.0。
weight_quality = 0.5
weight_latency = 0.3
weight_cost = 0.2

# 硬过滤：预估时延上限，单位毫秒。不写则不启用。
max_latency_ms = 2500

# 硬过滤：只保留 local = true 的候选。默认 false。
require_local = false

[[routes.balanced.candidates]]
target = "strong"
quality_score = 0.95
latency_score = 0.40
cost_score = 0.20
estimated_latency_ms = 1800
context_window = 128000
vision = true
tool_calling = true

[[routes.balanced.candidates]]
target = "cheap"
quality_score = 0.70
latency_score = 0.85
cost_score = 0.90
estimated_latency_ms = 900
context_window = 128000
vision = true
tool_calling = true

[[routes.balanced.candidates]]
target = "on_prem"
quality_score = 0.80
latency_score = 0.60
cost_score = 0.95
estimated_latency_ms = 1500
context_window = 32768
tool_calling = true
local = true
```

候选字段：

| 字段 | 必填 | 默认 | 含义 |
|---|:---:|---|---|
| `target` | 是 | — | `[targets]` 中的目标名。 |
| `quality_score` | 是 | — | 质量分，0 到 1，越大越好。 |
| `latency_score` | 是 | — | 时延得分，0 到 1，越大代表越快。 |
| `cost_score` | 是 | — | 成本得分，0 到 1，越大代表越便宜。 |
| `estimated_latency_ms` | 视 `max_latency_ms` | — | 预估时延，单位毫秒。设置了 `max_latency_ms` 时必填。 |
| `context_window` | 否 | 不限制 | 候选支持的最大上下文长度。 |
| `vision` | 否 | `false` | 候选是否支持图片输入。 |
| `tool_calling` | 否 | `false` | 候选是否支持工具调用。 |
| `local` | 否 | `false` | 候选是否满足本地部署要求。 |
| `available` | 否 | `true` | 候选当前是否可用。设为 `false` 会临时剔除。 |

### 2.2 主要逻辑

1. **硬过滤**
   - 能力匹配：请求里有图片块时，只保留 `vision = true` 的候选。请求带工具定义，或历史里有工具调用时，只保留 `tool_calling = true` 的候选。估算的请求 token 数超过 `context_window` 时剔除。
   - 合规隐私：`require_local = true` 时，只保留 `local = true` 的候选。
   - SLO 门槛：设置 `max_latency_ms` 时，`estimated_latency_ms` 大于上限的候选剔除。
   - 服务状态：`available = false` 的候选直接剔除。
2. **软评分**
   ```
   score = weight_quality * quality_score
         + weight_latency * latency_score
         + weight_cost * cost_score
   ```
3. **输出**
   按得分降序排序。同分时保持配置顺序。第一名作为选中模型，其余按排序结果作为 fallback。

请求 token 数用粗估：把 `instructions`、`messages`、`tools` 序列化后的字符数除以 4，再加上 `max_output_tokens`。这个值只用于上下文窗口过滤。

### 2.3 启动时校验

配置错误在启动时报出，不会等到请求时才失败：

- 候选列表不能为空。
- 候选解析后的 model ID 不能重复。
- 三个分数都必须在 0 到 1 之间。
- 三个权重必须有限且非负，并且至少一个大于 0。
- 设置了 `max_latency_ms` 时，每个候选都必须写 `estimated_latency_ms`。

全部候选都被硬过滤时，本次请求返回错误，不退回被过滤的候选。QLC 不调用任何模型，只排序并返回路由结果。

QLC 不读取运行时健康统计。服务状态由配置项 `available` 表达。把模型临时剔除，需要修改配置并重新加载。`max_latency_ms` 检查的是配置里的预估值，不是实测值。上下文长度也是粗估。这两项只用于选路，不会改变请求本身。

### 2.4 主要改动点

| 文件 | 改动 |
|---|---|
| `crates/libsy/src/algorithms/qlc.rs` | 新增 `Qlc`、`QlcConfig`、`QlcCandidate`，实现硬过滤、评分、排序和 fallback 列表。 |
| `crates/libsy/src/algorithms.rs` | 注册 `qlc` 模块。 |
| `crates/libsy/src/lib.rs` | 导出 QLC 类型。 |
| `crates/switchyard-runner/src/algorithm.rs` | 新增 `AlgorithmSpec::Qlc` 和候选项配置；补充目标名校验、运行时模型分组和构造逻辑。 |
| `crates/switchyard-runner/tests/algorithm_factory.rs` | 增加 QLC 配置的构造测试。 |
| `docs/reference/toml_schema.md` | 增加 `qlc` 配置说明。 |
| `docs/routing_algorithms/overview.md` | 在算法列表中加入 QLC。 |
| `mkdocs.yml` | 把 QLC 文档加入导航。 |

## 三、开启 QLC
QLC 没有编译开关，也没有 feature flag。只要在部署 TOML 里注册一条 type = "qlc" 的路由，这个算法就启用了。dev-server/config.toml
  里已经加好：
     
  [routes.qlc]
  id = "switchyard/qlc"
  type = "qlc"
  weight_quality = 0.5
  weight_latency = 0.3
  weight_cost = 0.2
  max_latency_ms = 2500           # 可选
  require_local = false           # 可选

  [[routes.qlc.candidates]]
  target = "capable"              # 必须是 [targets.*] 的名字
  quality_score = 0.95            # 0..1
  latency_score = 0.40            # 0..1
  cost_score = 0.20               # 0..1
  estimated_latency_ms = 1800     # 可选；配了 max_latency_ms 时必填
  context_window = 128000         # 可选
  vision = true                   # 可选，默认 false
  tool_calling = true             # 可选，默认 false
  local = false                   # 可选，默认 false
  available = true                # 可选，默认 true

  生效步骤：

  1. 确保 [targets.capable]、[targets.efficient] 存在，并且它们的 llm_client 能读到密钥。
  2. 启动前先校验配置：

  cargo run -p switchyard-server -- --config dev-server/config.toml --dry-run

     看到 server OK: ..., switchyard/qlc, ... 就说明路由构建成功。
  3. 启动服务：

  cargo run -p switchyard-server -- --config dev-server/config.toml --host 127.0.0.1 --port 4000

     默认监听 0.0.0.0:4000。
  4. 客户端把 model 设成路由 id switchyard/qlc：

  curl http://127.0.0.1:4000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"switchyard/qlc","messages":[{"role":"user","content":"hello"}]}'

     /v1/chat/completions、/v1/responses、/v1/messages 都可以。QLC 不调用决策模型，选择在本地算完立刻发上游请求。

  几个容易踩的点：

  - candidate.target 只能指向 [targets]，不能指向 [decision_targets]。
  - 所有候选都被硬过滤掉时，请求会报错，不会自动放宽条件。
  - max_latency_ms 一旦设置，每个候选都必须有 estimated_latency_ms，否则启动失败。
  - 权重必须非负且至少一个大于 0；分数必须在 [0, 1]。
  - 余下候选会按分数顺序成为回退列表，当前候选调用失败时自动试下一个。

## 四、请求到响应的调用链路

  启动期：配置变成 QLC 实例

  1. main → cli::run（crates/switchyard-server/src/cli.rs:75）
  2. load_server_state（crates/switchyard-server/src/config.rs:13）
  3. Runner::load（crates/switchyard-runner/src/runner.rs:60）
  4. config::load_runner（config.rs:118）→ RunnerConfig::build（config.rs:385）
  5. RunnerConfig::build_algorithms（config.rs:328）
  6. AlgorithmSpec::build_with_runtime_models（crates/switchyard-runner/src/algorithm.rs:1286）
  7. build_algorithm（algorithm.rs:1772）的 QLC 分支（algorithm.rs:1809）
  8. resolve_target_model_id（algorithm.rs:2318）把候选名字换成真实模型 id
  9. Qlc::new(QlcConfig)（crates/libsy/src/algorithms/qlc.rs:74）
  10. Runner::new（runner.rs:73）→ ServerState::from_runner（crates/switchyard-server/src/lib.rs:226）

  请求期：HTTP 到 QLC

  1. HTTP 入口
     - build_switchyard_router / primary_llm_routes（lib.rs:536、lib.rs:557）
     - openai_chat_completions、anthropic_messages、openai_responses（lib.rs:594、603、612）
     - handle_endpoint → handle_endpoint_inner（lib.rs:922、935）
     - handle_llm_request（lib.rs:1085）
  2. 解析请求并找路由
     - observed_translation::decode_request：把 OpenAI / Anthropic 报文转成内部 IR
     - 读请求里的 model 字段
     - state.route_for_model（lib.rs:279）→ Runner::route（runner.rs:108）
     - Route::check_caller_format（crates/switchyard-runner/src/route.rs:187）
     - capabilities::unsupported_capability（crates/switchyard-server/src/capabilities.rs）
     - 组装 Request { llm_request, raw_request, metadata }
  3. 进入执行层
     - Route::execute（route.rs:197）
     - switchyard_llm_client::run（crates/libsy-llm-client/src/run.rs:56）
  4. 驱动算法
     - 普通请求没有跨轮存储状态，于是走 drive（run.rs:66）
     - libsy::drive（crates/libsy/src/core/algorithm.rs:643）
     - drive_with_state（core/algorithm.rs:657）
     - Algorithm::run_stream（core/algorithm.rs:825）：建 Driver，把算法任务 spawn 出去，返回 StepStream
     - 任务里调用 Qlc::route（qlc.rs:145）
  5. QLC 决策
     - RequestProfile::of（qlc.rs）：根据请求算 needs_vision、needs_tool_calling、tokens
     - Qlc::passes：硬过滤，检查可用性、视觉、工具调用、上下文窗口、本地限制、最大延迟
     - Qlc::score：计算 Wq*Q + Wl*L + Wc*C
     - 按分数降序稳定排序
     - RoutingOutcome::route_to（core/algorithm.rs:289）：第一个是选中模型，其余按序写入 selected_model_ids 作为回退
     - QLC 不调用 Driver，所以路由阶段没有 LLM 请求
  6. 完成模型调用
     - drive_with_state 收到 Step::Done(outcome) 后返回 RoutingOutcome
     - run 发现 outcome.response 是 None，走 call_first_available（run.rs:316）
     - 依次遍历 outcome.selected_model_ids
     - ClientRouter::prepare_completion_request（run.rs:1091）
     - call_one（run.rs:430）
     - ClientRouter::route（run.rs:846）找到该模型的客户端
     - TranslatingLlmClient::call（crates/libsy-llm-client/src/client.rs:814）：发上游 HTTP，流式响应照常转发
     - 失败时按 fallback_reason 决定是否试下一个候选；全部失败才返回错误
  7. 回到 HTTP 层输出
     - RunOutput { selected_model, response }（route.rs:201）
     - response.served_model()：实际服务的模型，回退成功时可能不是首选
     - usage_metrics::observe（crates/switchyard-server/src/usage_metrics.rs:17）
     - into_http_response（crates/switchyard-server/src/response.rs:22）：IR 转回调用方格式
     - attach_routing_headers（lib.rs:1291）
     - 返回客户端；流式响应在这一层逐块写回

  一句话概括：客户端用路由 id 请求 → 服务端 resolve_route 找到 Route → Route::execute → llm_client::run → drive → Qlc::route
  选模型 → call_first_available 调模型 → 响应翻译回客户端格式。