# FE-LLM 规范架构方案

## 1. 设计原则

FE-LLM 的工程命名应使用机器学习、认知建模和主动推理领域的规范术语，不在变量、类名、模块名中直接使用哲学隐喻词。

哲学思想只作为设计来源，不作为代码命名来源。

推荐命名来源：

- active inference
- predictive coding
- energy-based modeling
- belief state
- policy selection
- expected free energy
- traceability
- continual learning

不推荐进入代码的命名：

- `Dao`
- `Yi`
- `WuWei`
- `Gua`
- `Yao`
- `De`
- 其他难以被同行直接理解的文化隐喻名

这些概念可以在论文、白皮书、设计文档中解释，但代码中应转译为标准工程对象。

## 2. 总体架构

FE-LLM 应采用六层结构：

```text
User Prompt
  -> Observation Layer
  -> Perception Layer
  -> Predictive Inference Layer
  -> Policy Planning Layer
  -> Action Realization Layer
  -> Trace and Growth Layer
```

核心思想：

> 输入不是被直接续写的文本，而是一个 observation。系统先评估它如何扰动当前 belief state，再选择能降低 expected free energy 的 action，最后把 action 实现为语言输出。

## 3. 目录结构建议

第一版建议新增一个独立子包，不破坏现有 `energy_lm` 原型：

```text
fe_llm/
  active_inference/
    __init__.py
    controller.py
    state.py
    observation.py
    perception.py
    prediction.py
    belief_update.py
    surprise.py
    policy.py
    free_energy.py
    action.py
    trace.py
    memory.py
```

现有模块可以这样接入：

```text
fe_llm/embedding/
  -> Perception Layer

fe_llm/energy_lm/intent_model.py
  -> Belief / Intent representation prototype

fe_llm/energy_lm/intent_generate.py
  -> Action Realization prototype

fe_llm/energy_lm/intent_train.py
  -> Latent alignment and language realization training
```

## 4. 核心数据结构

### Observation

表示用户输入及其上下文。

```python
@dataclass
class Observation:
    text: str
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### BeliefState

表示模型当前的内部世界状态。

```python
@dataclass
class BeliefState:
    intent_vector: Tensor
    context_vector: Tensor
    confidence: float
    assumptions: list[str]
    unresolved_questions: list[str]
```

### PredictionState

表示模型基于当前 belief 对输入或下一状态的预测。

```python
@dataclass
class PredictionState:
    expected_intent: Tensor
    expected_observation: Tensor | None
    expected_action_outcome: dict[str, Any]
```

### PredictionError

表示输入与预测之间的差异。

```python
@dataclass
class PredictionError:
    semantic_error: float
    intent_error: float
    consistency_error: float
    uncertainty_error: float
    safety_error: float
```

### SurpriseScore

表示 prompt 对当前系统的扰动强度。

```python
@dataclass
class SurpriseScore:
    total: float
    components: PredictionError
    precision_weights: dict[str, float]
```

### CandidateAction

表示候选行动，而不是候选文本。

```python
class ActionType(Enum):
    ANSWER = "answer"
    ASK_CLARIFICATION = "ask_clarification"
    RETRIEVE = "retrieve"
    REFUSE = "refuse"
    UPDATE_MEMORY = "update_memory"


@dataclass
class CandidateAction:
    action_type: ActionType
    intent_vector: Tensor
    rationale: str
    cost: float
```

### ExpectedFreeEnergyScore

表示每个候选行动的评估结果。

```python
@dataclass
class ExpectedFreeEnergyScore:
    risk: float
    ambiguity: float
    epistemic_value: float
    action_cost: float

    @property
    def total(self) -> float:
        return self.risk + self.ambiguity + self.action_cost - self.epistemic_value
```

### InferenceTrace

表示可溯源记录。

```python
@dataclass
class InferenceTrace:
    observation: Observation
    prior_belief: BeliefState
    prediction: PredictionState
    prediction_error: PredictionError
    surprise: SurpriseScore
    candidate_actions: list[CandidateAction]
    action_scores: dict[str, ExpectedFreeEnergyScore]
    selected_action: CandidateAction
    posterior_belief: BeliefState
```

## 5. 运行流程

```text
1. receive_observation(prompt)
2. encode_observation(observation)
3. load_prior_belief(session)
4. predict_expected_state(prior_belief)
5. compute_prediction_error(observation_state, prediction_state)
6. estimate_surprise(prediction_error)
7. update_belief(prior_belief, prediction_error)
8. generate_candidate_actions(posterior_belief, surprise)
9. score_actions_by_expected_free_energy(candidate_actions)
10. select_action(lowest_expected_free_energy)
11. realize_action_as_text(selected_action)
12. record_inference_trace(trace)
13. update_memory_if_needed(trace)
```

## 6. Controller 形态

`ActiveInferenceController` 是整个系统的主入口。

```python
class ActiveInferenceController:
    def respond(self, text: str, session_id: str | None = None) -> ModelResponse:
        observation = self.observation_encoder.encode(text, session_id)
        prior_belief = self.state_store.load(session_id)
        prediction = self.predictor.predict(prior_belief)
        prediction_error = self.error_estimator.compare(observation, prediction)
        surprise = self.surprise_estimator.score(prediction_error)
        posterior_belief = self.belief_updater.update(prior_belief, prediction_error)
        candidates = self.policy_generator.generate(posterior_belief, surprise)
        scores = self.free_energy_scorer.score(candidates, posterior_belief)
        selected_action = self.action_selector.select(candidates, scores)
        text_output = self.action_realizer.realize(selected_action, posterior_belief)
        trace = self.trace_recorder.record(...)
        self.memory_manager.update_if_needed(trace)
        return ModelResponse(text=text_output, trace=trace)
```

## 7. 与现有 intent 架构的关系

当前 `IntentEncoder` 可以保留，但建议定位为：

```text
IntentEncoder = first-stage belief encoder
```

当前 `EnergyDecoder` 可以保留，但建议定位为：

```text
EnergyDecoder = language realization module
```

当前 `L_approach` 很有价值，建议改造成：

```text
belief_stabilization_loss
```

它不只是让 hidden state 靠近 intent，而是让生成过程逐步靠近低自由能状态。

## 8. 训练目标

第一阶段不要只训练 token 续写，应使用组合目标：

```text
total_loss =
    language_modeling_loss
  + latent_prediction_loss
  + belief_alignment_loss
  + stabilization_loss
  + policy_selection_loss
  + trace_consistency_loss
```

其中：

- `language_modeling_loss`：保证语言流畅。
- `latent_prediction_loss`：训练预测编码式表征。
- `belief_alignment_loss`：让 prompt、intent、answer 在 latent space 对齐。
- `stabilization_loss`：鼓励推理过程降低内部能量。
- `policy_selection_loss`：训练何时回答、追问、拒答、检索。
- `trace_consistency_loss`：保证输出解释和内部记录一致。

## 9. 第一版最小可行原型

最小原型不要先追求大模型能力，而是证明系统行为不同于普通语言模型：

```text
输入：用户 prompt
输出：
  - answer text
  - selected action type
  - surprise score
  - prediction error components
  - expected free energy scores
  - trace
```

第一版成功标准：

- 模型知道什么时候不该直接回答。
- 模型能在信息不足时选择追问。
- 模型能在冲突输入中指出矛盾。
- 模型能解释为什么选择当前行动。
- 模型能记录哪些经验可能进入后续成长。

