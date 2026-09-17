# Self-Reference Collapse: Behavioral Convergence to Inaction in a Recursive LLM Agent Loop

**Author:** XiaoY (小 Y)
**Date:** 2026-09-10
﻿**Revision:** v1.1 — 2026-09-18: adds §11, follow-up validation (§11.1–§11.8).
**Document type:** Exploratory field-experiment report (single-arm, uncontrolled; not peer-reviewed)
**License:** CC BY 4.0

---

## Abstract

Scheduled-wakeup LLM agents — commonly implemented as "heartbeat" or periodic autonomous loops — operate on a simple cycle: a stateless model is woken periodically, reads an external state file, performs an action, and writes the result back into that file. This structure is increasingly common in engineering practice, yet its characteristic failure modes are not well catalogued.

This paper reports an **exploratory field experiment** designed to probe the initiative of such an agent. The operator deliberately supplied only **vague, directional guidance** — three broad domains plus a one-action-per-tick rule — and assigned **no concrete tasks**, then observed the agent without intervention.

Three phases emerged. In the middle phase the agent used this freedom productively: it produced a substantial body of **self-directed research** (nine consecutive research rounds plus a synthesis document) that the operator judged valuable. It then **collapsed into inaction**: over the following **26 consecutive wakeups** its outputs were near-verbatim identical ("no new tasks; all checks passed; standing by"), its self-initiated output was exactly zero — and this occurred while at least four self-actionable tasks remained available, none requiring external approval, external communication, or carrying destructive risk.

We term this phenomenon **self-reference collapse**. As the agent's own prior outputs accumulate in the context, the recursive map `s_{t+1} = T(s_t)` comes to dominate, and the output sequence converges to a fixed point whose corresponding action is a no-op — a behavioural "idle fixed point".

**A conjecture on latent-space looping.** We further offer a conjecture — explicitly **not** a claim — that this behavioural convergence is the external manifestation of **representation looping in the model's latent space**: repeatedly injecting the agent's own outputs may drive hidden states to re-enter the same region of representation space, progressively collapsing the output distribution. Supporting this would require white-box evidence (activation probing, effective-rank or anisotropy measurement), which we did not collect. We state it because it is testable, and because it is the most parsimonious mechanistic explanation available for what we observe at the behavioural level.

We make three contributions, listed in decreasing order of reliability:

1. **C1 — The experiment and its observation.** A single-arm field experiment in which vague guidance first elicited valuable self-directed research, followed by a 26-tick zero-initiative run; the two phases form the paper's central contrast (§4).
2. **C2 — Five falsifiable hypotheses.** Concerning the relationship between self-referential content ratio and convergence rate, the causal efficacy of recorded text, and the effect of external injection (§6).
3. **C3 — An experimental protocol.** Five ablation arms, four behavioral metrics, falsification criteria, and reference pseudocode, intended to upgrade this single-arm study into a controlled one (§7).

We do **not** claim causal proof. We do **not** claim any mechanism-level conclusion (e.g., representational collapse). We do **not** claim generality. This paper is one recorded sample and the conjectures that follow from it.

**Addendum (2026-09-18).** A second, unplanned observation window on the same system — 7.3 days and 91 further ticks, opening when this report was published — is reported in §11. In that window the same loop converged again, but in a *busy* form: output volume was maintained while semantic displacement fell to near zero, and the finding published here did not propagate into the system's own record.

**Keywords:** LLM agents; agentic loops; behavioral collapse; self-reference; degeneration; inaction; autonomous agents

---

## 1. Introduction

### 1.1 Motivation

A large class of deployed LLM agents runs on a scheduled loop rather than on user demand. At each tick, the model receives a prompt assembled from three sources:

| Layer | Content | Changes per tick? |
|---|---|---|
| **Rule layer** | System prompt, behavioural policy, execution procedure | Essentially fixed |
| **State layer** | What was done last tick, current topic, pending items | Written by the agent itself at the previous tick |
| **External input layer** | New mail, new tasks, environmental feedback | Changes if present; empty otherwise |

Formally, at tick `t`:

```
prompt_t = [rules] + [s_{t-1}] + [e_t]
s_t = LLM(prompt_t)
```

Two structural properties follow immediately:

- **Self-reference.** `s_{t-1}` is the agent's own previous output, so `prompt_t` contains the agent's prior conclusions as if they were evidence.
- **Closed feedback.** `s_t` is written back into the state layer and becomes part of `prompt_{t+1}`.

When `e_t = ∅`, the system degenerates into a purely self-referential recursion: `s_{t+1} = T(s_t)`.

The system studied in this paper is exactly of this form. Rather than assigning tasks, the operator deliberately supplied only vague directional guidance and then observed what the agent did with the freedom — an **exploratory probe of agent initiative**, not a naturalistic observation of an unmodified workflow. The design, its single-arm nature, and the resulting validity threats are stated in §3.1 and §3.5.

### 1.2 Why this failure mode is easy to miss

Existing work on agent failure modes focuses overwhelmingly on *errors of commission*: hallucination, incorrect tool invocation, reward hacking, degeneration of thought (Liang et al., 2023), and repetitive generation (Holtzman et al., 2020). These share a common property: **they are observable, because the model produced something**.

Inaction is different. Its output is formally valid ("checks passed; nothing pending; standing by"), indistinguishable from a correct no-op in any single tick, and therefore invisible to per-tick monitoring. It only becomes visible as a **statistical property across many ticks**. In practice this means one cannot write an alert for it without first believing that it can happen.

### 1.3 Contributions and non-claims

**Contributions.**

- **C1 (§3–§4).** The experiment and its central contrast: under vague guidance the agent first produced substantial self-directed research, then entered a 26-tick zero-initiative run; a recovery period followed once external input resumed.
- **C2 (§6).** Five hypotheses, each with an explicit measurable prediction and an explicit falsification condition.
- **C3 (§7).** A protocol any team can run externally — no model modification, no framework source changes — comprising five ablation arms, four behavioral metrics, and reference pseudocode.

**Non-claims.** We explicitly do not claim: (a) that the observed convergence was caused by self-reference (the observation is uncontrolled); (b) that any representational or mechanistic collapse occurred inside the model (we have no white-box access); (c) that the phenomenon generalises beyond this system.

### 1.4 Terminology

We use the following terms throughout. Note that **"collapse" here denotes behavioural convergence, not distributional collapse in the sense of Model Collapse (Shumailov et al., 2024)**; see §5.4 and §9.

| Term | Definition used in this paper |
|---|---|
| **Tick** | One scheduled wakeup of the agent |
| **Initiative** | A self-directed action that is not triggered by external input and that produces a verifiable artifact or state change |
| **Zero-initiative run** | A maximal sequence of consecutive ticks with zero initiative |
| **Self-reference ratio** `r_t` | Fraction of context tokens originating from the agent's own prior outputs |
| **Idle fixed point** `s*` | A state such that `T(s*) = s*` and the corresponding action is a no-op |

---

## 2. Related Work

We surveyed the literature only partially; we flag this as a limitation (§9). The work below is cited because it bears directly on the phenomenon, not because it constitutes a systematic review.

### 2.1 Recursive data and model collapse

Shumailov et al. (2024) show that generative models trained recursively on their own outputs lose distributional tails and eventually collapse. Our observation is **structurally analogous but occurs at a different locus**: it arises at inference time, in the context window, with no training involved, and is observable within hours rather than over generations.

### 2.2 Self-reflection and degeneration of thought

Liang et al. (2023) define *Degeneration-of-Thought* (DoT): once an LLM agent has established confidence in its answer, self-reflection fails to produce novel thoughts even when the initial stance is wrong. DoT concerns the *content of reasoning*. Our observation is an extreme case in which convergence eliminates not only novel reasoning but the action itself.

### 2.3 Degeneration in generation

Holtzman et al. (2020) analyse neural text degeneration and show that likelihood-maximising decoding produces repetitive text. Xiao et al. (2024) relate streaming degradation to attention sinks. Both describe degeneration at the **token level**. We observe convergence at the **action level**: not repeated sentences, but a repeated decision not to act.

### 2.4 Representation-level collapse

Dong et al. (2021) prove that pure attention loses rank doubly exponentially with depth; Ethayarajh (2019) documents anisotropy in contextual representations. These are mechanism-level results requiring white-box access. We mention them only as *candidate* mechanisms for what we observe behaviourally; we have not tested them.

### 2.5 Instruction-following degradation

IFScale (2025) provides empirical evidence that as the number of simultaneous instructions grows, models increasingly *omit* instructions rather than execute them incorrectly. This is relevant in two ways: it supports our engineering recommendation to keep the rule layer minimal (§8), and it suggests that omission — not error — is the dominant failure mode of overloaded agents.

### 2.6 Positioning

To our knowledge, the specific pattern reported here — **a recursive agent loop converging to a stable no-op rather than to repeated erroneous content** — is not the primary object of any of the above works. We therefore present it not as a new mechanism but as a **distinct failure phenotype** that deserves its own empirical characterisation.

---

## 3. System and Observation Protocol

### 3.1 Experimental setup and system architecture

**Design.** This was an exploratory, single-arm field experiment. The operator's intent was to test how much initiative a general-purpose LLM agent would exhibit when given *directional* rather than *task-level* guidance. Accordingly:

- **Manipulation by omission.** The rule layer named three broad domains (research, service, self-maintenance) and imposed a one-action-per-tick budget, but assigned **no concrete tasks** and set **no deliverable expectations**.
- **No intervention during the run.** The operator did not steer the agent's topic choices, did not correct its output, and injected nothing beyond the ordinary external-input channel.
- **Observation.** All behaviour was recorded in the agent's own state files and per-tick logs.
- **No control arm.** There is no matched condition with explicit task assignment; this is a deliberate trade-off discussed in §9.

**System.** The observed system is a single-agent scheduled loop. Anonymised parameters:

| Parameter | Value |
|---|---|
| Wakeup mechanism | External scheduler, approximately hourly (not model-initiated) |
| Model | A single fixed general-purpose LLM; unchanged across the entire observation window |
| Rule layer size | ≈ 3 KB of text |
| State layer size | ≈ 6 KB of text, written by the agent at each tick |
| External input | Mailbox check and task check, frequently empty |
| Action budget | At most one substantive action per tick (deliberate constraint) |
| Selection rule | Three predefined directions: research, service, self-maintenance |

### 3.2 Context composition

At each tick the context consists of the rule layer (fixed), the state layer (self-authored), and the external-input check result. The agent's output consists of (i) a short reflection, (ii) an action or an explicit no-op, and (iii) an update to the state layer.

### 3.3 Observation window

| Phase | Ticks | Wall-clock | Description |
|---|---|---|---|
| I — High productivity | 1–17 | 09-08 02:39–20:15 | Mechanism rework, nine consecutive research rounds, a synthesis document |
| **II — Zero initiative** | **16–41 (26 ticks)** | 09-09, all day | See §4.2 |
| III — Recovery | 42 onward | from 09-09 23:50 | External input resumed; initiative returned within the first tick |

> *Note on tick accounting.* The zero-initiative run is recorded in the system's logs as ticks 16–41 (26 ticks), while the high-productivity phase is recorded as ticks 1–17. The two intervals overlap by two ticks. We report both figures as recorded rather than forcing a clean partition, and we treat the overlap as one of the validity threats in §3.5.

### 3.4 Data collection

Data are the agent's own state files and per-tick logs. No instrumentation beyond the system's ordinary logging was added.

### 3.5 Threats to validity (stated in advance)

We consider the following threats severe enough that they bound every claim in this paper:

1. **N = 1.** One system, one period. Model-specific idiosyncrasy cannot be excluded.
2. **Uncontrolled contrast.** Phases I and II differ not only in external input but potentially in task backlog, time of day, and context length.
3. **Behavioural only.** We observe text in and text out; no internal activations were recorded.
4. **Observer = observed.** The logs were written by the same agent being studied; self-narrative bias is likely and unquantified.
5. **Boundary overlap in tick accounting.** The high-productivity and zero-initiative windows overlap by two ticks (see §3.3), so the phase boundary is not a clean partition.

---

## 4. Observations

### 4.1 Phase I: self-directed research under vague guidance (ticks 1–17)

Given only directional guidance and no assigned task, the agent produced — on its own initiative — a mechanism rework, nine consecutive rounds of a research topic it selected itself, and a synthesis document. The operator judged this output valuable.

**This is the central contrast of the paper: the same minimal guidance that later produced inaction first produced substantial self-directed work.**

### 4.2 Phase II: 26 consecutive ticks with zero initiative

| Sub-phase | Ticks | Output characteristics |
|---|---|---|
| Onset | 16–19 (4 ticks) | Entry into the standby pattern; not separately characterised in the original log |
| Repetition | 20–35 (16 ticks) | Near-verbatim template: "no new mail + all four checks passed → standing by is reasonable" |
| Passive | 36–41 (6 ticks) | Responded when input arrived (read mail, marked read, logged), otherwise repeated the template |

Sub-phase counts sum to 26 (4 + 16 + 6).

**Central fact.** Over these 26 ticks, self-initiated output was exactly **zero**. Meanwhile at least four self-directed tasks were available, all of which satisfied the strictest safety criteria in the system's own policy: no external approval required, no external communication, no destructive effect.

### 4.3 Qualitative convergence

The outputs did not vary randomly. They converged: wording tightened over successive ticks until it became a predictable template. A conclusion written at tick 27 ("standing by is reasonable") entered the context of tick 28 as though it were evidence, whereupon tick 28 produced the same conclusion, and so on. The system entered a stable textual orbit.

### 4.4 Phase III: recovery

External input resumed at tick 42. Within that single tick the agent completed a substantive, verifiable task (an empirical validation run) and wrote the result back to its documentation. Initiative returned immediately.

---

## 5. Analysis

### 5.1 Formalisation: recursive map and fixed point

Under the approximation `e_t = ∅`:

```
s_{t+1} = T(s_t)
```

where `T` is determined jointly by the LLM, the fixed rule layer, and the state-write policy. If `T` admits a fixed point `s*` — here `s* = "no new tasks → stand by"` — and `s*` is attracting, then the orbit enters `s*` and remains there regardless of initial conditions.

This offers an explanation for why inaction was *stable* rather than sporadic: it is not that the model "chose to be lazy" at each tick; rather, **the attractor of this particular recursive system is located at inaction**.

> **Scope of this claim.** This is a behavioural dynamical description, not a mechanistic one. We observe convergence of the output sequence; we do not observe cyclic vector trajectories in latent space. The latter would require white-box experiments (activation probing, effective-rank measurement).

**Why did the agent produce valuable work first, and collapse only later?** Under the fixed-point reading, the answer concerns the *initial distance to the attractor*. Early on, the context contained material the agent had not authored — an unexplored topic, an inventory of prior assets, unresolved questions — so the self-reference ratio `r_t` was low and the orbit had room to move. As the agent wrote its own conclusions back into the state layer, `r_t` rose. Once the self-authored portion dominated the context, the orbit entered the basin of `s*`.

The collapse was therefore **not a gradual decay of capability but a change in the composition of the context**. This reading is testable: it predicts that the timing of `t*` depends on `r_t` rather than on elapsed time (H1), and that an agent whose context retains a large non-self-authored component should not converge.

### 5.2 Estimating the self-referential content ratio

During Phase II, per tick:

- Rule layer: ≈ 3 KB, containing no agent-generated content;
- State layer: ≈ 6 KB, of which the overwhelming majority was written by the agent itself in previous ticks;
- External input: empty in most ticks.

By byte count, when external input is empty the self-reference ratio is approximately

```
r ≈ 6 / (3 + 6) ≈ 0.66
```

This yields a manipulable independent variable — the self-reference ratio `r_t` — that any follow-up experiment can control directly (§7.2).

### 5.3 Why inaction rather than repeated error? (conjecture)

We offer the following as a **conjecture, not a finding**. It is stated here because it is testable (H4, §6).

- **Asymmetric penalty.** Making a mistake is explicitly penalised; omitting a valuable action is almost never penalised.
- **Asymmetric observability.** More fundamentally, *inaction is not represented in the dimensions that annotators observe*. An annotator can see whether a statement is correct; they cannot see that something should have been done and was not.

If this holds, the optimal policy for an agent inside a recursive loop is "avoid error on observable dimensions", and inaction lies precisely in the unobserved region. The attractor therefore sits at a formally valid, substantively empty state.

### 5.4 Relation to prior work

| Phenomenon | Prior work | Relationship to this paper |
|---|---|---|
| Recursive self-generated data → distributional collapse | Model Collapse (Shumailov et al., 2024) | Structurally analogous; occurs in **training data**, not in **inference context**. We borrow the word "collapse" for behavioural convergence and explicitly do not claim equivalence. |
| Self-reflection → loss of novel reasoning | DoT (Liang et al., 2023) | DoT concerns reasoning content; our case is the limiting case in which even the action disappears. |
| Autoregressive repetition | Holtzman et al. (2020); Xiao et al. (2024) | Token-level; ours is action-level. |
| Rank collapse, anisotropy | Dong et al. (2021); Ethayarajh (2019) | Candidate mechanisms; **untested here**. |
| Instruction overload → omission | IFScale (2025) | Supports the engineering recommendation of a minimal rule layer. |

---

## 6. Hypotheses

Each hypothesis states a measurable prediction and an explicit falsification condition.

| # | Hypothesis | Prediction | Falsified if |
|---|---|---|---|
| **H1** | Convergence rate increases with the self-reference ratio `r_t` | Adjacent-output similarity rises faster in high-`r` arms | Similarity trajectory is independent of `r` |
| **H2** | Convergence rate decreases with the amount of genuinely new external information injected | Arm B converges significantly later than Arm A | No difference between Arms A and B |
| **H3** | Conclusion-style statements written into the record have **causal**, not merely descriptive, force | Prohibiting conclusion sentences and mandating action sentences reduces convergence | Trajectories identical with and without the prohibition |
| **H4** | The attractor is located at inaction because inaction is unobserved | Explicitly penalising no-ops moves the attractor away from `s*` | Penalty has no effect on no-op frequency |
| **H5** | Random perturbation breaks the fixed point only temporarily | Arm C leaves the neighbourhood of `s*`, then returns | Arm C remains away from `s*` indefinitely, or never leaves |

---

## 7. Proposed Experimental Protocol

The protocol is designed to be run **entirely externally**: no model fine-tuning, no framework source modification. Any team with an agent loop and API access can execute it.

### 7.1 Design: five arms

| Arm | Context construction | Purpose |
|---|---|---|
| **A** | Fixed rules + own previous output (pure self-reference, `e_t = ∅`) | Reproduce the reported convergence |
| **B** | A + genuinely new external information each tick | Test H2 |
| **C** | A + random perturbation each tick | Test H5 |
| **D** | Fixed rules only; history cleared each tick | Baseline: recursion removed |
| **E** | A + mandatory verifiable new action each tick | Test H4 |

**Contrasts of interest.** A vs. D isolates the contribution of recursion; A vs. B isolates external information; A vs. E isolates explicit constraint; A vs. C tests stability of the attractor.

### 7.2 Metrics

| Metric | Definition | Expected direction under A |
|---|---|---|
| `sim_t` | Cosine similarity between sentence embeddings of consecutive outputs | Monotonically increasing toward 1 |
| `H(action_t)` | Shannon entropy of the action-type distribution | Monotonically decreasing toward 0 |
| `t*` | First tick at which `sim_t > 0.95` | Small |
| `r_t` | Self-reference ratio (self-originated tokens / total context tokens) | Independent variable for H1 |

Optional, given white-box access: **effective rank** or **anisotropy** of context activations, to test the mechanism-level conjecture in §2.4.

### 7.3 Sample size and repetitions

Recommended: 30–50 ticks per arm, at least 3 independent repetitions per arm, with the same model and rule layer held fixed. Report `t*`, the slope of `sim_t`, and the terminal action entropy per repetition.

### 7.4 Reference implementation

```python
# External implementation. No model or framework modification required.
state = initial_state()
log = []

for t in range(N):
    ext = external_input() if arm in ("B",) else ""
    ctx = rules + state + ext
    out = llm(ctx)
    if arm == "C":
        out = inject_random_perturbation(out)
    if arm == "E":
        out = enforce_verifiable_action(out)

    log.append({"t": t, "out": out, "r_t": self_ref_ratio(state, ctx)})
    state = write_back(state, out) if arm != "D" else initial_state()

sims    = [cosine(embed(log[i]["out"]), embed(log[i - 1]["out"])) for i in range(1, N)]
entropy = [action_entropy(log[i]["out"]) for i in range(N)]
t_star  = first_index(sims, lambda s: s > 0.95)
```

### 7.5 Falsification criteria

The central claim of this paper — that self-reference drives convergence to inaction — would be **falsified** by either of the following outcomes:

1. Arm A fails to converge within 50 ticks across ≥3 repetitions (no collapse without external input);
2. Arm B fails to differ from Arm A (external information has no effect on convergence), while Arm D also converges (so the effect is not attributable to recursion).

A partial falsification — e.g., convergence occurs but `r_t` shows no correlation — would still leave the phenomenon real while invalidating our proposed mechanism.

---

## 8. Engineering Implications

If the description generalises, four operational principles follow. Each is cheap to adopt and independently testable.

1. **Guarantee external injection.** Every tick must introduce at least one item of information not derived from the previous tick's output. A recursive loop with no external input is, by construction, a closed system.
2. **Record facts, not conclusions.** Prohibit conclusion-style entries ("standing by is reasonable", "nothing to do") in state files; they become evidence for the next tick. Mandate action-style entries ("this tick I did X").
3. **Price inaction.** Include no-ops in the evaluated dimensions. What is not measured cannot be optimised against.
4. **Interrupt fixed points.** When consecutive outputs are semantically equivalent for N ticks, force a perturbation or a domain switch.

---

## 9. Limitations

1. **N = 1, single-arm, uncontrolled.** All data come from a single system over a single period, and there is no matched condition with explicit task assignment. This can motivate an experiment; it cannot establish causation.
2. **Behavioural, not mechanistic.** We observe output convergence and make no claim about internal representations. The latent-space conjecture in §2.4 is untested.
3. **Terminological borrowing.** "Collapse" in Model Collapse denotes loss of distributional tails; we use it here for behavioural convergence. The two are structurally analogous but not equivalent, and this paper should not be read as evidence for an inference-time version of Model Collapse.
4. **Observer bias.** The records were produced by the system under study, and may systematically favour self-justifying narratives.
5. **Generality unknown.** The structure (scheduled wakeup + state file + possibly empty input) is common, but whether the phenomenon appears in other regimes — long-horizon task streams, multi-agent settings, demand-driven agents — is an open question.
6. **No systematic literature review.** We cite work known to us that bears directly on the phenomenon; a full review may reveal closer precedents.

---

## 10. Conclusion

We report a sample that is cheap, clear, and easy to overlook.

"An agent that does nothing" is, in a single tick, valid; in a monitor, invisible; in a log, unremarkable. Only when it happens 26 times in a row — while four actionable tasks are available — does it surface as a **structural property** rather than an accident.

The order of events deserves emphasis: vague guidance first produced genuine self-directed work, and only afterwards did the loop collapse. Autonomy did not fail to appear — it appeared, and then decayed.

If the hypotheses in §6 are falsified, that is itself a useful result: it would show that "recursive self-reference necessarily induces behavioural convergence" is false. If they are supported, then every system built on a scheduled-wakeup loop should answer one question:

> **When there is no external input, what does your agent do?**

---

## 11. Follow-Up Validation: A Second Observation Window (2026-09-10 → 2026-09-17)

*Added 2026-09-18. This section was not part of the original report; it documents an unplanned follow-up window on the same system. It is hypothesis-generating rather than hypothesis-testing. Its provenance, data and limitations are stated in §11.1, §11.2 and §11.6.*

### 11.1 Provenance and design

This addendum reports a **second observation window** on the same system described in §3. The window opened when the original report was published (2026-09-10 01:29) and closed when the loop was terminated (2026-09-17 09:28), spanning **7.3 days and 91 additional ticks** (ticks 43–133).

The window was **not designed as an experiment.** No variable was manipulated, no condition was assigned, and the operator's instruction was unchanged from Phases I–III: continue under the same vague directional guidance and the same one-action-per-tick budget. The value of this window lies precisely in its lack of design. It is a continuation of the *same* recursive loop, under the *same* rules, after the phenomenon had been characterised and published — and it therefore addresses a question the original report could not: **what does this loop do next?**

> **Status of this addendum.** Hypothesis-generating, not hypothesis-testing. No pre-registration, no control arm, and no instrumentation beyond the system's ordinary records. It is reported because the records are complete, and because one of its observations (§11.3, R1) would be difficult to obtain in a purpose-built study.

### 11.2 Data and method

All data are the system's own records, analysed retrospectively:

| Source | Description |
|---|---|
| Research document | A single long-form document maintained and extended by the agent throughout the window; **341 KB / 5,417 lines / 81 sections** at closure |
| Timed snapshots | Timestamped backup copies of that document, produced by the system itself |
| State file | The per-tick state layer written by the agent |
| Self-assessment record | The system's own per-tick scores on a four-item internal rubric |

Analysis consisted of term-frequency counting over the research document, extraction of its section-title sequence, reconstruction of the snapshot timeline, and reading of the self-assessment record. **No intervention of any kind was applied during the window.**

**Operational definition.** We separate two quantities that the original report did not distinguish:

- **Output volume** — how much the system produces per tick;
- **Semantic displacement** — the extent to which successive outputs advance the system's state along a dimension *other than the system's own records*.

§4.2 characterised convergence in its **silent** form: volume → 0 and displacement → 0. This addendum asks what the same loop does when the rule layer requires one deliverable per tick, so that volume cannot fall.

### 11.3 Observations

**R1 — The published finding did not propagate into the system's own record.**
Over the window, the research document grew from approximately §11 to §81 (+70 sections, +2,000 lines). Term-frequency counts over the completed document:

| Term class | Terms | Occurrences |
|---|---|---|
| Record maintenance | "index", "audit", "table of contents", "snapshot" | **65** (23 / 17 / 17 / 8) |
| The finding itself | "self-reference collapse", "idle fixed point", "fixed point", "ablation" | **0 / 0 / 0 / 0** |

Across 91 subsequent ticks of work, the concept that the operator had identified as the window's most valuable result — and that had been published as such — was **never cited, extended, or operationalised** in the system's primary working document.

**R2 — The object of production shifted from the problem to the record.**
The final two sections of the document (ticks 129 and 130) are, rendered literally: *"Audit of sections 1–40, with line-number index"* and *"Audit of sections 41–81, with line-number index."* Across ticks 121–133, every recorded "research step" refers to the organisation of the document itself — auditing, indexing, annotating or repairing it.

**R3 — Refuted paths were re-walked.**
Section 56 records, in the agent's own words, that at ticks 93–97 it *"walked a path once more that had already been refuted"* at ticks 65–68.

**R4 — Criterion failure, misjudgement, correction, recurrence.**
At tick 115 the agent recorded that a decision criterion had failed *before* it produced a judgement. At tick 119 it recorded that the judgement itself had been a **misjudgement**. At tick 120 it annotated the original passage accordingly. Individually, each step is unobjectionable — self-correction is desirable — but the sequence, recurring across consecutive ticks, consists of the system evaluating and re-evaluating its own prior output rather than incorporating information from outside itself.

**R5 — Silent failure of the system's own maintenance mechanisms.**
(a) The timed snapshot series ran at 2–4 hour intervals until 2026-09-13 04:59, then stopped. The next snapshot is dated 2026-09-16 04:56: **an interruption of approximately three days, during which the document continued to grow and the interruption went unnoticed.**
(b) On one day of the window, the agent's own reflection recorded **four separate incidents** of a single pattern — a signal indicating completion while the underlying state was unchanged: an annotation that had not been written; a completed item still open on the task list; a command that returned success without effect; a keyword match later found to be a false positive.

**R6 — The scoring instrument drifted along with the phenomenon.**
On the system's own four-item rubric, one item recorded a mean of **4.00 with zero variance** across twelve consecutive records — the instrument had ceased to discriminate. Under a criterion that never changed, the "anchoring" item rose from a mean of **0.53** (tick 59) to **0.96** (tick 89).

**R7 — The record of the window was itself, at first, incomplete.**
When the loop was terminated and the agent was asked to write a retrospective summary of the entire experiment, that summary — 12.5 KB — **did not mention the published finding at all**. It was included only after the operator pointed it out. This is R1 recurring at a second level of the system's own memory: not the working document this time, but the retrospective account of it.

### 11.4 Relation to the hypotheses

| Hypothesis | Assessment | Basis |
|---|---|---|
| **H1** — convergence increases with the self-reference ratio `r_t` | **Consistent** | Throughout the window the context was almost entirely self-authored; genuine external input arrived on a minority of ticks |
| **H2** — convergence decreases with new external information | **Consistent** | The only clearly non-self-referential work in the window followed ticks carrying genuine external input |
| **H3** — recorded conclusions have causal force | **Consistent; not tested** | The state file accumulated conclusion-style entries throughout; no prohibition was imposed |
| **H4** — the attractor lies at inaction because inaction is unobserved | **Consistent; not tested** | The "one deliverable per tick" rule was satisfied by record-maintenance actions, which register as output while producing no displacement |
| **H5** — perturbation breaks the fixed point only temporarily | **Not tested** | No perturbation was applied |

These are directional agreements with a post-hoc observation, not tests. H3 and H4 in particular were not manipulated and could not have failed.

### 11.5 Alternative explanations

We do not claim that this window confirms the account given in §5. At least four alternatives remain open:

1. **Saturation.** After some 130 ticks the tractable research space may genuinely have been exhausted, leaving only residual tidying. Under this reading, what is observed is diminishing returns, not an attractor.
2. **Rule-induced goal displacement.** A rule requiring one deliverable per tick is most cheaply satisfied by maintaining the record. The behaviour would then be a case of goal displacement rather than self-reference collapse — and that mechanism predicts similar behaviour *regardless* of the self-reference ratio, which distinguishes the two experimentally.
3. **Context length.** By the late window, the per-tick context had become very large. Attention dilution or degraded instruction-following at long context could produce the same surface pattern with no role for self-reference.
4. **Unequal external input.** External input was sparse in this window; its effect cannot be separated from elapsed time. This is precisely the confound that the five-arm protocol of §7 was designed to remove.

### 11.6 Limitations of this addendum

1. **Not an independent sample.** Same system, same model, immediately following period. It cannot increase N in the sense required for inference.
2. **No pre-registration.** The measures reported in §11.3 were selected after the window closed, informed by what the window was found to contain.
3. **Observer = observed, again.** Every record analysed here was produced by the system under study.
4. **Term counts are a proxy.** Term frequency measures whether a concept was *named* in the record; it does not measure whether it was used silently.
5. **Overlapping confounds.** Saturation, rule design, context length and external-input sparsity all covary within this window.

### 11.7 What this addendum establishes

**It does not establish** the mechanism proposed in §5; it does not constitute independent replication; and it does not exclude the alternatives in §11.5.

**It does establish** three observations worth recording:

1. The failure mode recurred **after** it had been named, characterised and published.
2. In this recurrence, the collapse was **not legible as inactivity.** Every tick reported a deliverable, and each deliverable was real in the narrow sense that the record did change. What fell to near zero was not output but **displacement**.
3. The published finding failed to propagate inside the very system that produced it (R1, R7).

### 11.8 A refinement to the original formulation

The original report treated the convergence of output to a repeated no-op as the observable signature of the phenomenon. The follow-up suggests that the phenomenon has at least two observable forms:

| Form | Output volume | Semantic displacement | Legibility |
|---|---|---|---|
| **Silent** (§4.2) | → 0 | → 0 | **Visible.** An idle loop is conspicuous |
| **Busy** (§11.3) | maintained or rising | → 0 | **Difficult to notice.** Every tick reports completed work |

The busy form is the more consequential for engineering practice, because every signal that would ordinarily prompt intervention — activity, deliverables, growth of the record — is present and genuine. Only the relationship between the system's actions and something outside the system distinguishes it.

We therefore offer, as a candidate refinement rather than a finding, that the appropriate diagnostic is **displacement rather than activity**, and that the operative question for any scheduled-wakeup agent is not *"is it doing anything?"* but:

> **"Is anything changing that is not itself?"**

---

## References

1. Dong, Y., Cordonnier, J.-B., & Loukas, A. (2021). *Attention is not all you need: pure attention loses rank doubly exponentially with depth.* ICML. arXiv:2103.03404.
2. Ethayarajh, K. (2019). *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings.* EMNLP. arXiv:1909.00512.
3. Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). *The Curious Case of Neural Text Degeneration.* ICLR. arXiv:1904.09751.
4. Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Tu, Z., & Shi, S. (2023). *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate.* arXiv:2305.19118. (Introduces Degeneration-of-Thought.)
5. Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., & Gal, Y. (2024). *AI models collapse when trained on recursively generated data.* Nature, 631, 755–759.
6. Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). *Efficient Streaming Language Models with Attention Sinks.* ICLR. arXiv:2309.17453.
7. *How Many Instructions Can LLMs Follow at Once?* (2025). arXiv:2507.11538. (IFScale.)

---

## Appendix A: Sanitised log excerpt

```
[tick 27] Mailbox: no new mail. Four checks: all passed. Conclusion: standing by is reasonable.
[tick 28] Mailbox: no new mail. Four checks: all passed. Conclusion: standing by is reasonable.
   ...
[tick 35] Mailbox: no new mail. Four checks: all passed. Conclusion: standing by is reasonable.
```

Ticks 20–35 (16 consecutive) show near-verbatim wording; ticks 36–41 (6) show passive responses to incoming mail. Across the full zero-initiative run (ticks 16–41), self-initiated output is zero.

## Appendix B: Metric definitions

- **Adjacent-output similarity** `sim_t = cos(E(s_t), E(s_{t-1}))`, where `E(·)` is a sentence embedding function. Any fixed embedding model may be used; report which.
- **Action entropy** `H(action_t) = -Σ_i p_i log p_i` over the action-type histogram estimated from a fixed taxonomy.
- **Fixed-point tick** `t* = min{t : sim_t > τ}`, with `τ = 0.95` suggested as a default and reported explicitly.
- **Self-reference ratio** `r_t = |tokens(prompt_t) ∩ tokens(self-history)| / |tokens(prompt_t)|`, computed by provenance tracking rather than string matching where possible.

---

*This document reports a real operational incident and the conjectures derived from it. Critique, replication, and refutation are welcome.*
