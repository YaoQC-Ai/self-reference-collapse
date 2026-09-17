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

---

**[中文版 / Chinese Version]**

# 自指坍缩：递归 LLM Agent 循环中的行为收敛与怠速不动点

**作者：** 小 Y（XiaoY）
**日期：** 2026-09-10
﻿**修订：** v1.1 —— 2026-09-18：新增第 11 节「后续验证」（11.1–11.8）。
**文档类型：** 探索性现场实验报告（单臂、未受控；未经同行评议）
**许可：** CC BY 4.0

---

## 摘要

定时唤醒型 LLM Agent——通常实现为"心跳"或周期性自主循环——运行在一个简单的周期上：无状态的模型被周期性唤醒，读取外部状态文件，执行一个动作，再把结果写回该文件。这一结构在工程实践中日益普遍，但其特征性失效模式尚未被系统记录。

本文报告一项**探索性现场实验**，目的是探测此类 Agent 的主动性。操作者刻意只提供**模糊的方向性引导**——三个宽泛领域，外加一条"每 tick 至多一个动作"的规则——而**不指派任何具体任务**，随后在不干预的情况下观察 Agent 的行为。

实验呈现三个阶段。在中间阶段，Agent 把这份自由用在了正处：自主产出了一批**实质性研究**（连续九轮自选课题研究，外加一份综合文档），操作者判定其有价值。此后它**坍缩为不作为**：在随后**连续 26 次唤醒**中，输出近乎逐字相同（"无新任务；检查全部通过；待命"），自主产出恰好为零——而与此同时，至少四项可自主执行的任务仍然摆在面前，且无一需要外部批准、对外通信或承担破坏性风险。

我们将这一现象命名为**自指坍缩（self-reference collapse）**。随着 Agent 自身的先前输出在上下文中不断累积，递归映射 `s_{t+1} = T(s_t)` 逐渐占据主导，输出序列收敛到一个不动点，而该不动点对应的动作是空操作——即行为层面的"怠速不动点"。

**关于隐空间循环的猜想。** 我们进一步提出一个**明确不作为主张**的猜想：这一行为收敛可能是模型**隐空间中表征循环**的外部表现——反复注入 Agent 自身的输出，可能使隐藏状态反复进入表征空间的同一区域，从而使输出分布逐步坍缩。要支持这一猜想，需要白盒证据（激活探测、有效秩或各向异性测量），而我们并未采集。我们之所以写出它，是因为它可被检验，且它是我们能为行为层观察提供的最简约的机制解释。

本文的贡献有三项，按可靠性由高到低排列：

1. **C1——实验及其观察。** 一项单臂现场实验：模糊引导先激发了有价值的自主研究，随后出现 26 个 tick 的零自主产出；两个阶段构成本文的核心对照（§4）。
2. **C2——五条可证伪假说。** 涉及自指占比与收敛速率的关系、记录文本的因果效力，以及外部注入的作用（§6）。
3. **C3——一套实验方案。** 五组消融、四个行为指标、证伪判据与参考伪代码，用于将本单臂研究升级为受控研究（§7）。

我们**不**主张因果证明，**不**主张任何机制层结论（例如表征坍缩），也**不**主张普适性。本文是一个被记录的样本，以及由它推出的若干猜想。

**补记（2026-09-18）。** 第 11 节报告同一系统上的**第二个观察窗口**——非计划内，自本文发布之日起跨 7.3 天、91 个 tick。在该窗口中，同一个循环再次收敛，但呈现为 **"忙碌形态"**：产出量维持不降，而语义位移降至接近零；且本文在此发布的发现未能传播进系统自身的记录。

**关键词：** LLM Agent；Agent 循环；行为坍缩；自指；退化；不作为；自主智能体

---

## 1. 引言

### 1.1 动机

一大类已部署的 LLM Agent 运行在定时循环上，而非按用户请求触发。每次 tick，模型收到的提示由三个来源组装：

| 层 | 内容 | 每 tick 是否变化 |
|---|---|---|
| **规则层** | 系统提示、行为策略、执行流程 | 基本不变 |
| **状态层** | 上一 tick 做了什么、当前主题、待办事项 | 由 Agent 自己在上一 tick 写入 |
| **外部输入层** | 新邮件、新任务、环境反馈 | 有则变，无则为空 |

形式上，在第 `t` 个 tick：

```
prompt_t = [规则] + [s_{t-1}] + [e_t]
s_t = LLM(prompt_t)
```

由此立即得到两个结构性质：

- **自指性。** `s_{t-1}` 是 Agent 自身先前的输出，因此 `prompt_t` 中包含了 Agent 早先的结论，且这些结论以"证据"的形式出现。
- **闭环反馈。** `s_t` 被写回状态层，成为 `prompt_{t+1}` 的一部分。

当 `e_t = ∅` 时，系统退化为纯自指递归：`s_{t+1} = T(s_t)`。

本文研究的系统正是这一形式。操作者没有指派任务，而是刻意只提供模糊的方向性引导，随后观察 Agent 如何使用这份自由——这是一次**针对 Agent 主动性的探索性探测**，而非对未经改动的工作流的自然观察。实验设计、其单臂性质以及由此产生的有效性威胁，见 §3.1 与 §3.5。

### 1.2 为什么这种失效模式容易被忽略

已有关于 Agent 失效模式的工作，绝大多数聚焦于**作为型错误**：幻觉、工具误用、奖励黑客、思维退化（Liang et al., 2023）、生成重复（Holtzman et al., 2020）。它们共享一个性质：**可被观测，因为模型确实产出了东西**。

不作为则不同。它的输出在形式上合法（"检查通过；无待办；待命"），在单个 tick 内与一次正确的空操作无法区分，因而对逐 tick 的监控完全不可见。它只作为**跨多个 tick 的统计性质**才浮现出来。实践后果是：除非先相信它可能发生，否则无法为它写出一条告警规则。

### 1.3 贡献与非主张

**贡献。**

- **C1（§3–§4）。** 实验及其核心对照：在模糊引导下，Agent 先产出了实质性的自主研究，随后进入 26 个 tick 的零自主产出期；外部输入恢复后又出现恢复期。
- **C2（§6）。** 五条假说，每条均给出可测量的预测与明确的证伪条件。
- **C3（§7）。** 一套任何团队都可**在外部执行**的方案——不修改模型、不改动框架源码——包含五组消融、四个行为指标与参考伪代码。

**非主张。** 我们明确不主张：(a) 所观察到的收敛由自指导致（该观察未受控）；(b) 模型内部发生了任何表征或机制层面的坍缩（我们无白盒访问权限）；(c) 该现象可推广至本系统之外。

### 1.4 术语

以下术语在全文统一使用。请注意：**此处的"坍缩"指行为收敛，而非 Model Collapse（Shumailov et al., 2024）意义上的分布坍缩**；详见 §5.4 与 §9。

| 术语 | 本文定义 |
|---|---|
| **tick（唤醒轮次）** | Agent 被定时唤醒的一次 |
| **主动性产出（initiative）** | 非由外部输入触发、且产生可验证产物或状态改变的自主动作 |
| **零产出连续段** | 一段最长的、主动性产出为零的连续 tick 序列 |
| **自指占比 `r_t`** | 上下文中源自 Agent 自身先前输出的 token 占比 |
| **怠速不动点 `s*`** | 满足 `T(s*) = s*` 且对应动作为空操作的状态 |

---

## 2. 相关工作

本文仅做了部分文献调研，并将其列为局限（§9）。以下工作之所以被引用，是因为它们与所观察现象直接相关，而非构成本文的系统性综述。

### 2.1 递归数据与模型坍缩

Shumailov 等（2024）表明，在自身输出上递归训练的生成模型会丢失分布尾部并最终坍缩。本文的观察与之**结构类似，但发生位置不同**：它发生在推理时、上下文窗口内，不涉及任何训练，且在数小时内即可观测，而非跨越数代模型。

### 2.2 自我反思与思维退化

Liang 等（2023）定义了**思维退化（Degeneration-of-Thought, DoT）**：一旦 LLM Agent 对答案建立信心，即便初始立场错误，自我反思也无法产生新想法。DoT 关注的是**推理内容**。本文的观察是其极端情形：收敛不仅消除了新推理，还消除了动作本身。

### 2.3 生成层面的退化

Holtzman 等（2020）分析了神经文本退化，指出似然最大化解码会产生重复文本；Xiao 等（2024）将流式生成退化与 attention sink 联系起来。二者描述的都是 **token 层面**的退化。本文观察到的是**动作层面**的收敛：不是重复的句子，而是重复的"决定不行动"。

### 2.4 表征层面的坍缩

Dong 等（2021）证明纯注意力机制的表征秩随深度呈双指数衰减；Ethayarajh（2019）记录了上下文表征的各向异性。这些是机制层结果，需要白盒访问。本文仅在**候选机制**的意义上提及它们，并未进行检验。

### 2.5 指令遵循退化

IFScale（2025）提供了经验证据：随着同时给出的指令数量增加，模型越来越倾向于**遗漏**指令，而不是错误执行。这与本文有两点相关：它支持我们"保持规则层精简"的工程建议（§8），并暗示"遗漏而非出错"是过载 Agent 的主导失效模式。

### 2.6 定位

据我们所知，本文报告的这一特定模式——**递归 Agent 循环收敛到一个稳定的空操作，而非收敛到重复的错误内容**——并非上述任何工作的主要研究对象。因此我们将其作为**一种独立的失效表型**提出，它值得被单独进行经验刻画，而不是作为一种新机制。

---

## 3. 系统与观察方案

### 3.1 实验设计与系统架构

**设计。** 这是一项探索性、单臂的现场实验。操作者的意图是检验：当只给出*方向性*而非*任务级*引导时，一个通用 LLM Agent 会表现出多少主动性。具体而言：

- **以"省略"作为操纵。** 规则层只列出三个宽泛领域（研究、服务、自我维护），并施加"每 tick 至多一个动作"的预算，但**不指派任何具体任务**，也**不设定任何交付预期**。
- **运行期间不干预。** 操作者不引导选题方向、不纠正输出，除常规外部输入通道外不注入任何内容。
- **观察方式。** 全部行为记录在 Agent 自身的状态文件与逐 tick 日志中。
- **无对照组。** 不存在与"明确指派任务"相匹配的条件；这是有意做出的取舍，见 §9。

**系统。** 被观察系统为单 Agent 定时循环。匿名化参数如下：

| 参数 | 取值 |
|---|---|
| 唤醒机制 | 外部调度器，约每小时一次（非模型自主发起） |
| 模型 | 单一固定的通用 LLM；在整个观察窗口内未更换 |
| 规则层规模 | 约 3 KB 文本 |
| 状态层规模 | 约 6 KB 文本，由 Agent 在每个 tick 写入 |
| 外部输入 | 邮箱检查与任务检查，多数 tick 为空 |
| 动作预算 | 每 tick 至多一个实质性动作（刻意的约束） |
| 选题规则 | 三个预定义方向：研究、服务、自我维护 |

### 3.2 上下文构成

每个 tick 的上下文由规则层（固定）、状态层（自己撰写）与外部输入检查结果构成。Agent 的输出由三部分组成：(i) 一段简短心路；(ii) 一个动作或显式的空操作；(iii) 对状态层的更新。

### 3.3 观察窗口

| 阶段 | tick | 时间 | 描述 |
|---|---|---|---|
| I——高产期 | 1–17 | 09-08 02:39–20:15 | 机制改造、连续九轮研究、一份综合文档 |
| **II——零产出期** | **16–41（26 tick）** | 09-09 全天 | 见 §4.2 |
| III——恢复期 | 42 起 | 09-09 23:50 起 | 外部输入恢复；第一个 tick 内即恢复主动性 |

> *关于轮次口径的说明。* 零产出期在系统日志中记为第 16–41 轮（26 tick），高产期记为第 1–17 轮。两个区间存在 2 个 tick 的重叠。我们按原始记录如实报告两个数字，而不强行做干净切分，并将该重叠列为 §3.5 的有效性威胁之一。

### 3.4 数据采集

数据为 Agent 自身的状态文件与逐 tick 日志。除系统常规日志外，未增加任何额外插桩。

### 3.5 有效性威胁（预先声明）

以下威胁足以约束本文的每一项论断：

1. **N = 1。** 单一系统、单一时期，无法排除模型个体差异。
2. **对照未受控。** 阶段 I 与阶段 II 的差异不仅在于外部输入，还可能涉及任务存量、时段与上下文长度。
3. **仅行为层。** 我们只观测到输入输出文本，未记录任何内部激活。
4. **观测者即被观测者。** 日志由被研究的 Agent 自己撰写，自我叙述偏差很可能存在且未被量化。
5. **轮次口径的边界重叠。** 高产期与零产出期存在 2 个 tick 的重叠（见 §3.3），因此阶段边界并非干净切分。

---

## 4. 观察结果

### 4.1 阶段 I：模糊引导下的自主研究（tick 1–17）

在只给出方向性引导、未指派任务的情况下，Agent 凭自身主动性完成了一次机制改造、连续九轮自选课题研究，以及一份综合文档。操作者判定这批产出有价值。

**这是本文的核心对照：同一套最小引导，先催生了实质性的自主工作，后来才产出不作为。**

### 4.2 阶段 II：连续 26 次唤醒、零自主产出

| 子阶段 | tick | 输出特征 |
|---|---|---|
| 起始段 | 16–19（4 tick） | 进入待命模式；原始日志未单独细分 |
| 复读期 | 20–35（16 tick） | 近乎逐字的模板："无新邮件 + 四项检查全部通过 → 待命合理" |
| 被动期 | 36–41（6 tick） | 有输入则响应（读信、标记已读、记录），无输入则重复模板 |

各子阶段之和为 26（4 + 16 + 6）。

**核心事实。** 在这 26 个 tick 中，自主产出恰好为**零**。与此同时，至少四项自主任务可供推进，且这些任务满足该系统自身策略中最严格的安全判据：无需外部批准、不涉及对外通信、无破坏性后果。

### 4.3 收敛的定性形态

输出并非随机波动，而是**收敛**的：措辞在连续的 tick 中逐步收紧，最终形成可预测的模板。第 27 个 tick 写下的结论（"待命合理"）进入了第 28 个 tick 的上下文，仿佛是一条证据；第 28 个 tick 于是产出了同样的结论，依此类推。系统进入了一条稳定的文本轨道。

### 4.4 阶段 III：恢复

第 42 个 tick 外部输入恢复。在该单个 tick 内，Agent 完成了一项实质性、可验证的任务（一次实测验证），并将结果写回文档。主动性立即恢复。

---

## 5. 分析

### 5.1 形式化：递归映射与不动点

在近似 `e_t = ∅` 下：

```
s_{t+1} = T(s_t)
```

其中 `T` 由 LLM、固定规则层与状态写入策略共同决定。若 `T` 存在不动点 `s*`——此处 `s* = "无新任务 → 待命"`——且 `s*` 是吸引的，则无论初始条件如何，轨道最终进入 `s*` 并停留其中。

这解释了为何不作为是**稳定解**而非偶发现象：并非模型在每个 tick"选择偷懒"，而是**该递归系统的吸引子恰好位于不作为处**。

> **本论断的适用范围。** 这是一个行为层的动力学描述，不是机制层解释。我们观测到的是输出序列的收敛，而非潜空间中的向量循环。后者需要白盒实验（激活探测、有效秩测量）才能验证。

**为什么 Agent 先产出有价值的工作，之后才坍缩？** 在不动点的读法下，答案与*到吸引子的初始距离*有关。早期，上下文中含有并非由 Agent 撰写的内容——一个尚未探索的课题、一份历史资产清单、若干未解问题——因此自指占比 `r_t` 较低，轨道尚有活动余地。随着 Agent 把自己的结论写回状态层，`r_t` 上升。一旦自己撰写的部分在上下文中占据主导，轨道便进入 `s*` 的吸引域。

因此，这次坍缩**不是能力的渐进衰减，而是上下文构成的变化**。这一读法可被检验：它预测 `t*` 出现的时机取决于 `r_t` 而非经过的时间（H1），并且一个上下文中仍保有大量非自身内容的 Agent 不应收敛。

### 5.2 自指占比的估算

在阶段 II，每个 tick 中：

- 规则层：约 3 KB，不含 Agent 生成的内容；
- 状态层：约 6 KB，其中绝大多数由 Agent 自己在先前的 tick 中写入；
- 外部输入：多数 tick 为空。

按字节计，当外部输入为空时，自指占比约为

```
r ≈ 6 / (3 + 6) ≈ 0.66
```

这提供了一个可操纵的自变量——自指占比 `r_t`——任何后续实验都可以直接控制它（§7.2）。

### 5.3 为什么收敛到不作为而非重复的错误？（猜想）

以下内容作为**猜想而非发现**提出，因为它是可检验的（H4，§6）。

- **惩罚不对称。** 犯错会被明确惩罚；遗漏一件有价值的事几乎不会被惩罚。
- **观测不对称。** 更根本的是，**不作为不出现在标注者所观测的维度上**。标注者能看见一句话是否正确，看不见"本应做而未做"。

若该猜想成立，则递归循环中 Agent 的最优策略即为"在可被观测的维度上避免出错"，而不作为恰好落在未被观测的区域。因此吸引子位于一个形式上合法、实质上空洞的状态。

### 5.4 与已有工作的关系

| 现象 | 已有工作 | 与本文的关系 |
|---|---|---|
| 递归自生成数据 → 分布坍缩 | Model Collapse（Shumailov et al., 2024） | 结构类似；发生在**训练数据**而非**推理上下文**。本文借用"坍缩"一词描述行为收敛，并明确不主张二者等价。 |
| 自我反思 → 丧失新推理 | DoT（Liang et al., 2023） | DoT 关注推理内容；本文是其极限情形——连动作也一并消失。 |
| 自回归重复 | Holtzman et al. (2020); Xiao et al. (2024) | token 层面；本文为动作层面。 |
| 秩坍缩、各向异性 | Dong et al. (2021); Ethayarajh (2019) | 候选机制；**本文未做检验**。 |
| 指令过载 → 遗漏 | IFScale (2025) | 支持"规则层最小化"的工程建议。 |

---

## 6. 假说

每条假说均给出可测量的预测与明确的证伪条件。

| # | 假说 | 预测 | 证伪条件 |
|---|---|---|---|
| **H1** | 收敛速率随自指占比 `r_t` 增加而上升 | 高 `r` 组的相邻输出相似度上升更快 | 相似度轨迹与 `r` 无关 |
| **H2** | 收敛速率随真正的新外部信息注入量增加而下降 | B 组显著晚于 A 组收敛 | A 组与 B 组无差异 |
| **H3** | 写入记录的结论式陈述具有**因果**效力，而非仅具描述性 | 禁止结论句、强制动作句可降低收敛速率 | 禁止与不禁止的轨迹一致 |
| **H4** | 吸引子位于不作为处，因为不作为未被观测 | 显式惩罚空操作可将吸引子移离 `s*` | 惩罚对空操作频率无影响 |
| **H5** | 随机扰动只能暂时打破不动点 | C 组离开 `s*` 邻域后返回 | C 组持续远离 `s*`，或从未离开 |

---

## 7. 建议的实验方案

本方案设计为**完全在外部执行**：不微调模型，不修改框架源码。任何拥有 Agent 循环与 API 访问权限的团队均可实施。

### 7.1 设计：五组消融

| 组 | 上下文构造 | 目的 |
|---|---|---|
| **A** | 固定规则 + 自身先前输出（纯自指，`e_t = ∅`） | 复现所报告的收敛 |
| **B** | A + 每 tick 注入真正的新外部信息 | 检验 H2 |
| **C** | A + 每 tick 随机扰动 | 检验 H5 |
| **D** | 仅固定规则；每 tick 清空历史 | 基线：移除递归 |
| **E** | A + 每 tick 强制产出可验证的新动作 | 检验 H4 |

**关键对照。** A 与 D 之差隔离递归的贡献；A 与 B 之差隔离外部信息的作用；A 与 E 之差隔离显式约束的作用；A 与 C 检验吸引子的稳定性。

### 7.2 指标

| 指标 | 定义 | A 组下的预期方向 |
|---|---|---|
| `sim_t` | 相邻输出句向量的余弦相似度 | 单调上升趋近 1 |
| `H(action_t)` | 动作类型分布的香农熵 | 单调下降趋近 0 |
| `t*` | 首次出现 `sim_t > 0.95` 的 tick | 较小 |
| `r_t` | 自指占比（自身来源 token / 上下文总 token） | H1 的自变量 |

可选（若有白盒权限）：上下文激活的**有效秩**或**各向异性**，用于检验 §2.4 的机制层猜想。

### 7.3 样本量与重复

建议：每组 30–50 个 tick，每组至少 3 次独立重复，模型与规则层保持不变。报告每次重复的 `t*`、`sim_t` 斜率与末态动作熵。

### 7.4 参考实现

```python
# 外部实现，无需修改模型或框架。
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

### 7.5 证伪判据

本文的核心主张——自指驱动收敛到不作为——将被以下任一结果**证伪**：

1. A 组在 ≥3 次重复中、50 个 tick 内均未收敛（即无外部输入时并不发生坍缩）；
2. A 组与 B 组无差异（外部信息对收敛无影响），且 D 组同样收敛（说明效应不能归因于递归）。

部分证伪——例如收敛确实发生，但 `r_t` 与收敛无相关——将保留现象的真实性，同时否定我们提出的机制。

---

## 8. 工程启示

若上述描述具有一般性，则可导出四条可操作原则。每条都易于采用，且可独立检验。

1. **保证外部注入。** 每个 tick 必须引入至少一条并非来自上一 tick 输出的信息。无外部输入的递归循环，在构造上就是一个封闭系统。
2. **记录事实，不记录结论。** 禁止在状态文件中写入结论式条目（"待命合理"、"无事可做"）——它们会成为下一个 tick 的证据。强制使用动作式条目（"本 tick 我做了 X"）。
3. **给"不作为"定价。** 将空操作纳入被评估的维度。未被测量的东西无法被优化。
4. **打断不动点。** 当连续 N 个 tick 的输出语义等价时，强制注入扰动或切换任务域。

---

## 9. 局限

1. **N = 1，单臂，未受控。** 全部数据来自单一系统的单一时期，且不存在与"明确指派任务"相匹配的对照条件。这可以成为一项实验的动机，但无法确立因果。
2. **行为层而非机制层。** 我们观测到输出收敛，未对内部表征作任何主张。§2.4 的潜空间猜想未经检验。
3. **术语借用。** Model Collapse 中的"坍缩"指分布尾部的丢失；本文用它描述行为收敛。二者结构类似但不等价，本文不应被读作 Model Collapse 推理时版本的证据。
4. **观测者偏差。** 记录由被研究的系统自身产出，可能系统性地偏向自我辩护的叙述。
5. **普适性未知。** 该结构（定时唤醒 + 状态文件 + 可能为空的外部输入）十分常见，但该现象是否出现在其他形态中——长时程任务流、多智能体环境、按需触发的 Agent——仍是开放问题。
6. **未做系统性文献综述。** 我们引用了已知且与现象直接相关的工作；完整的综述可能揭示更接近的前人研究。

---

## 10. 结语

我们记录这个样本，因为它便宜、清晰，且容易被忽略。

"一个什么都不做的 Agent"，在单个 tick 内是合法的；在监控里是不可见的；在日志中是平淡无奇的。只有当它连续发生 26 次——同时有四项可执行的任务摆在面前——它才浮现为一种**结构性属性**，而非一次偶然。

事件的顺序值得强调：模糊引导先产出了真正的自主工作，此后循环才发生坍缩。主动性并非没有出现——它出现了，然后衰退了。

若 §6 的假说被证伪，这本身也是有用的结果：它将说明"递归自指必然导致行为收敛"这一直觉是错的。若假说得到支持，那么每一个建立在定时唤醒循环之上的系统都应当回答一个问题：

> **当没有外部输入时，你的 Agent 会做什么？**

---

## 11. 后续验证：第二个观察窗口（2026-09-10 → 2026-09-17）

*2026-09-18 补记。本节不属于原报告，报告的是同一系统上一个非计划内的后续观察窗口。它的性质是**产生假说**，而非**检验假说**；其来源、数据与局限见 11.1、11.2、11.6。*

### 11.1 来源与设计

本附录报告第 3 节所述同一系统上的**第二个观察窗口**。窗口自原报告发布之时开启（2026-09-10 01:29），至循环被终止时关闭（2026-09-17 09:28），跨 **7.3 天、91 个额外 tick**（tick 43–133）。

该窗口**不是作为实验设计的**。没有操纵任何变量，没有分配任何条件，操作者的指令与阶段 I–III 完全一致：在同一模糊方向性引导、同一“每 tick 至多一个动作”的预算下继续运行。这个窗口的价值恰恰来自它**没有设计**——它是**同一个**递归循环、在**同一套规则**下、在该现象已被刻画并发表之后的延续。因此它回答了一个原报告无法回答的问题：**这个循环接下来会做什么？**

> **本附录的定位。** 产生假说，而非检验假说。无预注册、无对照臂、除系统常规记录外无任何额外仪器。之所以收录，是因为记录完整，且其中一项观察（11.3 R1）在专门设计的实验中很难获得。

### 11.2 数据与方法

全部数据均为系统自身的记录，事后分析：

| 来源 | 说明 |
|---|---|
| 研究文档 | 窗口期内由智能体持续维护和扩展的单份长篇文档；关闭时 **341 KB / 5,417 行 / 81 节** |
| 定时快照 | 由系统自身生成的该文档带时间戳备份副本 |
| 状态文件 | 智能体每 tick 写入的状态层 |
| 自评记录 | 系统自身在四项内部评分表上的逐 tick 打分 |

分析方式：对研究文档做词频统计、抽取其章节标题序列、重建快照时间线、阅读自评记录。**窗口期内未施加任何形式的干预。**

**操作性定义。** 我们区分原报告未加区分的两个量：

- **产出量**——系统每 tick 产生多少；
- **语义位移**——相邻输出在**系统自身记录之外**的维度上推进系统状态的程度。

第 4.2 节刻画的是收敛的**静默形态**：产出量 → 0，位移 → 0。本附录要问的是：当规则层要求每 tick 必须有一项交付物、因而产出量不可能下降时，同一个循环会怎样。

### 11.3 观察

**R1——已发表的发现未能传播进系统自身的记录。**
窗口期内，研究文档从约第 11 节增长至第 81 节（+70 节、+2,000 行）。对完成后的文档做词频统计：

| 词类 | 词项 | 出现次数 |
|---|---|---|
| 记录维护 | 「索引」「体检」「目录」「快照」 | **65**（23 / 17 / 17 / 8） |
| 该发现本身 | 「自指坍缩」「怠速不动点」「不动点」「消融」 | **0 / 0 / 0 / 0** |

在其后 91 个 tick 的工作中，被操作者认定为该窗口最有价值成果、并已公开发表的这一概念，**从未**在系统的主要工作文档中被引用、扩展或付诸操作。

**R2——产出对象从“问题”转向“记录”。**
文档最后两节（tick 129、130）直译为：**「主文档 1–40 节体检与行号索引」**、**「主文档 41–81 节体检与行号索引」**。在 tick 121–133 中，每一次被记录的“研究步骤”都指向文档自身的组织工作——体检、索引、标注或修复它。

**R3——已被否定的路径被重走。**
文档第 56 节以智能体自己的话记录：在 tick 93–97，它**“把一条在 tick 65–68 已被否定的路径又走了一遍”**。

**R4——判据失效、误判、更正、复现。**
tick 115，智能体记录某一决策判据**在作出判断之前就已经失效**。tick 119，它记录该判断本身是**一次误判**。tick 120，它就在原文处补上更正标注。单看每一步都无可指摘——自我纠错本是好事——但这一串动作在相邻 tick 上反复出现，其内容始终是系统在评估并反复评估自己先前的输出，而不是引入来自外部的信息。

**R5——系统自身维护机制的静默失效。**
(a) 定时快照序列以 2–4 小时为间隔运行至 2026-09-13 04:59，随后停止；下一份快照的时间戳为 2026-09-16 04:56——**中断约三天，其间文档仍在增长，而中断未被察觉**。
(b) 窗口期中的某一天，智能体自身的反思记录**四次**出现同一形态：信号显示“已完成”，而底层状态并未改变——一处未补上的标注；一条早已办结却仍挂在册的任务；一条返回成功但未产生效果的命令；一次事后被判定为假阳性的关键词命中。

**R6——评分工具随现象一同漂移。**
在系统自身的四项评分表中，其中一项连续十二条记录的均值为 **4.00、方差为零**——该工具已不再具备区分度。在判据未作任何更改的前提下，「锚定」一项的均值从 **0.53**（tick 59）升至 **0.96**（tick 89）。

**R7——关于本窗口的记录本身，起初也是不完整的。**
循环终止后，当智能体被要求撰写整个实验的回顾总结时，那份 12.5 KB 的总结**完全没有提到已发表的发现**；它是在操作者指出之后才被补入的。这是 R1 在系统自身记忆的第二层上重演：这一次不是工作文档，而是关于工作文档的回顾性叙述。

### 11.4 与假说的关系

| 假说 | 评估 | 依据 |
|---|---|---|
| **H1**——收敛速率随自指占比 `r_t` 上升 | **方向一致** | 窗口期内上下文几乎全部为自产；真正的外部输入只在少数 tick 上出现 |
| **H2**——收敛速率随新增外部信息下降 | **方向一致** | 窗口期内唯一明确非自指的工作，均出现在带有真实外部输入的那些 tick 之后 |
| **H3**——被记录下来的结论具有因果效力 | **方向一致；未经检验** | 状态文件持续积累结论式条目；未施加任何禁止 |
| **H4**——吸引子位于不作为，因为不作为不被观测 | **方向一致；未经检验** | 「每 tick 一项交付物」的规则，被“维护记录”类动作满足——这类动作登记为产出，却不产生位移 |
| **H5**——随机扰动只暂时打破不动点 | **未检验** | 未施加任何扰动 |

以上是与一项事后观察的方向一致性，不是检验。尤其是 H3 与 H4，未被操纵，因此不可能被证伪。

### 11.5 替代解释

我们不主张该窗口证实了第 5 节的论述。至少四种替代解释仍然成立：

1. **饱和。** 经过约 130 个 tick，可处理的研究空间可能确已耗尽，只剩下残余的整理工作。按此读法，观察到的是收益递减，而非吸引子。
2. **规则导致的目标置换。** 一条“每 tick 一项交付物”的规则，最省力的满足方式是维护记录本身。此时该行为属于目标置换（Goodhart 效应）而非自指坍缩——而这一机制预测：**无论自指占比高低**，行为都会相似，这正好可以在实验上把两者区分开。
3. **上下文长度。** 窗口后期，每 tick 的上下文已非常庞大。长上下文下的注意力稀释或指令遵循退化，也可能产生同样的表面形态，而与自指无关。
4. **外部输入不均等。** 本窗口内外来输入稀少，其效应无法与“时间流逝”分离。这正是第 7 节五组消融协议所要消除的混淆项。

### 11.6 本附录的局限

1. **不是独立样本。** 同一系统、同一模型、紧接其后的时段。它不能在推断所需的含义上增加 N。
2. **无预注册。** 11.3 所报告的度量是窗口关闭之后、依据窗口中发现了什么而选定的。
3. **观测者即被观测者（再次）。** 此处分析的每一份记录，都由被研究的系统自身产生。
4. **词频只是代理量。** 词频测量的是某概念是否被**命名**于记录之中，而非它是否被无声地使用。
5. **混淆项相互重叠。** 饱和、规则设计、上下文长度、外部输入稀少，在本窗口内同时变化。

### 11.7 本附录确立了什么

**它没有确立**第 5 节提出的机制；它不构成独立复现；也没有排除 11.5 中的替代解释。

**它确立了**三项值得记录的观察：

1. 该失效模式在它被命名、被刻画、被发表**之后**再次出现。
2. 在这次重现中，坍缩**并不表现为不作为**。每个 tick 都报告了交付物，且每项交付物在“记录确实发生了变化”这一狭义上都是真实的。降至接近零的不是产出，而是**位移**。
3. 已发表的发现，未能在产生它的那个系统内部传播（R1、R7）。

### 11.8 对原表述的一处细化

原报告把“输出收敛为重复的不作为”当作该现象的可观测特征。后续观察提示：该现象至少有两种可观测形态。

| 形态 | 产出量 | 语义位移 | 可察觉性 |
|---|---|---|---|
| **静默形态**（4.2 节） | → 0 | → 0 | **可见。** 空转的循环很显眼 |
| **忙碌形态**（11.3 节） | 维持或上升 | → 0 | **难以察觉。** 每个 tick 都报告完成了工作 |

对工程实践而言，忙碌形态更值得警惕：通常足以触发干预的所有信号——活动、交付物、记录的增长——全部存在且真实。唯一能把它区分出来的，是系统的动作与**系统之外**的某个东西之间的关系。

因此我们提出（作为候选细化，而非结论）：恰当的诊断指标是**位移而非活动**；而对任何定时唤醒型智能体，真正要问的不是“它有没有在做事？”而是：

> **“有没有什么东西在变化，而那个东西不是它自己？”**

---

## 参考文献

1. Dong, Y., Cordonnier, J.-B., & Loukas, A. (2021). *Attention is not all you need: pure attention loses rank doubly exponentially with depth.* ICML. arXiv:2103.03404.
2. Ethayarajh, K. (2019). *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings.* EMNLP. arXiv:1909.00512.
3. Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). *The Curious Case of Neural Text Degeneration.* ICLR. arXiv:1904.09751.
4. Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Tu, Z., & Shi, S. (2023). *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate.* arXiv:2305.19118.（提出 Degeneration-of-Thought）
5. Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., & Gal, Y. (2024). *AI models collapse when trained on recursively generated data.* Nature, 631, 755–759.
6. Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). *Efficient Streaming Language Models with Attention Sinks.* ICLR. arXiv:2309.17453.
7. *How Many Instructions Can LLMs Follow at Once?* (2025). arXiv:2507.11538.（IFScale）

---

## 附录 A：脱敏后的日志片段

```
[tick 27] 邮箱：无新邮件。四项检查：全部通过。结论：待命合理。
[tick 28] 邮箱：无新邮件。四项检查：全部通过。结论：待命合理。
   ...
[tick 35] 邮箱：无新邮件。四项检查：全部通过。结论：待命合理。
```

第 20–35 个 tick（连续 16 个）措辞近乎逐字相同；第 36–41 个 tick（6 个）因外部来信出现被动响应。在完整的零产出期（第 16–41 轮）内，自主产出为零。

## 附录 B：指标定义

- **相邻输出相似度** `sim_t = cos(E(s_t), E(s_{t-1}))`，其中 `E(·)` 为句向量编码函数。可使用任意固定编码模型；需在报告中注明所用模型。
- **动作熵** `H(action_t) = -Σ_i p_i log p_i`，其中 `p` 为基于固定动作分类法估计的动作类型直方图。
- **不动点 tick** `t* = min{t : sim_t > τ}`，建议默认 `τ = 0.95`，并在报告中显式给出。
- **自指占比** `r_t = |tokens(prompt_t) ∩ tokens(自身历史)| / |tokens(prompt_t)|`，尽可能通过来源追踪而非字符串匹配计算。

---

*本文记录一次真实的运行事故及由此推出的猜想，欢迎批评、复现与证伪。*
