# Hybrid inference - agent guide

Read this before touching anything under `examples/hybrid-inference/`, `eval/cluster/`, or the OLS code
that tags requests for routing. The root `AGENTS.md` still governs general OLS code standards; this file
covers what is specific to the hybrid setup.

## What we are trying to do

Serve OpenShift Lightspeed (OLS) from **two backends behind one provider**, so OLS sees a single model
while a gateway picks the hardware per request:

- `cpu-small`: a model served by vLLM on a large CPU node (cheap, slower, good for most `ask` traffic).
- `gpu-large`: a model served by vLLM on one GPU (fast, used where it matters).

The goal is to keep as much work as possible on CPU and move to the GPU only when it pays off: for
`troubleshooting` mode, for the final answer of an `ask`, and once tool output has made the prompt large.
Accuracy and latency of each setup are measured with the experiment runner in `eval/`.

## How a request flows

```
user question -> OLS (/v1/query) -> tool loop: one LLM call per round
                                        |
                          POST /v1/chat/completions (headers say what the call is)
                                        v
                                     Praxis  -- routes on headers -->  cpu-small  or  gpu-large
```

- OLS sees one provider (`praxis`, type `rhoai_vllm`, model `hybrid-model`). Both vLLM services serve the
  same `--served-model-name`, so the routing is invisible to it.
- **Each tool-calling round is a separate HTTP request.** Praxis has no memory across rounds, and every
  round resends the whole growing prompt (system prompt, history, earlier tool calls and results).
- In `ask` mode, documentation is not pre-retrieved. It arrives as the result of a Solr docs-search tool
  call, so tool results are mostly retrieved docs and they drive prompt growth.

## The routing contract (headers set by OLS)

Praxis matches header values **exactly** (no regex, no numeric comparison). Keep this in mind before
designing any threshold logic in the gateway.

| Header | Set by | Meaning |
|---|---|---|
| `X-OLS-Mode` | `DocsSummarizer._query_mode_llm_params` (`ols/src/query_helpers/docs_summarizer.py`) | `ask` or `troubleshooting`; fixed per client |
| `X-OLS-Handoff: true` | `LLMExecutionAgent` (`ols/src/query_helpers/llm_execution_agent.py`) | per call: the forced last round after tool calling, or once the tool budget used reaches a threshold |

Names and defaults live in `ols/constants.py` (`QUERY_MODE_HEADER`, `HANDOFF_HEADER`,
`HANDOFF_TOOL_BUDGET_FRACTION`). Headers are sent only for `rhoai_vllm` providers.

Routes in `praxis/config.yaml` are first-match: `X-OLS-Handoff: true` -> `gpu-large`, then
`X-OLS-Mode: ask` -> `cpu-small`, then everything else -> `gpu-large`. Calls with no mode header (for
example the app server's LLM readiness check) therefore go to the GPU. Praxis answers `GET /healthz`
and `GET /v1/models` itself.

Things that look like LLM calls but are not tool rounds: history compression and the tool-result
classifier. They use separate clients that carry only `X-OLS-Mode`, so they stay on `cpu-small`.

### The handoff threshold

The fraction of the tool token budget (`context window x tool_budget_ratio`) at which calls are handed
off is read from a file, `handoff_tool_budget`, in the provider's credentials directory, falling back to
the default. That makes it a key in the credentials Secret. See the README's "Changing the threshold".
The reason it is not a normal setting: the operator's `OLSConfig` CRD exposes no env vars, volumes or
extra config for the app server (see "Operator constraints").

## Repo map (hybrid-relevant)

| Path | What it is |
|---|---|
| `examples/hybrid-inference/` | Kustomize app: `cpu-small/`, `gpu-large/` (KServe InferenceService, ServingRuntime, PVC), `praxis/` (gateway config + Deployment), `ols/olsconfig.yaml`, `build/buildconfig.yaml` |
| `examples/hybrid-inference/README.md` | Setup, node labels, routing test, custom image, handoff docs |
| `ols/src/query_helpers/docs_summarizer.py`, `llm_execution_agent.py` | Where requests are tagged and the tool loop runs |
| `ols/constants.py`, `ols/app/models/config.py` | Header names, defaults, provider/credentials config |
| `eval/` | Evaluation datasets, system configs and the cluster experiment runner |
| `eval/cluster/` | Job manifests, `experiments.yaml`, runner image build; see `eval/cluster/README.md` |
| `eval/scripts/` | `run_experiments.py` (runner), `build_dashboard.py` |
| `.github/workflows/build-ols-image.yaml` | Builds the OLS image; pushes to GHCR on pushes, build-only on PRs |
| `tests/unit/query_helpers/` | Tests for the tagging and tool loop |

## Cluster layout

Everything runs in the `openshift-lightspeed` namespace on OpenShift (the dev cluster is ROSA).

- **Nodes:** one labeled `inference=cpu-small` and one labeled `inference=gpu-large`, each tainted
  `inference=<value>:NoSchedule`. The InferenceServices select the label and tolerate the taint, so
  nothing else should land on them. A taint does not evict pods already running there, and workloads
  scheduled before the taint was applied stay put until evicted.
- **Operator-owned:** the app server (`lightspeed-app-server`), its ConfigMap and the OLS CSV.
- **Ours:** the two InferenceServices, Praxis (`praxis` Deployment, `praxis-config-<hash>` ConfigMap),
  the OLS `BuildConfig` and ImageStream, and the eval runner objects.

## Common tasks

**Deploy or update the stack:** follow the README. Before `oc apply -k examples/hybrid-inference`, run
`oc diff -k examples/hybrid-inference`. The live InferenceServices are often changed by the experiment
runner (model, memory, parsers), so a blanket apply can overwrite them and reload a model. When you only
changed Praxis, apply just its ConfigMap and Deployment. The ConfigMap name carries a content hash, so a
changed `praxis/config.yaml` rolls the Deployment by itself.

**Build and deploy a dev OLS image on the cluster:**
1. Push your branch to GitHub. The `BuildConfig` clones from Git, so unpushed work is not built. Set its
   `ref` to your branch (`oc patch bc lightspeed-service ...`) and put it back to `main` afterwards.
2. `oc start-build lightspeed-service -n openshift-lightspeed` (about 7 minutes). Confirm the build phase
   is `Complete` with `oc get build`; `--follow` can exit 0 even when the build failed.
3. The operator runs `--service-image=...lightspeed-service:hybrid` from the internal registry. The tag
   is reused and pulled only when a pod starts, so restart the app server
   (`oc delete pod -n openshift-lightspeed -l app.kubernetes.io/name=lightspeed-service-api`) and check the
   running image digest matches the build.
Production-style images come from GHCR via CI; the README covers pointing the CSV at one.

**Verify routing end to end:** send a real `ask` question through OLS (port-forward `svc/lightspeed-app-server`
8443 with a bearer token) and compare vLLM request counts before and after. Use the vLLM Prometheus
counters (`vllm:request_success_total`, `vllm:prompt_tokens_total` from `localhost:8080/metrics` inside
each `kserve-container`), not access-log lines, which do not map one to one to calls. A question must use
enough tool results to cross the threshold to exercise a handoff; heavy items from
`eval/eval_data.yaml` do this some of the time because the model chooses its own tool calls.

**Run evaluations:** `eval/cluster/README.md` is the guide (`experiments.yaml` -> ConfigMap -> Job ->
results). Each experiment restarts the CPU model pod. External-endpoint experiments temporarily patch
Praxis and restore it afterwards.

## Operator constraints (why some things look indirect)

- The `OLSConfig` CRD sets only what it defines (provider and model parameters such as
  `toolBudgetRatio`, `maxIterations`, a few `deployment` fields). New OLS settings cannot be passed through
  it, and the app-server Deployment is owned by the operator, so manual edits to it are reverted.
- Changing a Secret the operator references makes it set a `force-reload` annotation and **roll the app
  server**. That includes adding a key to the credentials Secret.
- Prefer solutions that need neither an OLS rebuild nor operator changes: gateway config, existing CRD
  fields, or a file in a mounted Secret.

## Conventions

- **Keep OLS changes small.** Put policy in Praxis config or cluster settings where the gateway can do it;
  change OLS only when the information exists only there (tool-loop state, token usage).
- Praxis cannot compare numbers or read per-query state across calls. Anything that needs a threshold or a
  count must be decided in OLS and sent as an exact-match header value.
- OLS code standards, tests and checks follow the root `AGENTS.md` (`make format`, `make verify`,
  `make test-unit`); tests for the tool loop use `unittest.mock` and live in
  `tests/unit/query_helpers/test_llm_execution_agent.py`.
- This is a fork (`RMalone8/lightspeed-service-hybrid-inference`), not upstream OLS: push to it and open PRs
  against it. The upstream Jira-prefixed commit rule in the root guide does not apply here.
- When hybrid behavior or the cluster workflow changes, update the example README and this file.

## Ask first

Commit and push only when asked. Ask before changing the cluster: labeling or tainting nodes, draining or
evicting pods, patching Secrets or the CSV, changing the build config's `ref`, or applying manifests that
overwrite live InferenceServices. Do not print secret values.
