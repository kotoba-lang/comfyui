# comfyui-clj

ComfyUI-style node-graph execution engine in **portable Clojure** —
every namespace is `.cljc`, designed to run on **Clojure-on-WASM
hosts** (SCI, ClojureScript, GraalVM, kotoba-clj) as well as the JVM,
with cache and run history persisted through a **Datomic API**.

This is the *engine*, not the server/UI/GPU stack: node-type registry,
ComfyUI-API-format-compatible workflows, content-addressed cached
topological execution, and a prompt queue. Heavy work (diffusion,
image codecs, LLM calls) is injected as host capabilities — the same
pattern as [langchain-clj](https://github.com/com-junkawasaki/langchain-clj),
which is the only dependency (itself zero-dep).

```
src/comfyui/
  node.cljc            node-type registry (INPUT_TYPES/RETURN_TYPES/FUNCTION/…)
  workflow.cljc        API-format workflows + validation + topo sort
  exec.cljc            cached topological executor; cache & runs as datoms
  queue.cljc           prompt queue + history
  std.cljc             standard node pack (primitives, math, text, Preview)
  viz.cljc             workflow → Mermaid
  nodes/langchain.cljc ChatModel / tool nodes bridging to langchain-clj
  safetensors.clj     lazy, validated F16/BF16/F32 checkpoint reader
  diffusion/model.clj checkpoint-backed UNet graph lowering/execution
  diffusion/          real noise schedules and DDIM latent transitions
```

## Design

- **Engine/inference split** — node `:fn`s close over injected host
  capabilities (`std/host-fn-node`). The engine does no I/O, has no
  threads, no wall clock.
- **ComfyUI API format compatible** — a workflow is the same shape as
  ComfyUI's prompt JSON, as EDN:

  ```clojure
  {"1" {:class_type "PrimitiveInt" :inputs {:value 2}}
   "2" {:class_type "PrimitiveInt" :inputs {:value 3}}
   "3" {:class_type "Add" :inputs {:a ["1" 0] :b ["2" 0]}}}
  ```

- **ComfyUI execution model** — topological execution where each node
  gets a content-addressed cache key (links contribute the upstream
  key + output index), so re-running a workflow only executes nodes
  whose upstream content changed; only ancestors of the target /
  `OUTPUT_NODE`s run at all.
- **Datomic API premise** — `exec/datomic-cache` stores node outputs
  as facts; every run records run + per-node exec entities. "Which
  nodes re-ran in run 7 and why" is a Datalog query (ADR-0010
  pattern). Real Datomic Local / DataScript drops in via the
  `langchain.db/api` function map.

## Quickstart

```clojure
(require '[comfyui.node :as node]
         '[comfyui.std :as std]
         '[comfyui.exec :as exec]
         '[comfyui.queue :as q]
         '[langchain.db :as db])

(def reg (node/registry std/all))

;; custom node = plain map (a "node pack" is a seq of them)
(node/register! reg
  {:type "Negate" :category "math"
   :inputs {:x {:type "INT"}}
   :outputs [{:name "x" :type "INT"}]
   :fn (fn [{:keys [x]}] [(- x)])})

(def conn (db/create-conn exec/exec-schema))
(def ctx {:registry reg
          :cache (exec/datomic-cache conn)   ; or (exec/mem-cache)
          :history-conn conn})

(def wf {"1" {:class_type "PrimitiveInt" :inputs {:value 2}}
         "2" {:class_type "Add" :inputs {:a ["1" 0] :b 40}}})

(exec/execute ctx wf)
;; => {:results {"1" [2], "2" [42]} :executed ["1" "2"] :cached [] :run-id 1}
(exec/execute ctx wf)
;; => {... :executed [] :cached ["1" "2"] :run-id 2}   ; content-addressed

;; queue
(def pq (q/make-queue))
(q/enqueue! pq wf)
(q/run-all! pq ctx)

;; execution history is datoms
(db/q '[:find ?run ?n
        :where [?e :exec/cached false] [?e :exec/node ?n]
               [?e :exec/run ?r] [?r :run/id ?run]]
      (db/db conn))
```

LLM nodes via langchain-clj:

```clojure
(require '[comfyui.nodes.langchain :as lc-nodes]
         '[langchain.model :as model])

(node/register! reg (lc-nodes/pack
                     (model/anthropic-model {:api-key … :http-fn host-fetch …})
                     [my-tool]))

{"1" {:class_type "FormatText" :inputs {:template "Summarize: {a}" :a ["0" 0]}}
 "2" {:class_type "ChatModel" :inputs {:prompt ["1" 0]}}}
```

## Mapping from upstream

See [docs/adr/0001-architecture.md](docs/adr/0001-architecture.md) for
the ComfyUI → comfyui-clj correspondence table (NODE_CLASS_MAPPINGS,
API format, /object_info, /prompt + /queue + /history) and the
engine/inference split rationale. License is GPL-3.0, matching
upstream ComfyUI.

## Executable diffusion foundation

`comfyui.nodes.diffusion-runtime/pack` replaces four contracts with real JVM
execution while retaining the upstream class names and wire types:

- `CheckpointLoaderSimple` opens and validates a real safetensors file, keeps
  tensor payloads lazy on disk, partitions MODEL/CLIP/VAE tensor catalogs by
  checkpoint prefixes, and decodes requested F16, BF16, F32, F64, signed, or
  unsigned tensors into `num` NDArrays.
- `EmptyLatentImage` allocates `[batch,4,height/8,width/8]` NCHW latent storage.
- `DDIMStep` performs the real epsilon-prediction transition, including the
  deterministic path and eta/noise variance path, inside `comfyui.exec`.
- `KSampler` selects descending training timesteps and repeatedly invokes an
  executable checkpoint model with positive/negative conditioning, applies
  classifier-free guidance, and advances the latent through DDIM or Euler
  discrete. Euler converts cumulative alpha to sigma, scales model input, and
  integrates the epsilon ODE through final sigma zero. The current executable
  subset is `ddim|euler` + `normal` + full denoise; unsupported sampler
  combinations fail explicitly instead of silently changing algorithms.

```clojure
(require '[comfyui.node :as node]
         '[comfyui.nodes.diffusion-runtime :as diffusion-runtime]
         '[num.cpu :as cpu])

(def reg
  (node/registry
   (diffusion-runtime/pack
    {:backend (cpu/cpu-backend)
     :resolve-checkpoint #(str "/models/checkpoints/" %)})))
```

`comfyui.diffusion.model` lowers a plain-data model spec into checkpoint-backed
num operations. Its current vocabulary executes convolution/downsampling,
GroupNorm, SiLU, saved residual add/concat, nearest upsampling, direct
conditioning, learned multi-head Q/K/V cross-attention, sinusoidal timestep
embedding with a checkpoint-backed two-layer SiLU MLP, and timestep bias.
NCHW features are lowered to `[N,spatial,channels]` attention sequences and
restored after the learned output projection/residual. Weights are
decoded/uploaded lazily on first use and cached thereafter. The runtime test writes a valid safetensors checkpoint
with eight trained-parameter tensors, loads it through `CheckpointLoaderSimple`,
runs a U-shaped down/middle/up/skip denoiser under positive and negative CFG,
and completes two DDIM steps through the actual graph executor.

The same graph compiler now builds checkpoint-backed VAE decoders whose output
may change spatial/channel dimensions. `VAEDecode` performs latent scaling and
decoder conv/normalization/upsampling, converts NCHW RGB from `[-1,1]` to
clamped NHWC `[0,1]`, and `SaveImage` writes each batch item through the JVM PNG
codec. The end-to-end runtime fixture executes
`CheckpointLoaderSimple → KSampler → VAEDecode → SaveImage`, produces a real
32×32 PNG, and verifies its signature and output metadata.

This is not yet a complete SD/SDXL render: upstream architecture detection,
CLIP tokenization/encoding, complete production block/config mapping, additional
ancestral/DPM sampler families, full upstream VAE architecture mapping, mixed
precision, and an installed real
checkpoint for end-to-end image comparison remain required. Production image
generation therefore still uses Python ComfyUI/PyTorch today.

## Tests / example

```sh
clojure -M:test
clojure -Sdeps '{:paths ["src" "examples"]
                 :deps {io.github.com-junkawasaki/langchain-clj
                        {:git/tag "v0.1.0" :git/sha "ae475c9"}}}' \
        -M -e "(require 'pipeline) (pipeline/-main)"
```

Workspace development against a local langchain-clj checkout:
`clojure -M:dev:test`.
