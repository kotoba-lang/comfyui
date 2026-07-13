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
  safetensors.clj      lazy, validated JVM checkpoint reader
  safetensors_deno.cljs direct validated Deno/Metal checkpoint reader
  diffusion/model.cljc checkpoint-backed UNet/VAE graph lowering/execution
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

`comfyui.nodes.diffusion-runtime/pack` replaces six contracts with real JVM
execution while retaining the upstream class names and wire types:

- `LoadImage` decodes an input-directory-confined image through ImageIO into
  batched NHWC RGB `[0,1]` and emits ComfyUI's inverse-alpha mask. Canonical
  path checks reject traversal and symlink escapes.
- `CheckpointLoaderSimple` opens and validates a real safetensors file, keeps
  tensor payloads lazy on disk, partitions MODEL/CLIP/VAE tensor catalogs by
  checkpoint prefixes, and decodes requested F16, BF16, F32, F64, signed, or
  unsigned tensors into `num` NDArrays.
- `EmptyLatentImage` allocates `[batch,4,height/8,width/8]` NCHW latent storage.
- `DDIMStep` performs the real epsilon-prediction transition, including the
  deterministic path and eta/noise variance path, inside `comfyui.exec`.
- `KSampler` selects descending training timesteps and repeatedly invokes an
  executable checkpoint model with positive/negative conditioning, applies
  classifier-free guidance, and advances the latent through DDIM, Euler
  discrete, or Euler ancestral. Euler converts cumulative alpha to sigma,
  scales model input, and integrates the epsilon ODE through final sigma zero;
  the ancestral variant splits each target into sigma-down plus seeded
  sigma-up noise. DPM++ 2M follows k-diffusion's exponential multistep
  recurrence, retains the previous denoised estimate for its second-order
  correction, and handles terminal sigma zero as an exact denoised-x0 step.
  Euler and DPM++ additionally accept the Karras rho-7 sigma schedule. Continuous
  sigma values are mapped back to fractional model timesteps by interpolation in
  log-sigma space, matching k-diffusion's discrete epsilon-model wrapper.
  The current executable subset is `ddim` + `normal`, or
  `euler|euler_ancestral|dpmpp_2m` + `normal|karras`. Denoise values in `(0,1]`
  are supported: the runtime builds `floor(steps/denoise)` levels and retains
  the final requested step interval, matching ComfyUI's partial-denoise schedule
  slicing. DDIM noises an existing latent in alpha space for partial denoise;
  sigma samplers use `latent + sigma*noise`. Unsupported sampler
  combinations fail explicitly instead of silently changing algorithms.
  Scheduler scalar multiplication stays device-native and every CFG,
  derivative, noise, x0, and superseded sample temporary is released after its
  final dispatch. Step history is metadata-only by default so 20–50 steps do
  not retain full latent tensors; diagnostics can opt into the intentionally
  owning `:retain-step-tensors? true` mode. A live Apple Metal verifier runs all
  four samplers for three steps, checks CPU parity, releases caller-owned
  inputs/output, and requires live GPU buffer count and bytes to return exactly
  to their pre-sampler baseline.
- `VAEEncode` normalizes NHWC RGB `[0,1]` into NCHW `[-1,1]`, executes the
  checkpoint encoder, selects the diagonal-Gaussian posterior mean, and applies
  the model scaling factor. Its output connects directly to partial-denoise
  `KSampler`; `VAEDecode` performs the inverse latent-to-image path. Encoder
  downsampling reproduces Diffusers' asymmetric right/bottom zero padding
  before its stride-2 convolution. On an `ITensorBackend`, padding and posterior
  channel selection dispatch device-native kernels with no intermediate readback.
  The node-boundary NHWC↔NCHW layout and `[0,1]`↔`[-1,1]` range conversions are
  likewise fused into one Metal kernel instead of downloading every pixel;
  `vae-encoder-metal-verify` checks both directions and the encoder graph against
  the CPU oracle on Apple Metal.

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

Call `comfyui.nodes.diffusion-runtime/release-components!` with loaded
MODEL/CLIP/VAE outputs when unloading a workflow. It deduplicates shared caches,
destroys cached GPU tensors, clears the caches, and closes each shared
safetensors mapping once. VAE encoder/decoder graphs also release replaced
intermediate tensors between layers while preserving caller inputs and outputs.
UNet graphs validate unique `:save` names and require every skip use to follow
its definition. The compiler counts `:add-saved`/`:cat-saved` consumers, keeps a
skip alive across every consumer, and destroys it immediately after its last use.
Remaining saved values and timestep embeddings are released when the graph
returns; caller inputs, cached weights, conditioning, and output remain owned.
Adjacent graph-level `GroupNorm → SiLU` layers are compiled into num's fused
Metal kernel. Diffusers ResBlocks use the same primitive internally and release
their normalization, convolution, and projected-skip temporaries as soon as each
dependent dispatch has been submitted.
SpatialTransformer blocks also remain device-native on Metal: last-axis
LayerNorm, GEGLU slicing/GELU/gating, rank-3/4 axis permutations, broadcast
batched matmul, and fused self/cross multi-head attention no longer materialize
host vectors. The Metal verifier executes a complete self-attention →
cross-attention → GEGLU block and compares its output with the CPU oracle.
Each residual stage now destroys its normalized input, projected Q/K/V,
attention output, and feed-forward activation immediately after final use,
instead of retaining a whole Transformer block's activation set.
VAE spatial attention and the legacy `:cross-attention` lowering use the same
fused device kernel, avoiding their former score-matrix host softmax path and
releasing Q/K/V and projection temporaries after their final dispatch.

`comfyui.diffusion.architecture` inspects tensor names/shapes without decoding
payloads and identifies SD1 (768 context), SD2 (1024), SDXL base (label
conditioning + 2048), SDXL refiner (label conditioning + 1280), and 9-channel
inpainting variants. It records the exact input/output/cross-attention tensors
used as evidence and returns `:unknown` for ambiguous layouts. Recognized
families automatically receive the exact 1000-step CompVis `scaled_linear`
beta/alpha schedule unless the host supplies one explicitly.

`comfyui.clip.tokenizer` implements OpenAI CLIP's byte-to-Unicode mapping,
Unicode-aware splitting, ranked byte-pair merges, special tokens, and 77-token
padding/truncation from standard `encoder.json` and `merges.txt` files. The
executable `CLIPTextEncode` node turns prompt text plus a loaded CLIP component
into token IDs and an attention mask inside the checkpoint-to-PNG graph.
When a CLIP graph spec is present, it additionally executes checkpoint-backed
token/position embeddings, pre-LayerNorm causal multi-head self-attention,
QuickGELU MLP/residual blocks, and final LayerNorm, returning both
`[1,77,hidden]` conditioning and pooled embeddings. Transformer tensors are
loaded once and reused across repeated prompt encodes. Token gather plus
position addition, LayerNorm, QuickGELU, causal multi-head attention, fused-QKV
slicing, and EOS pooling all stay device-native on Metal. Each block releases
its normalized/projection/MLP activations at final use; the live verifier runs a
multi-layer encoder against the CPU oracle and returns to its pre-encode GPU
buffer baseline after outputs and cached weights are released.
For recognized SD1/SD2 checkpoints, the loader derives this CLIP graph
automatically from the standard `CLIPTextModel` tensor catalog: contiguous
encoder layer count, hidden width/head count, both norms, Q/K/V/out projections,
MLP weights, and final norm. Auto-mapping is all-or-nothing; a missing tensor
returns no executable encoder instead of silently constructing a partial model.
For SDXL checkpoints containing two complete HF `CLIPTextModel` catalogs, both
encoders are inferred and executed automatically. Their per-token features are
concatenated on the hidden dimension (for example 768 + 1280 = 2048), while the
second encoder supplies pooled conditioning. Per-encoder token tensors and the
unused CLIP-L pool are released after concatenation by default; callers that
intentionally own both complete diagnostic results can set
`:retain-encoder-outputs? true` to receive `:clip-l` and `:clip-g`.
Native SDXL OpenCLIP catalogs are also recognized: `transformer.resblocks.*`,
fused `attn.in_proj_weight/bias`, `ln_final`, and `text_projection` are mapped
automatically. Fused projections are split into Q/K/V, CLIP-G uses its
penultimate transformer state, and pooled output is projected before entering
the dual-conditioning result.
The executable `CLIPTextEncodeSDXL` node carries original width/height, crop
offset, and target width/height as the standard six time IDs alongside token
conditioning and the projected pooled embedding. Invalid negative dimensions
or a non-dual encoder result fail before sampling.
When all standard `diffusion_model.time_embed.*` and
`diffusion_model.label_emb.0.*` tensors exist, the checkpoint loader now
prepends executable timestep and SDXL label-vector construction automatically.
The graph concatenates projected pooled CLIP-G with sinusoidal embeddings of
the six time IDs, runs the learned label MLP, adds it to the timestep vector,
and exposes that shared state to ResBlock `:add-embedding` projections.
`infer-unet-layout` now enumerates every indexed input, middle, and output
module from a CompVis checkpoint, classifying input convolution, ResBlock,
SpatialTransformer, downsample, and upsample modules. It rejects missing block
indices, unknown modules, incomplete middle blocks, or missing final norm/conv
tensors. Complete ResBlock catalogs lower to an executable compound operation
covering both norms/convolutions, learned timestep/label projection, optional
1×1 skip projection, and residual addition.
Complete SpatialTransformer catalogs also lower automatically, including
GroupNorm and proj-in/out, arbitrary contiguous transformer depth, self- and
cross-attention, three LayerNorms, GEGLU feed-forward layers, and all residuals.
When every module is complete, the loader assembles the whole UNet graph:
encoder outputs are saved, decoder blocks concatenate them in reverse order,
middle/down/up modules execute in checkpoint order, and final
GroupNorm/SiLU/convolution produces epsilon. Any unlowerable module prevents an
automatic denoiser instead of installing a partial graph.

The same graph compiler now builds checkpoint-backed VAE decoders whose output
may change spatial/channel dimensions. `VAEDecode` performs latent scaling and
decoder conv/normalization/upsampling, converts NCHW RGB from `[-1,1]` to
clamped NHWC `[0,1]` in one device-native kernel, and `SaveImage` writes each
batch item through the JVM PNG codec. The end-to-end runtime fixture executes
`CheckpointLoaderSimple → KSampler → VAEDecode → SaveImage`, produces a real
32×32 PNG, and verifies its signature and output metadata.

Diffusers `AutoencoderKL` files are inferred directly from their `config.json`.
Both current attention tensor names (`to_q/to_k/to_v/to_out.0`) and legacy
Diffusers 0.7 names (`query/key/value/proj_attn`) are recognized without
renaming checkpoint files. Legacy scheduler configs that predate serialized
`prediction_type` correctly default to their original epsilon prediction.
The decoder executes post-quant projection, mid-block ResNets and spatial self-
attention, every up block, final normalization, and RGB convolution. The encoder
executes every down block/downsample, its own mid-block attention, quant projection,
posterior-mean selection, and latent scaling without renaming tensors. Decoder and
encoder retain independent lazy tensor caches. The real-checkpoint verifier decodes
and then re-encodes the published tiny Stable Diffusion VAE, proving both paths load
their actual tensor catalogs:

```sh
clojure -M:real-diffusers-vae-verify vae.safetensors vae/config.json
```

`DiffusersPipelineLoader` loads the upstream directory layout without first
repacking it into a monolithic checkpoint: standalone UNet, Transformers
`CLIPTextModel`, VAE, and scheduler configs are validated and compiled as one
ComfyUI-compatible MODEL/CLIP/VAE triple. Configured small CLIP head counts and
Hugging Face EOS padding are preserved. `KSampler` now creates reproducible
seeded initial noise (DDIM alpha-space or Euler sigma-space) instead of treating
`EmptyLatentImage`'s zeros as the starting noisy latent. The full real-file
verifier executes prompt tokenization, positive/negative CLIP conditioning,
CFG UNet sampling, VAE decoding, and PNG output:

```sh
clojure -M:real-diffusers-pipeline-verify \
  unet/model.safetensors unet/config.json \
  text_encoder/model.safetensors text_encoder/config.json \
  vae/model.safetensors vae/config.json scheduler/scheduler_config.json \
  tokenizer/vocab.json tokenizer/merges.txt
```

The real verifiers also pin numerical reference values produced from the same
inputs by PyTorch 2.13, Diffusers 0.39, and Transformers 5.13. CLIP's first
hidden-state values agree within `1e-5`; VAE pixels agree within `1e-4` with a
total image-sum error below `1e-2`; UNet epsilon values agree within `1e-4`.
UNet timestep embeddings preserve Diffusers' `flip_sin_to_cos` and `freq_shift`
configuration. Multi-step sampling likewise preserves `linspace`, `leading`,
or `trailing` timestep spacing, `steps_offset`, and `set_alpha_to_one` rather
than silently substituting a generic schedule.

The pipeline verifier additionally runs a fixed-noise, two-step `[501, 1]`
DDIM trajectory through positive/negative CFG and the VAE. Against the same
PyTorch/Diffusers trajectory, guided epsilon sums agree within `1e-4`, final
latent and image sums within `1e-3`, and sampled output pixels within `1e-4`.
It was rerun against the public `Narsil/tiny-stable-diffusion-torch` split
safetensors pipeline after these compatibility changes: all 304 UNet, 84 CLIP,
and 70 VAE tensors loaded, a real 852-byte PNG was emitted, CLIP max error was
`6.07e-7`, two-step epsilon max error `1.03e-5`, latent-sum error `2.68e-6`,
and image max/sum errors `1.19e-5` / `7.80e-5`.
Full-denoise DDIM begins from the seeded noise tensor (`init_noise_sigma=1`),
while Euler retains sigma-scaled initialization.

The same public split checkpoint now runs directly through Deno WebGPU/Metal;
it is no longer restricted to the JVM CPU loader. The JVM exporter inspects
tensor metadata/configs and emits only the inferred graph, schedule, and token
IDs. Deno independently validates each safetensors byte window, decodes
floating-point and integer dtypes, lazily uploads the requested weights, and
executes CLIP → two CFG UNet/DDIM steps → VAE without a Python runtime:

```sh
clojure -M:export-diffusers-metal-spec pipeline.edn \
  unet/model.safetensors unet/config.json \
  text_encoder/model.safetensors text_encoder/config.json \
  vae/model.safetensors vae/config.json scheduler/scheduler_config.json \
  tokenizer/vocab.json tokenizer/merges.txt
clojure -M:real-diffusers-metal-verify
deno run --allow-all target/real-diffusers-metal-verify.cjs \
  pipeline.edn unet/model.safetensors text_encoder/model.safetensors \
  vae/model.safetensors output.png
```

On Apple M4, `Narsil/tiny-stable-diffusion-torch` completes this verifier with
the same pinned PyTorch tolerances above, peaks at 7,744,788 tracked GPU-buffer
bytes, and returns to exactly zero live buffers/bytes after outputs and all
three lazy weight caches are released. With a num backend exposing raw byte-view
upload, 7,590,224 checkpoint bytes are transferred from their validated F32
safetensors windows without first allocating per-element JavaScript numbers;
older backends retain the decoded-vector fallback. NCHW channel embedding broadcast is
lowered through device-native transpose plus last-axis bias dispatch, so this
real UNet path performs no synchronous GPU readback.
The Deno reader keeps an open file handle rather than retaining the whole
checkpoint in a `Uint8Array`: it reads the bounded JSON header once and seeks to
one validated tensor window per lazy cache miss. Across the 7,590,224 bytes used
by the public tiny pipeline, the largest host window is 294,912 bytes. The
verifiers close each file explicitly, reject reads after close, and still return
all tracked GPU storage to baseline.
F16 and BF16 safetensors windows use the same route when supported: encoded
16-bit words are uploaded unchanged, expanded by Metal into the current f32
execution graph, and the temporary packed buffer is immediately retired.
`safetensors-deno-metal-verify` builds a valid odd-length file containing both
dtypes and proves their device expansion plus exact return to the GPU-memory
baseline; an older num backend exercises the numerically identical host fallback.

For whole-model precision regression, `convert-safetensors-f16` rewrites one
checkpoint at a time, quantizing floating tensors while preserving integer
tensors, names, shapes, metadata, and lazy bounded I/O:

```sh
clojure -M:convert-safetensors-f16 model.safetensors model-f16.safetensors
```

The public tiny pipeline was converted across all 304 UNet tensors, 85 CLIP
entries, and 124 VAE entries. Its executable graph then loaded 304/84/70 F16
tensors through device expansion (458 direct uploads, zero decoded uploads),
matched an independent JVM CPU execution of the same quantized files within the
existing Metal tolerances, and returned to zero live GPU bytes. Requested
checkpoint traffic fell from 7,590,224 F32 bytes to 3,795,112 F16 bytes and the
largest host window from 294,912 to 147,456 bytes.

The Metal verifier now ends at an actual image artifact rather than a pixel
vector. `comfyui.png-deno` quantizes the final NHWC RGB values to RGB8, emits
filter-0 scanlines, compresses them with the host `CompressionStream`, and
writes CRC-populated IHDR/IDAT/IEND chunks without Python, JVM ImageIO, or a PNG
dependency. The F32 public-checkpoint run emits a valid non-interlaced 16×16 RGB
PNG of 852 bytes; its fully F16 checkpoint run emits 848 bytes. The verifier
parses IHDR dimensions itself, requires a non-empty compressed artifact, and an
external `file` probe recognizes both outputs as 8-bit/color RGB PNG files.

`comfyui.nodes.diffusion-runtime-deno/pack` exposes this path through real
ComfyUI node types rather than verifier-only function calls. Its executable
`DiffusersPipelineLoader`, `CLIPTextEncode`, `EmptyLatentImage`, `KSampler`,
`VAEDecode`, and `SaveImage` nodes are registered in the ordinary node registry
and run through `comfyui.exec/execute-async`. Tokenization and seeded noise stay
explicit host capabilities; unsupported sampler combinations fail rather than
silently changing algorithms. The live graph gate runs this API-format graph:

```text
DiffusersPipelineLoader ─┬─ CLIPTextEncode(positive) ─┐
                         ├─ CLIPTextEncode(negative) ─┼─ KSampler
EmptyLatentImage ─────────────────────────────────────┘     │
DiffusersPipelineLoader.VAE ────────────────────────── VAEDecode → SaveImage
```

```sh
clojure -M:real-diffusers-graph-metal-verify
deno run --allow-all target/real-diffusers-graph-metal-verify.cjs \
  pipeline.edn unet/model.safetensors text_encoder/model.safetensors \
  vae/model.safetensors output-directory
```

Both F32 and fully converted F16 public checkpoints execute all seven nodes,
match their independent numerical oracles, emit 852/848-byte PNGs, close the
three lazy files, release conditioning/latent/image and all cached weights, and
finish at zero tracked GPU buffers. The measured F32 peak is 7,738,644 bytes.
The Deno `KSampler` shares the production scheduler implementations with the
JVM runtime: `ddim`, `euler`, `euler_ancestral`, and `dpmpp_2m`, using `normal`
or Karras rho-7 sigma schedules where valid, plus partial-denoise timestep
slicing. Initial and ancestral noise remain an injected seeded host function.
The real F32 seven-node graph was run for all seven valid sampler/scheduler
combinations; every run produced a finite 16×16 PNG and returned to zero live
GPU buffers, with PNG sizes from 847 to 859 bytes. DDIM additionally retains its
pinned PyTorch trajectory comparison; the scheduler's dedicated live Metal gate
provides CPU parity for Euler/ancestral/DPM++ transitions.

This path now has an end-to-end wall-clock measurement, not a queue-submission
timing. The interval starts after WebGPU device negotiation and ends only after
lazy tensor reads, pipeline compilation, seven-node execution, final GPU
readback, zlib compression, and PNG write have completed. Five fresh Deno
processes on Apple M4 measured F32 DDIM at 456.876–502.047 ms (mean 481.715 ms,
median 480.645 ms). The independent JVM CPU pipeline took 55.765 s for the same
fixed two-step graph, roughly 116× the Metal interval. F16 checkpoint expansion
measured 557.774–601.661 ms (mean 581.062 ms): it halves checkpoint traffic but
is slower for this tiny model because 458 separate half-expansion dispatches
dominate. Single fresh-process measurements for the other six F32 combinations
were 488.312–541.193 ms. These are tiny 16×16 output measurements with warm OS
file cache, not evidence for 512×512 production throughput; full-size profiling
and kernel fusion remain required.

This is not yet a verified production SD/SDXL render: the automatic graph
mapping still needs full-size validation and pixel/numerical comparison against
upstream Diffusers, and additional ancestral/DPM-SDE/3M sampler families,
exponential/polyexponential schedules, additional VAE variants, mixed precision, and an installed real
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
