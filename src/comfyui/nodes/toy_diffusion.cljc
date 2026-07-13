(ns comfyui.nodes.toy-diffusion
  "ADR-2607131500 Phase 1 — ONE node whose `:fn` is not host-bound-absent
  (unlike every node in `comfyui.nodes.diffusion`) but REAL: conv2d + relu +
  self-attention, dispatched through `kotoba-lang/num`'s verified CPU-backend
  tensor ops (num.tensor, ADR-2607131500). This is the honest minimal answer
  to 'reimplement ComfyUI's diffusion compute in Clojure' — proof the real
  `comfyui.exec` topological executor can run REAL tensor math for a node,
  not just validate shapes and proxy to Python (which is still what every
  node in `comfyui.nodes.diffusion` + `murakumo.infer.media` actually does
  for production image generation, and continues to do — this node does not
  replace that).

  NOT a real diffusion model: no trained weights, no safetensors loading, no
  real UNet architecture, no noise schedule, no sampler loop. 'ToyDiffusion
  Step' is ONE forward pass through a conv2d→relu→self-attention→reshape
  chain — the actual architectural SIGNATURE of a diffusion UNet block
  (spatial conv + attention), on a tiny single-channel 'image', with
  caller-supplied (not trained) weights. See ADR-2607131500 for the full,
  explicit scope fence.

  Optional dependency: unlike every other node pack in this repo, requiring
  this namespace pulls in `kotoba-lang/num` — NOT part of comfyui-clj's base
  deps (this repo's own deps.edn keeps `langchain` as the only third-party
  dep by design). A host registers this pack only if it brings `num` itself,
  exactly like `comfyui.gateway`'s JVM branch requires a host to bring
  http-kit/jsonista."
  (:require [num.core :as nm]
            [num.tensor :as t]))

(defn- t' [type-name category inputs outputs]
  {:type type-name :category category :inputs inputs :outputs outputs})

(def toy-diffusion-step
  "A conv2d + relu + self-attention step. Inputs (all `num.array/NDArray`,
  not ComfyUI's Python tensor wire format):
    :image  — [H W] single-channel latent
    :kernel — [kh kw] conv kernel, caller-supplied (no weight loading here)
  Output: `[H-kh+1 W-kw+1]` — same spatial-shrink convention as
  `num.tensor/conv2d` (no padding, stride 1; see its own docstring)."
  (t' "ToyDiffusionStep" "diffusion-toy"
      {:image {:type "LATENT"} :kernel {:type "*"}}
      [{:name "LATENT" :type "LATENT"}]))

(def pack
  "Register alongside comfyui.nodes.diffusion's contracts-only pack when a
  host wants at least one node that actually computes."
  [(assoc toy-diffusion-step
          :fn (fn [{:keys [image kernel]}]
                (let [conv (nm/relu (t/conv2d image kernel))
                      [oh ow] (:shape conv)
                      flat (t/reshape conv [(* oh ow) 1])
                      attended (t/attention flat flat flat)]
                  [(t/reshape attended [oh ow])])))])
