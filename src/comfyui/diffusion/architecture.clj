(ns comfyui.diffusion.architecture
  "Read-only diffusion checkpoint architecture inference from safetensors
  names and shapes. No payload is decoded during inspection."
  (:require [clojure.string :as str]
            [comfyui.diffusion.scheduler :as scheduler]
            [comfyui.safetensors :as safe]))

(defn- find-name [names suffix]
  (first (filter #(str/ends-with? % suffix) names)))

(defn- shape-at [checkpoint tensor-name]
  (some-> (safe/tensor-info checkpoint tensor-name) (get "shape") vec))

(defn infer
  "Infer Stable Diffusion family/config from a validated SafeTensorFile.
  Returns evidence with `:family :unknown` rather than guessing when required
  structural tensors are absent or contradictory."
  [checkpoint]
  (let [names (safe/tensor-names checkpoint)
        input-name (find-name names "diffusion_model.input_blocks.0.0.weight")
        output-name (find-name names "diffusion_model.out.2.weight")
        cross-name (first (filter #(re-find #"transformer_blocks\.0\.attn2\.to_k\.weight$" %)
                                  names))
        input-shape (shape-at checkpoint input-name)
        output-shape (shape-at checkpoint output-name)
        cross-shape (shape-at checkpoint cross-name)
        in-channels (first (rest input-shape))
        model-channels (first input-shape)
        out-channels (first output-shape)
        context-dim (second cross-shape)
        label-conditioning? (boolean
                             (some #(str/includes? % "diffusion_model.label_emb.") names))
        clip? (boolean (some #(or (str/starts-with? % "cond_stage_model.")
                                  (str/starts-with? % "text_encoder.")) names))
        vae? (boolean (some #(or (str/starts-with? % "first_stage_model.")
                                 (str/starts-with? % "vae.")) names))
        base-family
        (cond
          (not (and (= 4 (count input-shape)) (= 4 (count output-shape))
                    (number? context-dim))) :unknown
          (and label-conditioning? (= context-dim 2048)) :stable-diffusion-xl-base
          (and label-conditioning? (= context-dim 1280)) :stable-diffusion-xl-refiner
          (= context-dim 768) :stable-diffusion-v1
          (= context-dim 1024) :stable-diffusion-v2
          :else :unknown)
        inpainting? (= 9 in-channels)]
    {:family base-family
     :variant (when inpainting? :inpainting)
     :in-channels in-channels
     :model-channels model-channels
     :out-channels out-channels
     :context-dim context-dim
     :label-conditioning? label-conditioning?
     :clip? clip?
     :vae? vae?
     :evidence {:input-conv input-name :input-shape input-shape
                :output-conv output-name :output-shape output-shape
                :cross-attention cross-name :cross-attention-shape cross-shape}}))

(defn default-alphas-cumprod
  "CompVis/Stable-Diffusion 1000-step scaled-linear schedule. Returns nil for
  an unknown family so callers do not silently apply the wrong schedule."
  [{:keys [family]}]
  (when-not (= :unknown family)
    (scheduler/alphas-cumprod
     (scheduler/scaled-linear-betas 1000 0.00085 0.012))))
