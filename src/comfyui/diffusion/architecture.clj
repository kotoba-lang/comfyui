(ns comfyui.diffusion.architecture
  "Read-only diffusion checkpoint architecture inference from safetensors
  names and shapes. No payload is decoded during inspection."
  (:require [clojure.string :as str]
            [comfyui.diffusion.scheduler :as scheduler]
            [comfyui.safetensors :as safe])
  (:import [java.util.regex Pattern]))

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
                                  (str/starts-with? % "text_encoder.")
                                  (str/starts-with? % "text_encoder_2.")
                                  (str/starts-with? % "conditioner.embedders.")) names))
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

(defn- infer-hf-clip-spec [checkpoint names root]
    (let [token-name (str root "embeddings.token_embedding.weight")
          position-name (str root "embeddings.position_embedding.weight")
          layer-pattern (when root
                          (re-pattern
                           (str "^" (Pattern/quote root)
                                "encoder\\.layers\\.(\\d+)\\.layer_norm1\\.weight$")))
          layer-indices (when layer-pattern
                          (->> names
                               (keep #(some->> (re-matches layer-pattern %)
                                               second Long/parseLong))
                               sort vec))
          contiguous? (= layer-indices (vec (range (count layer-indices))))
          hidden (second (shape-at checkpoint token-name))
          heads (when (and hidden (zero? (mod (long hidden) 64)))
                  (quot (long hidden) 64))
          layer-spec
          (fn [index]
            (let [base (str root "encoder.layers." index ".")]
              {:norm1-weight (str base "layer_norm1.weight")
               :norm1-bias (str base "layer_norm1.bias")
               :query-weight (str base "self_attn.q_proj.weight")
               :query-bias (str base "self_attn.q_proj.bias")
               :key-weight (str base "self_attn.k_proj.weight")
               :key-bias (str base "self_attn.k_proj.bias")
               :value-weight (str base "self_attn.v_proj.weight")
               :value-bias (str base "self_attn.v_proj.bias")
               :output-weight (str base "self_attn.out_proj.weight")
               :output-bias (str base "self_attn.out_proj.bias")
               :norm2-weight (str base "layer_norm2.weight")
               :norm2-bias (str base "layer_norm2.bias")
               :fc1-weight (str base "mlp.fc1.weight")
               :fc1-bias (str base "mlp.fc1.bias")
               :fc2-weight (str base "mlp.fc2.weight")
               :fc2-bias (str base "mlp.fc2.bias")}))
          layers (mapv layer-spec layer-indices)
          final-weight (str root "final_layer_norm.weight")
          final-bias (str root "final_layer_norm.bias")
          required (concat [token-name position-name final-weight final-bias]
                           (mapcat vals layers))]
      (when (and root contiguous? (seq layers) (pos-int? heads)
                 (every? names required))
        {:token-embedding token-name :position-embedding position-name
         :layers layers :heads heads
         :final-norm-weight final-weight :final-norm-bias final-bias
         :eps 1.0e-5})))

(defn infer-clip-spec
  "Infer executable HF CLIP graphs. SD1/SD2 return one encoder; SDXL returns
  a dual encoder spec when two complete text encoders are present."
  [checkpoint {:keys [family]}]
  (let [names (set (safe/tensor-names checkpoint))
        token-suffix "embeddings.token_embedding.weight"
        roots (->> names
                   (filter #(str/ends-with? % token-suffix))
                   (map #(subs % 0 (- (count %) (count token-suffix))))
                   sort vec)
        specs (vec (keep #(infer-hf-clip-spec checkpoint names %) roots))]
    (cond
      (contains? #{:stable-diffusion-v1 :stable-diffusion-v2} family)
      (when (= 1 (count specs)) (first specs))

      (contains? #{:stable-diffusion-xl-base :stable-diffusion-xl-refiner} family)
      (when (>= (count specs) 2)
        {:mode :dual :encoders (vec (take 2 specs))})

      :else nil)))
