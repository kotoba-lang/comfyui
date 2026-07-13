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

(defn infer-sdxl-conditioning-layers
  "Infer the shared SDXL timestep and pooled/time-ID label MLP prefix. Returns
  graph layers that populate the embedding state, or nil when incomplete."
  [checkpoint {:keys [family]}]
  (when (contains? #{:stable-diffusion-xl-base :stable-diffusion-xl-refiner} family)
    (let [names (set (safe/tensor-names checkpoint))
          name! (fn [suffix] (find-name names suffix))
          time-first-weight (name! "diffusion_model.time_embed.0.weight")
          time-first-bias (name! "diffusion_model.time_embed.0.bias")
          time-second-weight (name! "diffusion_model.time_embed.2.weight")
          time-second-bias (name! "diffusion_model.time_embed.2.bias")
          label-first-weight (name! "diffusion_model.label_emb.0.0.weight")
          label-first-bias (name! "diffusion_model.label_emb.0.0.bias")
          label-second-weight (name! "diffusion_model.label_emb.0.2.weight")
          label-second-bias (name! "diffusion_model.label_emb.0.2.bias")
          required [time-first-weight time-first-bias time-second-weight time-second-bias
                    label-first-weight label-first-bias label-second-weight label-second-bias]]
      (when (every? some? required)
        [{:op :timestep-vector
          :first-weight time-first-weight :first-bias time-first-bias
          :second-weight time-second-weight :second-bias time-second-bias}
         {:op :sdxl-label-embedding
          :first-weight label-first-weight :first-bias label-first-bias
         :second-weight label-second-weight :second-bias label-second-bias}]))))

(defn- indexed-values [names pattern group]
  (->> names
       (keep #(some-> (re-matches pattern %)
                      (nth group) Long/parseLong))
       distinct sort vec))

(defn- module-kind [names prefix section block module]
  (let [base (if (= section "middle_block")
               (str prefix section "." module ".")
               (str prefix section "." block "." module "."))
        present? #(contains? names (str base %))]
    (cond
      (present? "in_layers.0.weight") :resblock
      (some #(str/starts-with? % (str base "transformer_blocks.0.")) names)
      :spatial-transformer
      (present? "op.weight") :downsample
      (present? "conv.weight") :upsample
      (and (= section "input_blocks") (= block 0) (= module 0)
           (present? "weight")) :input-conv
      :else :unknown)))

(defn- section-layout [names prefix section]
  (let [pattern (re-pattern
                 (str "^" (Pattern/quote (str prefix section "."))
                      "(\\d+)\\.(\\d+)\\..+$"))
        blocks (indexed-values names pattern 1)]
    (mapv
     (fn [block]
       (let [block-pattern (re-pattern
                            (str "^" (Pattern/quote (str prefix section "." block "."))
                                 "(\\d+)\\..+$"))
             modules (indexed-values names block-pattern 1)]
         {:index block
          :modules (mapv (fn [module]
                           {:index module
                            :kind (module-kind names prefix section block module)
                            :prefix (str prefix section "." block "." module ".")})
                         modules)}))
     blocks)))

(defn infer-unet-layout
  "Enumerate a CompVis SD/SDXL UNet topology from tensor names without decoding
  payloads. `:complete?` requires contiguous block indices, known module kinds,
  all three middle modules, and final norm/conv tensors."
  [checkpoint]
  (let [names (set (safe/tensor-names checkpoint))
        input-name (find-name names "diffusion_model.input_blocks.0.0.weight")
        marker "input_blocks.0.0.weight"
        prefix (when input-name (subs input-name 0 (- (count input-name) (count marker))))
        inputs (when prefix (section-layout names prefix "input_blocks"))
        outputs (when prefix (section-layout names prefix "output_blocks"))
        middle (when prefix
                 (mapv (fn [module]
                         {:index module
                          :kind (module-kind names prefix "middle_block" 0 module)
                          :prefix (str prefix "middle_block." module ".")})
                       (range 3)))
        contiguous? (fn [blocks]
                      (= (mapv :index blocks) (vec (range (count blocks)))))
        final-norm (when prefix (str prefix "out.0.weight"))
        final-conv (when prefix (str prefix "out.2.weight"))
        known? #(not= :unknown (:kind %))
        complete? (boolean
                   (and prefix (seq inputs) (seq outputs)
                        (contiguous? inputs) (contiguous? outputs)
                        (every? known? (mapcat :modules inputs))
                        (every? known? (mapcat :modules outputs))
                        (= [:resblock :spatial-transformer :resblock]
                           (mapv :kind middle))
                        (contains? names final-norm)
                        (contains? names (str prefix "out.0.bias"))
                        (contains? names final-conv)
                        (contains? names (str prefix "out.2.bias"))))]
    {:prefix prefix :input-blocks (or inputs []) :middle-block middle
     :output-blocks (or outputs []) :final-norm final-norm
     :final-conv final-conv :complete? complete?}))

(defn infer-resblock-layer
  "Lower one enumerated `:resblock` module to the executable graph vocabulary.
  Returns nil unless every mandatory tensor and each optional skip pair exists."
  [checkpoint {:keys [kind prefix]}]
  (when (= :resblock kind)
    (let [names (set (safe/tensor-names checkpoint))
          tensor #(str prefix %)
          layer {:op :resblock :groups 32
                 :in-norm-weight (tensor "in_layers.0.weight")
                 :in-norm-bias (tensor "in_layers.0.bias")
                 :in-conv-weight (tensor "in_layers.2.weight")
                 :in-conv-bias (tensor "in_layers.2.bias")
                 :embedding-weight (tensor "emb_layers.1.weight")
                 :embedding-bias (tensor "emb_layers.1.bias")
                 :out-norm-weight (tensor "out_layers.0.weight")
                 :out-norm-bias (tensor "out_layers.0.bias")
                 :out-conv-weight (tensor "out_layers.3.weight")
                 :out-conv-bias (tensor "out_layers.3.bias")}
          skip-weight (tensor "skip_connection.weight")
          skip-bias (tensor "skip_connection.bias")
          mandatory (vals (dissoc layer :op :groups))
          skip-present [(contains? names skip-weight) (contains? names skip-bias)]]
      (when (and (every? names mandatory) (or (every? true? skip-present)
                                              (every? false? skip-present)))
        (cond-> layer
          (first skip-present) (assoc :skip-weight skip-weight :skip-bias skip-bias))))))

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
         :layers layers :heads heads :hidden hidden :format :hf
         :final-norm-weight final-weight :final-norm-bias final-bias
         :eps 1.0e-5})))

(defn- infer-openclip-spec [checkpoint names root]
  (let [token-name (str root "token_embedding.weight")
        position-name (str root "positional_embedding")
        layer-pattern (re-pattern
                       (str "^" (Pattern/quote root)
                            "transformer\\.resblocks\\.(\\d+)\\.ln_1\\.weight$"))
        indices (->> names
                     (keep #(some->> (re-matches layer-pattern %)
                                     second Long/parseLong))
                     sort vec)
        contiguous? (= indices (vec (range (count indices))))
        hidden (second (shape-at checkpoint token-name))
        heads (when (and hidden (zero? (mod (long hidden) 64)))
                (quot (long hidden) 64))
        layer-spec
        (fn [index]
          (let [base (str root "transformer.resblocks." index ".")]
            {:norm1-weight (str base "ln_1.weight")
             :norm1-bias (str base "ln_1.bias")
             :in-proj-weight (str base "attn.in_proj_weight")
             :in-proj-bias (str base "attn.in_proj_bias")
             :output-weight (str base "attn.out_proj.weight")
             :output-bias (str base "attn.out_proj.bias")
             :norm2-weight (str base "ln_2.weight")
             :norm2-bias (str base "ln_2.bias")
             :fc1-weight (str base "mlp.c_fc.weight")
             :fc1-bias (str base "mlp.c_fc.bias")
             :fc2-weight (str base "mlp.c_proj.weight")
             :fc2-bias (str base "mlp.c_proj.bias")}))
        layers (mapv layer-spec indices)
        final-weight (str root "ln_final.weight")
        final-bias (str root "ln_final.bias")
        projection (str root "text_projection")
        required (concat [token-name position-name final-weight final-bias projection]
                         (mapcat vals layers))]
    (when (and contiguous? (seq layers) (pos-int? heads) (every? names required))
      {:token-embedding token-name :position-embedding position-name
       :layers layers :heads heads :hidden hidden :format :openclip
       :final-norm-weight final-weight :final-norm-bias final-bias
       :text-projection projection :return-penultimate? true :eps 1.0e-5})))

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
        hf-specs (keep #(infer-hf-clip-spec checkpoint names %) roots)
        open-token-suffix "token_embedding.weight"
        open-roots (->> names
                        (filter #(and (str/ends-with? % open-token-suffix)
                                      (not (str/ends-with? % token-suffix))))
                        (map #(subs % 0 (- (count %) (count open-token-suffix))))
                        sort vec)
        open-specs (keep #(infer-openclip-spec checkpoint names %) open-roots)
        specs (vec (sort-by :hidden (concat hf-specs open-specs)))]
    (cond
      (contains? #{:stable-diffusion-v1 :stable-diffusion-v2} family)
      (when (= 1 (count specs)) (first specs))

      (contains? #{:stable-diffusion-xl-base :stable-diffusion-xl-refiner} family)
      (when (>= (count specs) 2)
        {:mode :dual :encoders (vec (take 2 specs))})

      :else nil)))
