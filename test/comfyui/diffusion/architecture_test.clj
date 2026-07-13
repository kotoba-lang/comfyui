(ns comfyui.diffusion.architecture-test
  (:require [clojure.test :refer [deftest is testing]]
            [comfyui.diffusion.architecture :as architecture]
            [comfyui.diffusion.scheduler :as scheduler]))

(defn- checkpoint [context-dim in-channels label?]
  {:tensors
   (cond->
    {"model.diffusion_model.input_blocks.0.0.weight"
     {"shape" [320 in-channels 3 3]}
     "model.diffusion_model.out.2.weight" {"shape" [4 320 3 3]}
     "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"
     {"shape" [320 context-dim]}
     "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
     {"shape" [49408 context-dim]}
     "first_stage_model.decoder.conv_out.weight" {"shape" [3 128 3 3]}}
     label? (assoc "model.diffusion_model.label_emb.0.0.weight"
                   {"shape" [1280 2816]}))})

(deftest detects-stable-diffusion-families-from-structural-evidence
  (testing "SD1/SD2 context widths and inpainting channels"
    (is (= {:family :stable-diffusion-v1 :variant nil
            :in-channels 4 :context-dim 768}
           (select-keys (architecture/infer (checkpoint 768 4 false))
                        [:family :variant :in-channels :context-dim])))
    (is (= {:family :stable-diffusion-v2 :variant :inpainting
            :in-channels 9 :context-dim 1024}
           (select-keys (architecture/infer (checkpoint 1024 9 false))
                        [:family :variant :in-channels :context-dim]))))
  (testing "SDXL base/refiner use label conditioning plus distinct context widths"
    (is (= :stable-diffusion-xl-base
           (:family (architecture/infer (checkpoint 2048 4 true)))))
    (is (= :stable-diffusion-xl-refiner
           (:family (architecture/infer (checkpoint 1280 4 true))))))
  (testing "unsupported context width remains unknown"
    (is (= :unknown (:family (architecture/infer (checkpoint 1536 4 false)))))))

(deftest stable-diffusion-scaled-linear-schedule-is-exact
  (let [betas (scheduler/scaled-linear-betas 3 0.00085 0.012)
        middle (Math/pow (/ (+ (Math/sqrt 0.00085) (Math/sqrt 0.012)) 2.0) 2)]
    (is (= 0.00085 (first betas)))
    (is (< (Math/abs (- middle (second betas))) 1.0e-15))
    (is (< (Math/abs (- 0.012 (last betas))) 1.0e-15))
    (is (= 1000 (count (architecture/default-alphas-cumprod
                        {:family :stable-diffusion-v1}))))
    (is (nil? (architecture/default-alphas-cumprod {:family :unknown})))))

(deftest infers-complete-standard-sd1-clip-transformer-spec
  (let [root "cond_stage_model.transformer.text_model."
        base (checkpoint 768 4 false)
        layer-base (str root "encoder.layers.0.")
        names [(str root "embeddings.position_embedding.weight")
               (str root "final_layer_norm.weight") (str root "final_layer_norm.bias")
               (str layer-base "layer_norm1.weight") (str layer-base "layer_norm1.bias")
               (str layer-base "self_attn.q_proj.weight") (str layer-base "self_attn.q_proj.bias")
               (str layer-base "self_attn.k_proj.weight") (str layer-base "self_attn.k_proj.bias")
               (str layer-base "self_attn.v_proj.weight") (str layer-base "self_attn.v_proj.bias")
               (str layer-base "self_attn.out_proj.weight") (str layer-base "self_attn.out_proj.bias")
               (str layer-base "layer_norm2.weight") (str layer-base "layer_norm2.bias")
               (str layer-base "mlp.fc1.weight") (str layer-base "mlp.fc1.bias")
               (str layer-base "mlp.fc2.weight") (str layer-base "mlp.fc2.bias")]
        complete (update base :tensors merge
                         (into {} (map (fn [name] [name {"shape" [768]}]) names)))
        info (architecture/infer complete)
        spec (architecture/infer-clip-spec complete info)]
    (is (= 12 (:heads spec)))
    (is (= 1 (count (:layers spec))))
    (is (= (str layer-base "self_attn.q_proj.weight")
           (:query-weight (first (:layers spec)))))
    (is (nil? (architecture/infer-clip-spec
               (update complete :tensors dissoc (str layer-base "mlp.fc2.bias")) info)))))

(defn- hf-clip-tensors [root hidden]
  (let [layer (str root "encoder.layers.0.")
        suffixes ["layer_norm1.weight" "layer_norm1.bias"
                  "self_attn.q_proj.weight" "self_attn.q_proj.bias"
                  "self_attn.k_proj.weight" "self_attn.k_proj.bias"
                  "self_attn.v_proj.weight" "self_attn.v_proj.bias"
                  "self_attn.out_proj.weight" "self_attn.out_proj.bias"
                  "layer_norm2.weight" "layer_norm2.bias"
                  "mlp.fc1.weight" "mlp.fc1.bias"
                  "mlp.fc2.weight" "mlp.fc2.bias"]]
    (merge {(str root "embeddings.token_embedding.weight") {"shape" [100 hidden]}
            (str root "embeddings.position_embedding.weight") {"shape" [77 hidden]}
            (str root "final_layer_norm.weight") {"shape" [hidden]}
            (str root "final_layer_norm.bias") {"shape" [hidden]}}
           (into {} (map (fn [suffix] [(str layer suffix) {"shape" [hidden]}]) suffixes)))))

(deftest infers-sdxl-dual-hf-clip-spec
  (let [checkpoint* (update (checkpoint 2048 4 true) :tensors merge
                            (hf-clip-tensors "conditioner.embedders.0.transformer.text_model."
                                             768)
                            (hf-clip-tensors "text_encoder_2.text_model." 1280))
        spec (architecture/infer-clip-spec checkpoint* (architecture/infer checkpoint*))]
    (is (= :dual (:mode spec)))
    (is (= [12 20] (mapv :heads (:encoders spec))))
    (is (= 2 (count (:encoders spec))))))
