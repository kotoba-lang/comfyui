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
