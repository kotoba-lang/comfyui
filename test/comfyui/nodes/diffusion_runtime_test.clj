(ns comfyui.nodes.diffusion-runtime-test
  (:require [clojure.data.json :as json]
            [clojure.test :refer [deftest is]]
            [comfyui.exec :as exec]
            [comfyui.node :as node]
            [comfyui.nodes.diffusion-runtime :as runtime]
            [comfyui.safetensors :as safe]
            [num.array :as arr]
            [num.cpu :as cpu])
  (:import [java.nio ByteBuffer ByteOrder]
           [java.nio.file Files OpenOption]
           [java.nio.file.attribute FileAttribute]))

(def backend (cpu/cpu-backend))

(defn- checkpoint-fixture []
  (let [header (.getBytes
                (json/write-str
                 {"model.diffusion_model.input.weight"
                  {"dtype" "F32" "shape" [1] "data_offsets" [0 4]}
                  "cond_stage_model.token.weight"
                  {"dtype" "F32" "shape" [1] "data_offsets" [4 8]}
                  "first_stage_model.decoder.weight"
                  {"dtype" "F32" "shape" [1] "data_offsets" [8 12]}})
                "UTF-8")
        buffer (doto (ByteBuffer/allocate (+ 8 (alength header) 12))
                 (.order ByteOrder/LITTLE_ENDIAN))
        path (Files/createTempFile "comfyui-runtime-" ".safetensors"
                                   (make-array FileAttribute 0))]
    (.putLong buffer (long (alength header)))
    (.put buffer header)
    (doseq [x [1.0 2.0 3.0]] (.putFloat buffer (float x)))
    (Files/write path (.array buffer) (make-array OpenOption 0))
    path))

(deftest executable-pack-loads-checkpoint-and-allocates-real-latent
  (let [path (checkpoint-fixture)
        registry (node/registry (runtime/pack {:backend backend
                                               :resolve-checkpoint (constantly path)}))
        workflow {"load" {:class_type "CheckpointLoaderSimple"
                           :inputs {:ckpt_name "fixture.safetensors"}}
                  "latent" {:class_type "EmptyLatentImage"
                             :inputs {:width 16 :height 24 :batch_size 2}}}]
    (try
      (let [result (exec/execute {:registry registry} workflow)
            [model clip vae] (get-in result [:results "load"])
            latent (get-in result [:results "latent" 0])
            checkpoint (:comfyui/checkpoint model)]
        (is (= ["latent" "load"] (sort (:executed result))))
        (is (= [2 4 3 2] (:shape latent)))
        (is (= [:model :clip :vae]
               (mapv :comfyui/component [model clip vae])))
        (is (= [["model.diffusion_model.input.weight"]
                ["cond_stage_model.token.weight"]
                ["first_stage_model.decoder.weight"]]
               (mapv :comfyui/tensor-names [model clip vae])))
        (is (= [1.0]
               (arr/->vec ((:comfyui/read-tensor model)
                           backend "model.diffusion_model.input.weight"))))
        (safe/close! checkpoint))
      (finally
        (Files/deleteIfExists path)))))

(deftest ddim-node-executes-inside-the-comfyui-graph
  (let [registry (node/registry (runtime/pack {:backend backend}))
        sample (arr/from-vec backend [3.0] [1 1 1 1])
        epsilon (arr/from-vec backend [1.0] [1 1 1 1])
        workflow {"step" {:class_type "DDIMStep"
                           :inputs {:sample sample :epsilon epsilon
                                    :alpha 0.25 :alpha_prev 0.5 :eta 0.0}}}
        result (exec/execute {:registry registry} workflow)
        [previous x0] (get-in result [:results "step"])]
    (is (= ["step"] (:executed result)))
    (is (< (Math/abs (- (/ (- 3.0 (Math/sqrt 0.75)) 0.5)
                           (first (arr/->vec x0))))
           1.0e-10))
    (is (= [1 1 1 1] (:shape previous)))))

(deftest ksampler-node-runs-iterative-cfg-denoising
  (let [registry (node/registry (runtime/pack {:backend backend}))
        sample (arr/from-vec backend [2.0] [1 1 1 1])
        negative (arr/from-vec backend [0.1] [1 1 1 1])
        positive (arr/from-vec backend [0.3] [1 1 1 1])
        model {:comfyui/alphas-cumprod [0.9 0.7 0.5]
               :comfyui/denoise (fn [_sample _timestep conditioning] conditioning)}
        workflow {"sample" {:class_type "KSampler"
                             :inputs {:model model :positive positive :negative negative
                                      :latent_image sample :seed 7 :steps 2 :cfg 2.0
                                      :sampler_name "ddim" :scheduler "normal"
                                      :denoise 1.0}}}
        output (get-in (exec/execute {:registry registry} workflow)
                       [:results "sample" 0])]
    (is (= [1 1 1 1] (:shape output)))
    (is (not= (arr/->vec sample) (arr/->vec output)))))
