(ns comfyui.real-diffusers-unet-verify
  "Execute a real small Diffusers Stable Diffusion UNet safetensors graph."
  (:require [clojure.data.json :as json]
            [comfyui.nodes.diffusion-runtime :as runtime]
            [comfyui.safetensors :as safe]
            [num.array :as arr]
            [num.cpu :as cpu])
  (:import [java.io FileReader]))

(defn -main [& [checkpoint-path config-path]]
  (when-not (and checkpoint-path config-path)
    (throw (ex-info "usage: real-diffusers-unet-verify unet.safetensors config.json" {})))
  (let [config (with-open [reader (FileReader. config-path)]
                 (json/read reader))
        backend (cpu/cpu-backend)
          loader (first (filter #(= "CheckpointLoaderSimple" (:type %))
                                (runtime/pack {:backend backend
                                               :resolve-checkpoint (constantly checkpoint-path)
                                               :diffusers-config config})))
          [model _clip _vae] ((:fn loader) {:ckpt_name checkpoint-path})
          spec (:comfyui/model-spec model)
        denoise (:comfyui/denoise model)]
    (try
      (let [checkpoint (:comfyui/checkpoint model)
          sample (arr/from-vec backend
                               (map #(- (* 0.001 %) 0.1) (range (* 4 8 8)))
                               [1 4 8 8])
          conditioning {:tensor (arr/from-vec backend (repeat (* 4 32) 0.01)
                                              [1 4 32])}
          started (System/nanoTime)
          output (denoise sample 10 conditioning)
          values (arr/->vec output)
          report {:architecture (:architecture spec) :layers (count (:layers spec))
                  :checkpoint-tensors (count (safe/tensor-names checkpoint))
                  :loaded-tensors (count @(-> denoise meta :comfyui/tensor-cache))
                  :shape (:shape output)
                  :finite (every? #(Double/isFinite (double %)) values)
                  :seconds (/ (- (System/nanoTime) started) 1.0e9)}]
      (println (pr-str report))
      (when-not (and spec (= [1 4 8 8] (:shape output)) (:finite report)
                     (pos? (:loaded-tensors report)))
          (throw (ex-info "real Diffusers UNet verification failed" report))))
      (finally
        (safe/close! (:comfyui/checkpoint model))))))
