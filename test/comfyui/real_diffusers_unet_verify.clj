(ns comfyui.real-diffusers-unet-verify
  "Execute a real small Diffusers Stable Diffusion UNet safetensors graph."
  (:require [clojure.data.json :as json]
            [comfyui.diffusion.architecture :as architecture]
            [comfyui.diffusion.model :as model]
            [comfyui.safetensors :as safe]
            [num.array :as arr]
            [num.cpu :as cpu])
  (:import [java.io FileReader]))

(defn -main [& [checkpoint-path config-path]]
  (when-not (and checkpoint-path config-path)
    (throw (ex-info "usage: real-diffusers-unet-verify unet.safetensors config.json" {})))
  (with-open [checkpoint (safe/open-file checkpoint-path)
              reader (FileReader. config-path)]
    (let [config (json/read reader)
          spec (architecture/infer-diffusers-unet-spec checkpoint config)
          backend (cpu/cpu-backend)
          reads (atom 0)
          component {:comfyui/read-tensor
                     (fn [_backend name]
                       (swap! reads inc)
                       (safe/read-tensor checkpoint backend name))}
          denoise (model/compile-denoiser component backend spec)
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
                  :loaded-tensors @reads :shape (:shape output)
                  :finite (every? #(Double/isFinite (double %)) values)
                  :seconds (/ (- (System/nanoTime) started) 1.0e9)}]
      (println (pr-str report))
      (when-not (and spec (= [1 4 8 8] (:shape output)) (:finite report)
                     (pos? @reads))
        (throw (ex-info "real Diffusers UNet verification failed" report))))))
