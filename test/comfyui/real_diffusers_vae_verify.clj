(ns comfyui.real-diffusers-vae-verify
  "Decode a latent through a real Diffusers AutoencoderKL safetensors file."
  (:require [clojure.data.json :as json]
            [clojure.string :as str]
            [comfyui.nodes.diffusion-runtime :as runtime]
            [comfyui.safetensors :as safe]
            [num.array :as arr]
            [num.cpu :as cpu])
  (:import [java.io FileReader]))

(defn -main [& [checkpoint-path config-path]]
  (when-not (and checkpoint-path config-path)
    (throw (ex-info "usage: real-diffusers-vae-verify vae.safetensors config.json" {})))
  (let [config (with-open [reader (FileReader. config-path)]
                 (json/read reader))
        backend (cpu/cpu-backend)
        definitions (runtime/pack
                     {:backend backend
                      :resolve-checkpoint (constantly checkpoint-path)
                      :diffusers-vae-config config})
        loader (first (filter #(= "CheckpointLoaderSimple" (:type %)) definitions))
        vae-decode (first (filter #(= "VAEDecode" (:type %)) definitions))
        [_model _clip vae] ((:fn loader) {:ckpt_name checkpoint-path})]
    (try
      (let [latent (arr/from-vec
                    backend (map #(- (* 0.002 %) 0.25) (range (* 4 8 8)))
                    [1 4 8 8])
            started (System/nanoTime)
            [image] ((:fn vae-decode) {:samples latent :vae vae})
            values (arr/->vec image)
            cache @(-> (:comfyui/decode vae) meta :comfyui/tensor-cache)
            report {:architecture (get-in vae [:comfyui/model-spec :architecture])
                    :checkpoint-tensors (count (safe/tensor-names
                                                (:comfyui/checkpoint vae)))
                    :loaded-tensors (count cache)
                    :encoder-tensors-loaded
                    (count (filter #(str/starts-with? % "encoder.")
                                   (keys cache)))
                    :shape (:shape image)
                    :finite (every? #(Double/isFinite (double %)) values)
                    :range [(apply min values) (apply max values)]
                    :seconds (/ (- (System/nanoTime) started) 1.0e9)}]
        (println (pr-str report))
        (when-not (and (= :diffusers-autoencoder-kl (:architecture report))
                       (= [1 16 16 3] (:shape report)) (:finite report)
                       (every? #(<= 0.0 % 1.0) values)
                       (pos? (:loaded-tensors report))
                       (zero? (:encoder-tensors-loaded report)))
          (throw (ex-info "real Diffusers VAE verification failed" report))))
      (finally
        (safe/close! (:comfyui/checkpoint vae))))))
