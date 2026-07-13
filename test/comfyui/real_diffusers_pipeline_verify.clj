(ns comfyui.real-diffusers-pipeline-verify
  "Run real split Diffusers UNet, CLIP, scheduler, and VAE files to a PNG."
  (:require [clojure.data.json :as json]
            [comfyui.clip.tokenizer :as tokenizer]
            [comfyui.exec :as exec]
            [comfyui.node :as node]
            [comfyui.nodes.diffusion-runtime :as runtime]
            [comfyui.safetensors :as safe]
            [num.array :as arr]
            [num.cpu :as cpu])
  (:import [java.nio.file Files Path]
           [java.nio.file.attribute FileAttribute]))

(defn- read-json [path]
  (json/read-str (Files/readString (Path/of path (make-array String 0)))))

(defn -main
  [& [unet-path unet-config-path text-path text-config-path vae-path
      vae-config-path scheduler-config-path vocab-path merges-path]]
  (when-not merges-path
    (throw (ex-info
            (str "usage: real-diffusers-pipeline-verify UNET UNET_CONFIG TEXT "
                 "TEXT_CONFIG VAE VAE_CONFIG SCHEDULER_CONFIG VOCAB MERGES") {})))
  (let [backend (cpu/cpu-backend)
        output-dir (Files/createTempDirectory "comfyui-diffusers-output-"
                                              (make-array FileAttribute 0))
        clip-tokenizer (tokenizer/load-tokenizer
                        vocab-path merges-path {:pad-token "<|endoftext|>"})
        pipeline {:unet unet-path :unet-config (read-json unet-config-path)
                  :text-encoder text-path :text-config (read-json text-config-path)
                  :vae vae-path :vae-config (read-json vae-config-path)
                  :scheduler-config (read-json scheduler-config-path)}
        registry (node/registry
                  (runtime/pack
                   {:backend backend :clip-tokenizer clip-tokenizer
                    :output-directory output-dir
                    :resolve-diffusers-pipeline (constantly pipeline)}))
        workflow
        {"load" {:class_type "DiffusersPipelineLoader"
                  :inputs {:pipeline_name "tiny-stable-diffusion"}}
         "positive" {:class_type "CLIPTextEncode"
                     :inputs {:clip ["load" 1] :text "a tiny red robot"}}
         "negative" {:class_type "CLIPTextEncode"
                     :inputs {:clip ["load" 1] :text "blurry"}}
         "latent" {:class_type "EmptyLatentImage"
                   :inputs {:width 64 :height 64 :batch_size 1}}
         "sample" {:class_type "KSampler"
                   :inputs {:model ["load" 0] :positive ["positive" 0]
                            :negative ["negative" 0] :latent_image ["latent" 0]
                            :seed 260713 :steps 1 :cfg 2.0
                            :sampler_name "ddim" :scheduler "normal" :denoise 1.0}}
         "decode" {:class_type "VAEDecode"
                   :inputs {:samples ["sample" 0] :vae ["load" 2]}}
         "save" {:class_type "SaveImage"
                 :inputs {:images ["decode" 0] :filename_prefix "diffusers-real"}}}
        started (System/nanoTime)
        result (exec/execute {:registry registry} workflow)
        model (get-in result [:results "load" 0])
        clip (get-in result [:results "load" 1])
        vae (get-in result [:results "load" 2])]
    (try
      (let [image (get-in result [:results "decode" 0])
            positive (get-in result [:results "positive" 0])
            clip-values (vec (arr/->vec (:tensor positive)))
            reference-clip-first8 [-0.8407002091407776 -0.3963874876499176
                                   -0.6832109093666077 0.021882688626646996
                                   -0.6473066210746765 0.13609130680561066
                                   -1.265396237373352 0.06672965735197067]
            clip-reference-max-error
            (apply max (map #(Math/abs (- %1 %2))
                            (take 8 clip-values) reference-clip-first8))
            saved (get-in result [:results "save" 0 :images 0])
            path (Path/of (:path saved) (make-array String 0))
            report {:image-shape (:shape image)
                    :png-bytes (Files/size path)
                    :model-tensors (count @(-> (:comfyui/denoise model)
                                               meta :comfyui/tensor-cache))
                    :clip-tensors (count @(-> (:comfyui/encode clip)
                                              meta :comfyui/tensor-cache))
                    :clip-input-ids (vec (take 12 (:input-ids positive)))
                    :clip-sum (reduce + clip-values)
                    :clip-first8 (vec (take 8 clip-values))
                    :clip-reference-max-error clip-reference-max-error
                    :vae-tensors (count @(-> (:comfyui/decode vae)
                                             meta :comfyui/tensor-cache))
                    :seconds (/ (- (System/nanoTime) started) 1.0e9)}]
        (println (pr-str report))
        (when-not (and (= [1 16 16 3] (:image-shape report))
                       (> (:png-bytes report) 100)
                       (= [0 197 85 358 211 83 362 385 67 879 1 1]
                          (:clip-input-ids report))
                       (< clip-reference-max-error 1.0e-5)
                       (every? pos? ((juxt :model-tensors :clip-tensors :vae-tensors)
                                     report)))
          (throw (ex-info "real Diffusers pipeline verification failed" report))))
      (finally
        (doseq [checkpoint (map :comfyui/checkpoint [model clip vae])]
          (safe/close! checkpoint))))))
