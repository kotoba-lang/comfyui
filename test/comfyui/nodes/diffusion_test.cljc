(ns comfyui.nodes.diffusion-test
  "Proves comfyui-clj VALIDATES the exact ComfyUI-API-format workflows that
  murakumo's fleet operator (murakumo.infer.media) builds and dispatches —
  image, video, audio — using the diffusion node contracts. The graph algebra
  (links, types, topo order) is shared kotoba data; only the compute is a host
  pack. This is the kotoba-native bridge: no Python, no murakumo→comfyui dep."
  (:require [clojure.test :refer [deftest is testing]]
            [comfyui.node :as node]
            [comfyui.nodes.diffusion :as diff]
            [comfyui.workflow :as wf]))

(def reg (node/registry diff/pack))

;; byte-for-byte the shape murakumo.infer.media/txt2img-workflow emits
(def sdxl-workflow
  {"1" {:class_type "CheckpointLoaderSimple" :inputs {:ckpt_name "animagine-xl-4.0.safetensors"}}
   "2" {:class_type "CLIPTextEncode" :inputs {:clip ["1" 1] :text "a cloud spirit"}}
   "3" {:class_type "CLIPTextEncode" :inputs {:clip ["1" 1] :text "lowres"}}
   "4" {:class_type "EmptyLatentImage" :inputs {:width 832 :height 1216 :batch_size 1}}
   "5" {:class_type "KSampler" :inputs {:model ["1" 0] :positive ["2" 0] :negative ["3" 0]
                                        :latent_image ["4" 0] :seed 20260703 :steps 24
                                        :cfg 6.0 :sampler_name "euler_ancestral"
                                        :scheduler "normal" :denoise 1.0}}
   "6" {:class_type "VAEDecode" :inputs {:samples ["5" 0] :vae ["1" 2]}}
   "7" {:class_type "SaveImage" :inputs {:images ["6" 0] :filename_prefix "murakumo"}}})

(def i2v-workflow
  {"1" {:class_type "ImageOnlyCheckpointLoader" :inputs {:ckpt_name "svd_xt.safetensors"}}
   "2" {:class_type "LoadImage" :inputs {:image "seed.png"}}
   "4" {:class_type "SVD_img2vid_Conditioning"
        :inputs {:clip_vision ["1" 1] :init_image ["2" 0] :vae ["1" 2]
                 :width 640 :height 640 :video_frames 48 :motion_bucket_id 127
                 :fps 16 :augmentation_level 0.0}}
   "5" {:class_type "KSampler" :inputs {:model ["1" 0] :positive ["4" 0] :negative ["4" 1]
                                        :latent_image ["4" 2] :seed 1 :steps 20 :cfg 3.0
                                        :sampler_name "euler" :scheduler "karras" :denoise 1.0}}
   "6" {:class_type "VAEDecode" :inputs {:samples ["5" 0] :vae ["1" 2]}}
   "7" {:class_type "SaveAnimatedWEBP" :inputs {:images ["6" 0] :filename_prefix "murakumo-vid"
                                                :fps 16 :lossless false :quality 85 :method "default"}}})

(def audio-workflow
  {"1" {:class_type "CheckpointLoaderSimple" :inputs {:ckpt_name "stable-audio-open-1.0.safetensors"}}
   "2" {:class_type "CLIPTextEncode" :inputs {:clip ["1" 1] :text "rain on a tin roof"}}
   "3" {:class_type "CLIPTextEncode" :inputs {:clip ["1" 1] :text ""}}
   "4" {:class_type "EmptyLatentAudio" :inputs {:seconds 10}}
   "5" {:class_type "KSampler" :inputs {:model ["1" 0] :positive ["2" 0] :negative ["3" 0]
                                        :latent_image ["4" 0] :seed 1 :steps 50 :cfg 5.0
                                        :sampler_name "dpmpp_3m_sde_gpu" :scheduler "exponential"
                                        :denoise 1.0}}
   "6" {:class_type "VAEDecodeAudio" :inputs {:samples ["5" 0] :vae ["1" 2]}}
   "7" {:class_type "SaveAudio" :inputs {:audio ["6" 0] :filename_prefix "murakumo-aud"}}})

(deftest validates-image
  (let [{:keys [valid? errors]} (wf/validate reg sdxl-workflow)]
    (is valid? (str "SDXL errors: " (pr-str errors)))))

(deftest validates-video
  (let [{:keys [valid? errors]} (wf/validate reg i2v-workflow)]
    (is valid? (str "i2v errors: " (pr-str errors)))))

(deftest validates-audio
  (let [{:keys [valid? errors]} (wf/validate reg audio-workflow)]
    (is valid? (str "audio errors: " (pr-str errors)))))

(deftest topo-order-is-executable
  (testing "the checkpoint loads before sampling, sampling before decode/save"
    (let [order (wf/topo-sort sdxl-workflow)
          pos (into {} (map-indexed (fn [i id] [id i]) order))]
      (is (< (pos "1") (pos "5")))
      (is (< (pos "5") (pos "6")))
      (is (< (pos "6") (pos "7"))))))

(deftest catches-a-broken-link
  (testing "a dangling link is a validation error, not a silent bad render"
    (let [broken (assoc-in sdxl-workflow ["5" :inputs :positive] ["99" 0])
          {:keys [valid? errors]} (wf/validate reg broken)]
      (is (not valid?))
      (is (some #(= :dangling-link (:error %)) errors)))))
