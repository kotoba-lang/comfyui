(ns comfyui.real-diffusers-graph-metal-verify
  "Execute a real API-format ComfyUI graph through Deno Metal to PNG."
  (:require [cljs.reader :as reader]
            [comfyui.exec :as exec]
            [comfyui.node :as node]
            [comfyui.nodes.diffusion-runtime-deno :as runtime]
            [comfyui.png-deno :as png]
            [comfyui.safetensors-deno :as safe]
            [num.array :as arr]
            [num.deno-gpu :as dg]))

(defn- close? [expected actual tolerance]
  (< (Math/abs (- expected actual)) tolerance))

(defn- f16-file? [path]
  (let [checkpoint (safe/open-file path)
        floating #{"F16" "BF16" "F32" "F64"}
        dtypes (map #(get % "dtype") (vals (:tensors checkpoint)))
        result (and (some floating dtypes)
                    (every? #(or (not (floating %)) (= "F16" %)) dtypes))]
    (safe/close-file! checkpoint)
    result))

(defn -main
  [& [spec-path unet-path text-path vae-path output-directory
      sampler-name scheduler-name]]
  (when-not output-directory
    (throw (ex-info "usage: SPEC UNET TEXT VAE OUTPUT_DIRECTORY" {})))
  (let [{:keys [unet clip vae alphas positive negative]}
        (reader/read-string (js/Deno.readTextFileSync spec-path))
        f16? (every? f16-file? [unet-path text-path vae-path])
        direct? (boolean (resolve 'num.deno-gpu/upload-byte-view))
        sampler-name (or sampler-name "ddim")
        scheduler-name (or scheduler-name "normal")]
    (-> (dg/request-device)
        (.then
         (fn [request]
           (let [backend (dg/backend request)
                 baseline (dg/backend-stats backend)
                 tokenize (fn [text]
                            (case text
                              "a tiny red robot" positive
                              "blurry" negative
                              (throw (ex-info "unexpected verifier prompt" {:text text}))))
                 noise-fn (fn [shape _seed backend]
                            (arr/from-vec backend
                                          (mapv #(* 0.5 (Math/sin %))
                                                (range (arr/nelems shape))) shape))
                 pack (runtime/pack
                       {:backend backend :tokenize tokenize :noise-fn noise-fn
                        :output-directory output-directory
                        :timesteps-fn (fn [_ _] [501 1])
                        :pipeline {:unet unet-path :text-encoder text-path
                                   :vae vae-path :unet-spec unet
                                   :clip-spec clip :vae-spec vae :alphas alphas}})
                 registry (node/registry pack)
                 events (atom [])
                 workflow
                 {"1" {:class_type "DiffusersPipelineLoader"
                       :inputs {:pipeline_name "tiny-stable-diffusion"}}
                  "2" {:class_type "CLIPTextEncode"
                       :inputs {:clip ["1" 1] :text "a tiny red robot"}}
                  "3" {:class_type "CLIPTextEncode"
                       :inputs {:clip ["1" 1] :text "blurry"}}
                  "4" {:class_type "EmptyLatentImage"
                       :inputs {:width 64 :height 64 :batch_size 1}}
                  "5" {:class_type "KSampler"
                       :inputs {:model ["1" 0] :positive ["2" 0]
                                :negative ["3" 0] :latent_image ["4" 0]
                                :seed 0 :steps 2 :cfg 2.0
                                :sampler_name sampler-name :scheduler scheduler-name
                                :denoise 1.0}}
                  "6" {:class_type "VAEDecode"
                       :inputs {:samples ["5" 0] :vae ["1" 2]}}
                  "7" {:class_type "SaveImage"
                       :inputs {:images ["6" 0] :filename_prefix "metal_graph"}}}]
             (-> (exec/execute-async
                  {:registry registry :on-event #(swap! events conj %)} workflow)
                 (.then
                  (fn [execution]
                    (let [components (get-in execution [:results "1"])
                          positive-result (get-in execution [:results "2" 0])
                          negative-result (get-in execution [:results "3" 0])
                          sample-result (get-in execution [:results "5" 0])
                          latent (:samples sample-result)
                          image (get-in execution [:results "6" 0])
                          ui (get-in execution [:results "7" 0])
                          reader-stats (mapv #(safe/reader-stats
                                              (:comfyui/checkpoint %)) components)
                          path (get-in ui [:images 0 :path])
                          png-bytes (js/Deno.readFileSync path)
                          reads (js/Promise.all
                                 #js [(arr/->vec latent) (arr/->vec image)])]
                      (arr/release-all!
                       [(:tensor positive-result) (:pooled positive-result)
                        (:tensor negative-result) (:pooled negative-result)
                        latent image])
                      (runtime/release-components! components)
                      (.then
                       reads
                       (fn [values]
                         (let [latent-values (aget values 0)
                               image-values (aget values 1)
                               stats (dg/backend-stats backend)]
                           (when-not
                               (and (= ["1" "4" "2" "3" "5" "6" "7"]
                                       (:executed execution))
                                    (= 7 (count @events))
                                    (= [16 16] (png/dimensions png-bytes))
                                    (> (.-byteLength png-bytes) 500)
                                    (or (not direct?)
                                        (every? pos? (map :direct-uploads reader-stats)))
                                    (= 2 (count (:history sample-result)))
                                    (every? #(js/Number.isFinite %) latent-values)
                                    (every? #(js/Number.isFinite %) image-values)
                                    (or (not= "ddim" sampler-name)
                                        (and (close? (if f16? 15.374797308671601
                                                         15.379568099975586)
                                                     (reduce + latent-values) 1.0e-3)
                                             (close? (if f16? 391.73747777734826
                                                         391.7310791015625)
                                                     (reduce + image-values) 1.0e-2)))
                                    (= (:live-buffers baseline) (:live-buffers stats))
                                    (= (:live-bytes baseline) (:live-bytes stats)))
                             (throw (ex-info "real ComfyUI Metal graph verification failed"
                                             {:executed (:executed execution)
                                              :events @events
                                              :latent-sum (reduce + latent-values)
                                              :image-sum (reduce + image-values)
                                              :reader-stats reader-stats
                                              :png path :stats stats})))
                           (println "OK API-format 7-node ComfyUI graph → Metal → PNG on"
                                    (dg/adapter-description request)
                                    "png-bytes" (.-byteLength png-bytes)
                                    "peak-bytes" (:peak-live-bytes stats)
                                    "checkpoint-dtype" (if f16? "F16" "F32")
                                    "sampler" sampler-name "scheduler" scheduler-name
                                    "output" path)))))))))))
        (.catch (fn [error]
                  (println "ERROR:" (or (.-stack error) (str error)))
                  (when-let [data (ex-data error)]
                    (println "DATA:" (pr-str data)))
                  (js/Deno.exit 1))))))

(set! *main-cli-fn* -main)
