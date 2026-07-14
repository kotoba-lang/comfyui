(ns comfyui.standard-vae-metal-verify
  "Exercise production tiled VAE decode with a real four-block standard SD VAE."
  (:require [cljs.reader :as reader]
            [comfyui.diffusion.model :as model]
            [comfyui.diffusion.tiled-vae-deno :as tiled-vae]
            [comfyui.png-deno :as png]
            [comfyui.safetensors-deno :as safe]
            [num.array :as arr]
            [num.deno-gpu :as dg]))

(def latent-size 64)
(def output-size 512)

(defn -main [& [spec-path checkpoint-path output-path]]
  (when-not checkpoint-path
    (throw (ex-info "usage: SPEC CHECKPOINT [OUTPUT]" {})))
  (let [spec (reader/read-string (js/Deno.readTextFileSync spec-path))
        checkpoint (safe/open-file checkpoint-path)
        output-path (or output-path "/tmp/comfyui-standard-vae-metal.png")]
    (-> (dg/request-device)
        (.then
         (fn [request]
           (let [backend (dg/backend request)
                 baseline (dg/backend-stats backend)
                 decode (model/compile-decoder (safe/component checkpoint) backend spec)
                 cache (-> decode meta :comfyui/tensor-cache)
                 latent (arr/from-vec
                         backend
                         (mapv #(* 0.25 (Math/sin (* 0.01 %)))
                               (range (* 4 latent-size latent-size)))
                         [1 4 latent-size latent-size])
                 started (.now js/performance)]
             (-> (tiled-vae/decode-tiled decode latent {:tile-size 8 :overlap 2})
                 (.then
                  (fn [image]
                    (when-not (= [1 output-size output-size 3] (:shape image))
                      (throw (ex-info "standard VAE image shape mismatch"
                                      {:shape (:shape image)})))
                    (-> (arr/->vec image)
                        (.then (fn [values]
                                 {:image image :values values
                                  :png (png/encode-rgb values output-size output-size)})))))
                 (.then
                  (fn [{:keys [image values png]}]
                    (-> png
                        (.then
                         (fn [bytes]
                           (when-not (every? #(js/Number.isFinite %) values)
                             (throw (ex-info "standard VAE produced non-finite pixels" {})))
                           (js/Deno.writeFileSync output-path bytes)
                           (let [weights-loaded (count @cache)
                                 reader-stats (safe/reader-stats checkpoint)]
                             (arr/release-all! [latent image])
                             (arr/release-all! (vals @cache))
                             (reset! cache {})
                             (safe/close-file! checkpoint)
                             (let [stats (dg/backend-stats backend)
                                   elapsed (- (.now js/performance) started)]
                               (when-not (and (= [output-size output-size]
                                                  (png/dimensions bytes))
                                              (> (.-byteLength bytes) 10000)
                                              (= (:live-buffers baseline)
                                                 (:live-buffers stats))
                                              (= (:live-bytes baseline)
                                                 (:live-bytes stats)))
                                 (throw (ex-info "standard VAE Metal lifecycle failed"
                                                 {:bytes (.-byteLength bytes)
                                                  :stats stats})))
                               (println "OK production VAEDecodeTiled standard SD VAE → 512x512 Metal PNG on"
                                        (dg/adapter-description request)
                                        "tiles" 121
                                        "png-bytes" (.-byteLength bytes)
                                        "weights" weights-loaded
                                        "checkpoint-bytes" (:window-bytes reader-stats)
                                        "peak-bytes" (:peak-live-bytes stats)
                                        "elapsed-ms" (.toFixed elapsed 3)
                                        "output" output-path))))))))))))
        (.catch (fn [error]
                  (safe/close-file! checkpoint)
                  (println "ERROR:" (or (.-stack error) (str error)))
                  (when-let [data (ex-data error)] (println "DATA:" (pr-str data)))
                  (js/Deno.exit 1))))))

(set! *main-cli-fn* -main)
