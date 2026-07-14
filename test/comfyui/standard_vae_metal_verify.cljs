(ns comfyui.standard-vae-metal-verify
  "Decode a real standard four-block SD VAE on Deno WebGPU/Metal in bounded
  tiles, matching the production strategy required by WebGPU binding limits."
  (:require [cljs.reader :as reader]
            [comfyui.diffusion.model :as model]
            [comfyui.png-deno :as png]
            [comfyui.safetensors-deno :as safe]
            [num.array :as arr]
            [num.deno-gpu :as dg]
            [num.tensor :as t]))

(def latent-size 64)
(def tile-size 8)
(def tile-stride 6)
(def scale-factor 8)
(def output-size (* latent-size scale-factor))
(def tile-pixels (* tile-size scale-factor))
(def feather-pixels (* (- tile-size tile-stride) scale-factor))
(def tile-origins
  (conj (vec (range 0 (- latent-size tile-size) tile-stride))
        (- latent-size tile-size)))

(defn- latent-tile-values [origin-y origin-x]
  (vec
   (for [channel (range 4)
         y (range tile-size)
         x (range tile-size)]
     (let [global-y (+ origin-y y)
           global-x (+ origin-x x)
           index (+ (* channel latent-size latent-size)
                    (* global-y latent-size) global-x)]
       (* 0.25 (Math/sin (* 0.01 index)))))))

(defn- edge-weight [coordinate origin]
  (let [leading (if (zero? origin) 1.0
                    (min 1.0 (/ (inc coordinate) feather-pixels)))
        trailing (if (= (+ origin tile-size) latent-size) 1.0
                     (min 1.0 (/ (- tile-pixels coordinate) feather-pixels)))]
    (* leading trailing)))

(defn- blend-tile! [output weights tile-values origin-y origin-x]
  (dotimes [y tile-pixels]
    (dotimes [x tile-pixels]
      (let [weight (* (edge-weight y origin-y) (edge-weight x origin-x))]
        (dotimes [channel 3]
          (let [source (+ (* y tile-pixels 3) (* x 3) channel)
                destination (+ (* (+ (* origin-y scale-factor) y) output-size 3)
                               (* (+ (* origin-x scale-factor) x) 3) channel)]
            (aset output destination
                  (+ (aget output destination) (* weight (nth tile-values source))))
            (aset weights destination (+ (aget weights destination) weight))))))))

(defn- decode-tile! [backend decode output weights origin-y origin-x]
  (let [latent (arr/from-vec backend (latent-tile-values origin-y origin-x)
                             [1 4 tile-size tile-size])
        decoded (decode latent)
        image (t/nchw-to-rgb-image decoded)]
    (when-not (and (= [1 3 64 64] (:shape decoded))
                   (= [1 64 64 3] (:shape image)))
      (throw (ex-info "standard VAE tile shape mismatch"
                      {:decoded (:shape decoded) :image (:shape image)})))
    (-> (arr/->vec image)
        (.then (fn [values]
                 (when-not (every? #(js/Number.isFinite %) values)
                   (throw (ex-info "standard VAE tile contains non-finite values"
                                   {:origin [origin-y origin-x]})))
                 (blend-tile! output weights values origin-y origin-x)
                 (arr/release-all! [latent decoded image]))))))

(defn- normalize-output! [output weights]
  (dotimes [index (.-length output)]
    (let [weight (aget weights index)]
      (when-not (pos? weight)
        (throw (ex-info "tiled VAE left an uncovered output pixel" {:index index})))
      (aset output index (/ (aget output index) weight)))))

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
                 component (safe/component checkpoint)
                 decode (model/compile-decoder component backend spec)
                 cache (-> decode meta :comfyui/tensor-cache)
                 output (js/Float32Array. (* output-size output-size 3))
                 weights (js/Float32Array. (* output-size output-size 3))
                 tiles (for [y tile-origins x tile-origins]
                         [y x])
                 started (.now js/performance)
                 decoded-all
                 (reduce (fn [promise [y x]]
                           (.then promise
                                  (fn [_] (decode-tile! backend decode output weights y x))))
                         (js/Promise.resolve nil) tiles)]
             (-> decoded-all
                 (.then (fn [_]
                          (normalize-output! output weights)
                          (png/encode-rgb
                           (vec (js/Array.from output)) output-size output-size)))
                 (.then
                  (fn [bytes]
                    (js/Deno.writeFileSync output-path bytes)
                    (let [weights-loaded (count @cache)
                          reader-stats (safe/reader-stats checkpoint)]
                      (arr/release-all! (vals @cache))
                      (reset! cache {})
                      (safe/close-file! checkpoint)
                      (let [stats (dg/backend-stats backend)
                            elapsed (- (.now js/performance) started)]
                        (when-not (and (= [output-size output-size]
                                           (png/dimensions bytes))
                                       (> (.-byteLength bytes) 10000)
                                       (= (:live-buffers baseline) (:live-buffers stats))
                                       (= (:live-bytes baseline) (:live-bytes stats)))
                          (throw (ex-info "standard VAE Metal lifecycle failed"
                                          {:bytes (.-byteLength bytes) :stats stats})))
                        (println "OK standard SD VAE tiled 64x64 latent → 512x512 Metal PNG on"
                                 (dg/adapter-description request)
                                 "tiles" (count tiles)
                                 "png-bytes" (.-byteLength bytes)
                                 "weights" weights-loaded
                                 "checkpoint-bytes" (:window-bytes reader-stats)
                                 "peak-bytes" (:peak-live-bytes stats)
                                 "elapsed-ms" (.toFixed elapsed 3)
                                 "output" output-path)))))))))
        (.catch (fn [error]
                  (safe/close-file! checkpoint)
                  (println "ERROR:" (or (.-stack error) (str error)))
                  (when-let [data (ex-data error)] (println "DATA:" (pr-str data)))
                  (js/Deno.exit 1))))))

(set! *main-cli-fn* -main)
