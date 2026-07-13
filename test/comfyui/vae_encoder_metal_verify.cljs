(ns comfyui.vae-encoder-metal-verify
  "Prove the ComfyUI VAE graph's asymmetric pad, posterior channel slice, and
  latent scale stay device-native on Deno WebGPU backed by Apple Metal."
  (:require [comfyui.diffusion.model :as model]
            [num.array :as arr]
            [num.cpu :as cpu]
            [num.deno-gpu :as dg]
            [num.tensor :as t]))

(def values (mapv #(- (* 0.125 %) 0.75) (range 16)))
(def spec {:layers []
           :encoder-layers [{:op :groupnorm :groups 2}
                            {:op :silu}
                            {:op :pad-right-bottom}
                            {:op :take-channels :channels 2}
                            {:op :scale :factor 0.5}]})
(def component
  {:comfyui/read-tensor
   (fn [_ name] (throw (ex-info "unexpected tensor read" {:name name})))})

(defn- compile-and-run [backend]
  ((model/compile-encoder component backend spec)
   (arr/from-vec backend values [1 4 2 2])))

(def denoiser-spec
  {:layers [{:op :scale :factor 2.0}
            {:op :save :name :skip}
            {:op :scale :factor 3.0}
            {:op :add-saved :name :skip}
            {:op :scale :factor 0.25}]})

(defn- denoise [backend]
  ((model/compile-denoiser component backend denoiser-spec)
   (arr/from-vec backend [1.0 -2.0 3.0 -4.0] [1 4]) 0 nil))

(def image-values
  [0.0 0.25 0.5, 0.75 1.0 0.1,
   0.2 0.4 0.6, 0.8 0.9 1.0])

(defn- image-conversions [backend]
  (let [image (arr/from-vec backend image-values [2 1 2 3])
        nchw (t/rgb-image-to-nchw image)]
    [nchw (t/nchw-to-rgb-image nchw)]))

(defn -main [& _]
  (let [cpu-backend (cpu/cpu-backend)
        expected-output (compile-and-run cpu-backend)
        expected (arr/->vec expected-output)
        expected-denoise (arr/->vec (denoise cpu-backend))
        [expected-nchw expected-image] (image-conversions cpu-backend)
        expected-nchw-values (arr/->vec expected-nchw)
        expected-image-values (arr/->vec expected-image)]
    (-> (dg/request-device)
        (.then
         (fn [request]
           (let [gpu (dg/backend request)
                 output (compile-and-run gpu)
                 [gpu-nchw gpu-image] (image-conversions gpu)]
             (.then
              (js/Promise.all
               #js [(arr/->vec output) (arr/->vec (denoise gpu))
                    (arr/->vec gpu-nchw) (arr/->vec gpu-image)])
              (fn [results]
                (let [actual (aget results 0)
                      actual-denoise (aget results 1)
                      actual-nchw (aget results 2)
                      actual-image (aget results 3)
                      close? (fn [expected actual]
                               (and (= (count expected) (count actual))
                                    (every? true?
                                            (map #(< (Math/abs (- %1 %2)) 1.0e-6)
                                                 expected actual))))]
                  (when-not
                      (and (= [1 2 3 3] (:shape output))
                           (= [2 3 1 2] (:shape gpu-nchw))
                           (= [2 1 2 3] (:shape gpu-image))
                           (close? expected actual)
                           (close? expected-denoise actual-denoise)
                           (close? expected-nchw-values actual-nchw)
                           (close? expected-image-values actual-image))
                    (throw
                     (ex-info "Metal liveness graphs differ from CPU"
                              {:shape (:shape output)
                               :expected expected :actual actual
                               :expected-denoise expected-denoise
                               :actual-denoise actual-denoise
                               :expected-nchw expected-nchw-values
                               :actual-nchw actual-nchw
                               :expected-image expected-image-values
                               :actual-image actual-image})))
                  (println "OK VAE + image-boundary + saved-skip graphs match CPU with"
                           "GPUBuffer.destroy on"
                           (dg/adapter-description request))))))))
        (.catch
         (fn [error]
           (println "ERROR:" (or (.-stack error) (str error)))
           (js/Deno.exit 1))))))

(set! *main-cli-fn* -main)
