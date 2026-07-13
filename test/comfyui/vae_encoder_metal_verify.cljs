(ns comfyui.vae-encoder-metal-verify
  "Prove the ComfyUI VAE graph's asymmetric pad, posterior channel slice, and
  latent scale stay device-native on Deno WebGPU backed by Apple Metal."
  (:require [comfyui.diffusion.model :as model]
            [num.array :as arr]
            [num.cpu :as cpu]
            [num.deno-gpu :as dg]))

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

(defn -main [& _]
  (let [cpu-backend (cpu/cpu-backend)
        expected-output (compile-and-run cpu-backend)
        expected (arr/->vec expected-output)
        expected-denoise (arr/->vec (denoise cpu-backend))]
    (-> (dg/request-device)
        (.then
         (fn [request]
           (let [gpu (dg/backend request)
                 output (compile-and-run gpu)]
             (.then
              (js/Promise.all
               #js [(arr/->vec output) (arr/->vec (denoise gpu))])
              (fn [results]
                (let [actual (aget results 0)
                      actual-denoise (aget results 1)]
                  (when-not
                      (and (= [1 2 3 3] (:shape output))
                           (= (count expected) (count actual))
                           (every? true?
                                   (map #(< (Math/abs (- %1 %2)) 1.0e-6)
                                        expected actual))
                           (every? true?
                                   (map #(< (Math/abs (- %1 %2)) 1.0e-6)
                                        expected-denoise actual-denoise)))
                    (throw
                     (ex-info "Metal liveness graphs differ from CPU"
                              {:shape (:shape output)
                               :expected expected :actual actual
                               :expected-denoise expected-denoise
                               :actual-denoise actual-denoise})))
                  (println "OK VAE + saved-skip liveness graphs match CPU with"
                           "GPUBuffer.destroy on"
                           (dg/adapter-description request))))))))
        (.catch
         (fn [error]
           (println "ERROR:" (or (.-stack error) (str error)))
           (js/Deno.exit 1))))))

(set! *main-cli-fn* -main)
