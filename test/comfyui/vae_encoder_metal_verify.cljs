(ns comfyui.vae-encoder-metal-verify
  "Prove the ComfyUI VAE graph's asymmetric pad, posterior channel slice, and
  latent scale stay device-native on Deno WebGPU backed by Apple Metal."
  (:require [comfyui.diffusion.model :as model]
            [num.array :as arr]
            [num.cpu :as cpu]
            [num.deno-gpu :as dg]))

(def values (mapv #(- (* 0.125 %) 0.75) (range 16)))
(def spec {:layers []
           :encoder-layers [{:op :pad-right-bottom}
                            {:op :take-channels :channels 2}
                            {:op :scale :factor 0.5}]})
(def component
  {:comfyui/read-tensor
   (fn [_ name] (throw (ex-info "unexpected tensor read" {:name name})))})

(defn- compile-and-run [backend]
  ((model/compile-encoder component backend spec)
   (arr/from-vec backend values [1 4 2 2])))

(defn -main [& _]
  (let [expected-output (compile-and-run (cpu/cpu-backend))
        expected (arr/->vec expected-output)]
    (-> (dg/request-device)
        (.then
         (fn [request]
           (let [gpu (dg/backend request)
                 output (compile-and-run gpu)]
             (-> (arr/->vec output)
                 (.then
                  (fn [actual]
                    (when-not (and (= [1 2 3 3] (:shape output))
                                   (= (count expected) (count actual))
                                   (every? true?
                                           (map #(< (Math/abs (- %1 %2)) 1.0e-6)
                                                expected actual)))
                      (throw (ex-info "Metal VAE encoder graph differs from CPU"
                                      {:shape (:shape output)
                                       :expected expected :actual actual})))
                    (println "OK VAE encoder graph: asymmetric pad + channel slice + scale"
                             "match CPU on" (dg/adapter-description request))))))))
        (.catch
         (fn [error]
           (println "ERROR:" (or (.-stack error) (str error)))
           (js/Deno.exit 1))))))

(set! *main-cli-fn* -main)
