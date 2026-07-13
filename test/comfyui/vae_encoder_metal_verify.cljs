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

(defn- identity-values [size scale]
  (mapv (fn [index]
          (if (= (quot index size) (mod index size)) scale 0.0))
        (range (* size size))))

(defn- vae-attention [backend]
  (let [matrix (arr/from-vec backend (identity-values 2 0.5) [2 2])
        zeros (arr/zeros backend [2])
        arrays {"norm.w" (arr/from-vec backend [1.0 1.0] [2])
                "norm.b" zeros
                "q.w" matrix "q.b" zeros "k.w" matrix "k.b" zeros
                "v.w" matrix "v.b" zeros "out.w" matrix "out.b" zeros}
        decoder (model/compile-decoder
                 {:comfyui/read-tensor (fn [_ name] (get arrays name))}
                 backend
                 {:layers [{:op :vae-attention :groups 1
                            :norm-weight "norm.w" :norm-bias "norm.b"
                            :query-weight "q.w" :query-bias "q.b"
                            :key-weight "k.w" :key-bias "k.b"
                            :value-weight "v.w" :value-bias "v.b"
                            :output-weight "out.w" :output-bias "out.b"}]})]
    (decoder (arr/from-vec backend [0.2 -0.1 0.7 0.3] [1 2 1 2]))))

(defn- legacy-cross-attention [backend]
  (let [matrix (arr/from-vec backend (identity-values 2 0.5) [2 2])
        zeros (arr/zeros backend [2])
        arrays {"q.w" matrix "q.b" zeros "k.w" matrix "k.b" zeros
                "v.w" matrix "v.b" zeros "out.w" matrix "out.b" zeros}
        denoise (model/compile-denoiser
                 {:comfyui/read-tensor (fn [_ name] (get arrays name))}
                 backend
                 {:layers [{:op :cross-attention :heads 1
                            :query-weight "q.w" :query-bias "q.b"
                            :key-weight "k.w" :key-bias "k.b"
                            :value-weight "v.w" :value-bias "v.b"
                            :output-weight "out.w" :output-bias "out.b"}]})]
    (denoise (arr/from-vec backend [0.2 -0.1 0.7 0.3] [1 2 1 2]) 0
             {:tensor (arr/from-vec backend [0.1 0.4, -0.2 0.5, 0.3 -0.1]
                                            [1 3 2])})))

(defn- spatial-transformer [backend]
  (let [matrix (arr/from-vec backend (identity-values 2 0.2) [2 2])
        zeros (arr/zeros backend [2])
        ones (arr/from-vec backend [1.0 1.0] [2])
        conv (arr/from-vec backend [0.5 0.0 0.0 0.5] [2 2 1 1])
        attention (fn [prefix]
                    {(str prefix ".q") matrix (str prefix ".k") matrix
                     (str prefix ".v") matrix (str prefix ".o") matrix
                     (str prefix ".ob") zeros})
        arrays (merge
                {"gn.w" ones "gn.b" zeros "pin.w" conv "pin.b" zeros
                 "pout.w" conv "pout.b" zeros
                 "n1.w" ones "n1.b" zeros "n2.w" ones "n2.b" zeros
                 "n3.w" ones "n3.b" zeros
                 "ff.in.w" (arr/from-vec backend [0.4 0.1, -0.2 0.3,
                                                    0.5 -0.1, 0.2 0.4] [4 2])
                 "ff.in.b" (arr/zeros backend [4])
                 "ff.out.w" matrix "ff.out.b" zeros}
                (attention "self") (attention "cross"))
        attention-spec (fn [prefix]
                         {:query-weight (str prefix ".q")
                          :key-weight (str prefix ".k")
                          :value-weight (str prefix ".v")
                          :output-weight (str prefix ".o")
                          :output-bias (str prefix ".ob")})
        spec {:layers
              [{:op :spatial-transformer :groups 1
                :norm-weight "gn.w" :norm-bias "gn.b"
                :proj-in-weight "pin.w" :proj-in-bias "pin.b"
                :proj-out-weight "pout.w" :proj-out-bias "pout.b"
                :blocks [{:heads 1
                          :norm1-weight "n1.w" :norm1-bias "n1.b"
                          :self-attention (attention-spec "self")
                          :norm2-weight "n2.w" :norm2-bias "n2.b"
                          :cross-attention (attention-spec "cross")
                          :norm3-weight "n3.w" :norm3-bias "n3.b"
                          :feed-forward {:project-weight "ff.in.w"
                                         :project-bias "ff.in.b"
                                         :output-weight "ff.out.w"
                                         :output-bias "ff.out.b"}}]}]}
        component {:comfyui/read-tensor (fn [_ name] (get arrays name))}
        denoise (model/compile-denoiser component backend spec)
        sample (arr/from-vec backend (mapv #(- (* 0.1 %) 0.3) (range 8)) [1 2 2 2])
        condition {:tensor (arr/from-vec backend [0.2 -0.1, 0.4 0.3, -0.2 0.5]
                                                 [1 3 2])}]
    (denoise sample 0 condition)))

(defn- broadcast-batched-matmul [backend]
  (t/matmul
   (arr/from-vec backend (mapv #(* 0.1 (inc %)) (range 12)) [2 1 2 3])
   (arr/from-vec backend (mapv #(- (* 0.05 %) 0.2) (range 12)) [1 2 3 2])))

(defn -main [& _]
  (let [cpu-backend (cpu/cpu-backend)
        expected-output (compile-and-run cpu-backend)
        expected (arr/->vec expected-output)
        expected-denoise (arr/->vec (denoise cpu-backend))
        [expected-nchw expected-image] (image-conversions cpu-backend)
        expected-nchw-values (arr/->vec expected-nchw)
        expected-image-values (arr/->vec expected-image)
        expected-transformer (arr/->vec (spatial-transformer cpu-backend))
        expected-batched (arr/->vec (broadcast-batched-matmul cpu-backend))
        expected-vae-attention (arr/->vec (vae-attention cpu-backend))
        expected-cross-attention (arr/->vec (legacy-cross-attention cpu-backend))]
    (-> (dg/request-device)
        (.then
         (fn [request]
           (let [gpu (dg/backend request)
                 output (compile-and-run gpu)
                 [gpu-nchw gpu-image] (image-conversions gpu)]
             (.then
              (js/Promise.all
               #js [(arr/->vec output) (arr/->vec (denoise gpu))
                    (arr/->vec gpu-nchw) (arr/->vec gpu-image)
                    (arr/->vec (spatial-transformer gpu))
                    (arr/->vec (broadcast-batched-matmul gpu))
                    (arr/->vec (vae-attention gpu))
                    (arr/->vec (legacy-cross-attention gpu))])
              (fn [results]
                (let [actual (aget results 0)
                      actual-denoise (aget results 1)
                      actual-nchw (aget results 2)
                      actual-image (aget results 3)
                      actual-transformer (aget results 4)
                      actual-batched (aget results 5)
                      actual-vae-attention (aget results 6)
                      actual-cross-attention (aget results 7)
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
                           (close? expected-image-values actual-image)
                           (close? expected-transformer actual-transformer)
                           (close? expected-batched actual-batched)
                           (close? expected-vae-attention actual-vae-attention)
                           (close? expected-cross-attention actual-cross-attention))
                    (throw
                     (ex-info "Metal liveness graphs differ from CPU"
                              {:shape (:shape output)
                               :expected expected :actual actual
                               :expected-denoise expected-denoise
                               :actual-denoise actual-denoise
                               :expected-nchw expected-nchw-values
                               :actual-nchw actual-nchw
                               :expected-image expected-image-values
                               :actual-image actual-image
                               :expected-transformer expected-transformer
                               :actual-transformer actual-transformer
                               :expected-batched expected-batched
                               :actual-batched actual-batched
                               :expected-vae-attention expected-vae-attention
                               :actual-vae-attention actual-vae-attention
                               :expected-cross-attention expected-cross-attention
                               :actual-cross-attention actual-cross-attention})))
                  (println "OK VAE attention + image-boundary + SpatialTransformer graphs match CPU with"
                           "GPUBuffer.destroy on"
                           (dg/adapter-description request))))))))
        (.catch
         (fn [error]
           (println "ERROR:" (or (.-stack error) (str error)))
           (js/Deno.exit 1))))))

(set! *main-cli-fn* -main)
