(ns comfyui.diffusion.model-test
  (:require [clojure.test :refer [deftest is testing]]
            [comfyui.diffusion.model :as model]
            [num.array :as arr]
            [num.cpu :as cpu]))

(def backend (cpu/cpu-backend))

(defn- identity-values [rows columns scale]
  (mapv (fn [index]
          (let [row (quot index columns) column (mod index columns)]
            (if (= row column) scale 0.0)))
        (range (* rows columns))))

(deftest learned-cross-attention-and-timestep-embedding-execute
  (let [arrays
        {"q.w" (arr/from-vec backend (identity-values 4 4 1.0) [4 4])
         "q.b" (arr/from-vec backend [0 0 0 0] [4])
         "k.w" (arr/from-vec backend
                             [1 0 0, 0 1 0, 0 0 1, 0.25 0.25 0.25] [4 3])
         "k.b" (arr/from-vec backend [0 0 0 0] [4])
         "v.w" (arr/from-vec backend
                             [0.5 0 0, 0 0.5 0, 0 0 0.5, 0.1 0.2 0.3] [4 3])
         "v.b" (arr/from-vec backend [0 0 0 0] [4])
         "o.w" (arr/from-vec backend (identity-values 4 4 0.75) [4 4])
         "o.b" (arr/from-vec backend [0.01 -0.02 0.03 -0.04] [4])
         "time.0.w" (arr/from-vec backend (identity-values 4 4 0.1) [4 4])
         "time.0.b" (arr/from-vec backend [0 0 0 0] [4])
         "time.2.w" (arr/from-vec backend (identity-values 4 4 0.2) [4 4])
         "time.2.b" (arr/from-vec backend [0 0 0 0] [4])}
        reads (atom 0)
        component {:comfyui/read-tensor
                   (fn [_ tensor-name] (swap! reads inc) (get arrays tensor-name))}
        spec {:layers
              [{:op :timestep-embedding
                :first-weight "time.0.w" :first-bias "time.0.b"
                :second-weight "time.2.w" :second-bias "time.2.b"}
               {:op :cross-attention :heads 2
                :query-weight "q.w" :query-bias "q.b"
                :key-weight "k.w" :key-bias "k.b"
                :value-weight "v.w" :value-bias "v.b"
                :output-weight "o.w" :output-bias "o.b"}]}
        denoise (model/compile-denoiser component backend spec)
        sample (arr/from-vec backend
                             [0.2 -0.1 0.3 0.4, -0.2 0.1 0.5 -0.3,
                              0.6 0.2 -0.4 0.1, 0.1 0.7 -0.2 0.3]
                             [1 4 2 2])
        condition-a (arr/from-vec backend
                                  [1 0 0, 0 1 0, 0 0 1] [1 3 3])
        condition-b (arr/from-vec backend
                                  [0.2 0.3 0.4, -0.1 0.5 0.2, 0.7 -0.2 0.1]
                                  [1 3 3])
        output-a (denoise sample 10 condition-a)
        output-b (denoise sample 10 condition-b)
        output-time (denoise sample 20 condition-a)]
    (is (= [1 4 2 2] (:shape output-a)))
    (is (not= (arr/->vec output-a) (arr/->vec output-b)))
    (is (not= (arr/->vec output-a) (arr/->vec output-time)))
    (is (= 12 @reads) "each checkpoint tensor is loaded once")
    (denoise sample 10 condition-a)
    (is (= 12 @reads) "subsequent denoising reuses the tensor cache")
    (testing "cross-attention rejects a ComfyUI conditioning shape it cannot lower"
      (is (thrown? clojure.lang.ExceptionInfo
                   (denoise sample 10 (arr/from-vec backend [1 2 3] [3])))))))
