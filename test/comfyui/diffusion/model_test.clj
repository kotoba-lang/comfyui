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

(deftest sdxl-pooled-time-ids-flow-through-label-and-resblock-embedding
  (let [zeros4 (arr/from-vec backend [0 0 0 0] [4])
        arrays
        {"time.0.w" (arr/from-vec backend (identity-values 4 4 0.3) [4 4])
         "time.0.b" zeros4
         "time.2.w" (arr/from-vec backend (identity-values 4 4 0.4) [4 4])
         "time.2.b" zeros4
         "label.0.w" (arr/from-vec backend
                                       (mapv #(* 0.002 (inc %)) (range 56)) [4 14])
         "label.0.b" zeros4
         "label.2.w" (arr/from-vec backend (identity-values 4 4 0.5) [4 4])
         "label.2.b" zeros4
         "emb.w" (arr/from-vec backend [0.3 0.1 -0.2 0.4,
                                         -0.1 0.2 0.5 0.3] [2 4])
         "emb.b" (arr/from-vec backend [0.01 -0.02] [2])}
        reads (atom 0)
        component {:comfyui/read-tensor
                   (fn [_ name] (swap! reads inc) (get arrays name))}
        spec {:layers
              [{:op :timestep-vector
                :first-weight "time.0.w" :first-bias "time.0.b"
                :second-weight "time.2.w" :second-bias "time.2.b"}
               {:op :sdxl-label-embedding
                :first-weight "label.0.w" :first-bias "label.0.b"
                :second-weight "label.2.w" :second-bias "label.2.b"}
               {:op :add-embedding :weight "emb.w" :bias "emb.b"}]}
        denoise (model/compile-denoiser component backend spec)
        sample (arr/zeros backend [1 2 2 2])
        pooled (arr/from-vec backend [0.25 -0.5] [1 2])
        condition-a {:pooled pooled :time-ids [1024 1024 0 0 1024 1024]}
        condition-b {:pooled pooled :time-ids [768 1344 16 0 768 1344]}
        output-a (denoise sample 10 condition-a)
        output-b (denoise sample 10 condition-b)]
    (is (= [1 2 2 2] (:shape output-a)))
    (is (not= (arr/->vec output-a) (arr/->vec output-b)))
    (is (= 10 @reads))
    (denoise sample 20 condition-a)
    (is (= 10 @reads) "SDXL embedding tensors remain cached")))

(deftest standard-compvis-resblock-executes-learned-skip-and-embedding
  (let [norm-w (arr/from-vec backend [1 1] [2])
        norm-b (arr/from-vec backend [0 0] [2])
        conv-w (arr/from-vec backend
                             (vec (for [out (range 2) in (range 2)
                                        row (range 3) col (range 3)]
                                    (if (and (= out in) (= row 1) (= col 1))
                                      0.25 0.0))) [2 2 3 3])
        arrays {"time.0.w" (arr/from-vec backend (identity-values 4 4 0.2) [4 4])
                "time.0.b" (arr/zeros backend [4])
                "time.2.w" (arr/from-vec backend (identity-values 4 4 0.3) [4 4])
                "time.2.b" (arr/zeros backend [4])
                "in.norm.w" norm-w "in.norm.b" norm-b
                "in.conv.w" conv-w "in.conv.b" norm-b
                "emb.w" (arr/from-vec backend [0.2 0.1 -0.1 0.3,
                                                -0.2 0.4 0.1 0.2] [2 4])
                "emb.b" norm-b
                "out.norm.w" norm-w "out.norm.b" norm-b
                "out.conv.w" conv-w "out.conv.b" norm-b
                "skip.w" (arr/from-vec backend (identity-values 2 2 0.75) [2 2 1 1])
                "skip.b" norm-b}
        component {:comfyui/read-tensor (fn [_ name] (get arrays name))}
        spec {:layers
              [{:op :timestep-vector
                :first-weight "time.0.w" :first-bias "time.0.b"
                :second-weight "time.2.w" :second-bias "time.2.b"}
               {:op :resblock :groups 1
                :in-norm-weight "in.norm.w" :in-norm-bias "in.norm.b"
                :in-conv-weight "in.conv.w" :in-conv-bias "in.conv.b"
                :embedding-weight "emb.w" :embedding-bias "emb.b"
                :out-norm-weight "out.norm.w" :out-norm-bias "out.norm.b"
                :out-conv-weight "out.conv.w" :out-conv-bias "out.conv.b"
                :skip-weight "skip.w" :skip-bias "skip.b"}]}
        denoise (model/compile-denoiser component backend spec)
        sample (arr/from-vec backend (mapv #(- (* 0.1 %) 0.4) (range 16))
                             [1 2 2 4])
        output-a (denoise sample 5 nil)
        output-b (denoise sample 15 nil)]
    (is (= (:shape sample) (:shape output-a)))
    (is (every? #(Double/isFinite %) (arr/->vec output-a)))
    (is (not= (arr/->vec output-a) (arr/->vec output-b)))))
