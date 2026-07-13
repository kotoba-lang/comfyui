(ns comfyui.clip.encoder-test
  (:require [clojure.test :refer [deftest is]]
            [comfyui.clip.encoder :as clip]
            [num.array :as arr]
            [num.cpu :as cpu]))

(def backend (cpu/cpu-backend))

(defn- matrix [rows columns f]
  (arr/from-vec backend
                (vec (for [row (range rows) column (range columns)]
                       (double (f row column))))
                [rows columns]))

(defn- vector* [values] (arr/from-vec backend values [(count values)]))

(deftest checkpoint-backed-causal-clip-transformer-encodes-tokens
  (let [identity4 (matrix 4 4 #(if (= %1 %2) 1.0 0.0))
        zero4 (vector* [0 0 0 0])
        one4 (vector* [1 1 1 1])
        arrays
        {"token" (matrix 6 4 #(* 0.05 (+ 1 (* %1 4) %2)))
         "position" (matrix 5 4 #(* 0.01 (+ %1 %2)))
         "n1.w" one4 "n1.b" zero4
         "q.w" identity4 "q.b" zero4
         "k.w" identity4 "k.b" zero4
         "v.w" identity4 "v.b" zero4
         "o.w" identity4 "o.b" zero4
         "n2.w" one4 "n2.b" zero4
         "fc1.w" (matrix 8 4 #(if (= (mod %1 4) %2) 0.5 0.0))
         "fc1.b" (vector* (repeat 8 0.0))
         "fc2.w" (matrix 4 8 #(if (or (= %2 %1) (= %2 (+ 4 %1))) 0.25 0.0))
         "fc2.b" zero4
         "final.w" one4 "final.b" zero4}
        reads (atom 0)
        component {:comfyui/read-tensor
                   (fn [_ name] (swap! reads inc) (get arrays name))}
        layer {:norm1-weight "n1.w" :norm1-bias "n1.b"
               :query-weight "q.w" :query-bias "q.b"
               :key-weight "k.w" :key-bias "k.b"
               :value-weight "v.w" :value-bias "v.b"
               :output-weight "o.w" :output-bias "o.b"
               :norm2-weight "n2.w" :norm2-bias "n2.b"
               :fc1-weight "fc1.w" :fc1-bias "fc1.b"
               :fc2-weight "fc2.w" :fc2-bias "fc2.b"}
        spec {:token-embedding "token" :position-embedding "position"
              :layers [layer] :heads 2
              :final-norm-weight "final.w" :final-norm-bias "final.b"}
        encode (clip/compile-encoder component backend spec)
        tokens {:input-ids [1 2 3 4 0] :attention-mask [1 1 1 1 0]}
        result (encode tokens)]
    (is (= [1 5 4] (:shape (:tensor result))))
    (is (= [1 4] (:shape (:pooled result))))
    (is (every? #(Double/isFinite %) (arr/->vec (:tensor result))))
    (is (= 20 @reads))
    (encode tokens)
    (is (= 20 @reads) "all transformer weights remain cached")))
