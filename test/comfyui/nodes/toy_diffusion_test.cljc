(ns comfyui.nodes.toy-diffusion-test
  "The full chain this ADR-2607131500 phase set out to prove end-to-end: a
  ComfyUI-API-format workflow, run through the REAL comfyui.exec topological
  executor, dispatching a node whose :fn does REAL tensor math (conv2d + relu
  + attention) via kotoba-lang/num — not a shape contract, not a proxy to
  Python. conv2d/attention's own arithmetic is already exhaustively hand-
  verified in num's own test suite (num.tensor-test); this test's job is the
  WIRING — node registry -> exec engine -> real :fn -> real compute — plus a
  sanity bound on the output (attention output is a convex combination of
  its V rows, so it must fall within [min(V) max(V)])."
  (:require [clojure.test :refer [deftest is testing]]
            [num.array :as arr]
            [num.cpu :as cpu]
            [comfyui.exec :as exec]
            [comfyui.node :as node]
            [comfyui.nodes.toy-diffusion :as toy]))

(def backend (cpu/cpu-backend))

(deftest toy-diffusion-step-runs-real-compute-through-the-real-executor
  (testing "a 1-node workflow dispatches ToyDiffusionStep's real :fn through
            comfyui.exec/execute (topological order, content-addressed cache
            key computation, run bookkeeping) — the same engine any real
            ComfyUI-API-format graph runs through"
    (let [reg (node/register! (node/registry) toy/pack)
          image (arr/from-vec backend (range 1 10) [3 3])   ; [[1 2 3][4 5 6][7 8 9]]
          kernel (arr/from-vec backend [1 1 1 1] [2 2])       ; all-ones 2x2
          workflow {"1" {:class_type "ToyDiffusionStep"
                         :inputs {:image image :kernel kernel}}}
          {:keys [results executed]} (exec/execute {:registry reg :cache nil} workflow)]
      (is (= ["1"] executed))
      (let [[out] (get results "1")]
        (is (= [2 2] (:shape out)))
        ;; conv+relu of this fixture is [[12 16][24 28]] (hand-verified in
        ;; num.tensor-test) BEFORE attention; attention's output is a convex
        ;; combination of those same 4 values (V = the conv+relu output
        ;; itself), so every output element must land in [12 28].
        (let [vs (arr/->vec out)]
          (is (every? #(<= 12.0 % 28.0) vs)
              (str "attention output out of the convex-combination bound: " vs)))))))

(deftest unregistered-node-type-fails-loudly
  (testing "an empty registry can't run the workflow — validation catches it,
            not a silent no-op"
    (let [image (arr/from-vec backend (range 1 10) [3 3])
          kernel (arr/from-vec backend [1 1 1 1] [2 2])
          workflow {"1" {:class_type "ToyDiffusionStep"
                         :inputs {:image image :kernel kernel}}}]
      (is (thrown? #?(:clj Exception :cljs js/Error)
                   (exec/execute {:registry (node/registry) :cache nil} workflow))))))
