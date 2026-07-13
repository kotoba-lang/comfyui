(ns comfyui.diffusion.scheduler-test
  (:require [comfyui.diffusion.scheduler :as scheduler]
            [num.array :as arr]
            [num.cpu :as cpu]
            #?(:clj [clojure.test :refer [deftest is testing]]
               :cljs [cljs.test :refer-macros [deftest is testing]])))

(def backend (cpu/cpu-backend))
(defn- approx? [a b] (< (Math/abs (- (double a) (double b))) 1.0e-10))

(deftest linear-schedule-and-cumulative-alphas
  (is (= [0.1 0.2 0.3]
         (scheduler/linear-betas 3 0.1 0.3)))
  (let [alphas (scheduler/alphas-cumprod [0.1 0.2 0.3])]
    (is (every? true? (map approx? alphas [0.9 0.72 0.504]))))
  (is (thrown? #?(:clj Exception :cljs js/Error)
               (scheduler/linear-betas 0 0.1 0.2))))

(deftest forward-noise-equation-is-numeric
  (let [clean (arr/from-vec backend [2.0 -1.0] [1 1 1 2])
        noise (arr/from-vec backend [4.0 2.0] [1 1 1 2])
        out (arr/->vec (scheduler/add-noise clean noise 0.25))]
    (is (every? true?
                (map approx? out
                     [(+ 1.0 (* 4.0 (Math/sqrt 0.75)))
                      (+ -0.5 (* 2.0 (Math/sqrt 0.75)))])))))

(deftest deterministic-ddim-step-matches-the-equation
  (let [sample (arr/from-vec backend [3.0] [1 1 1 1])
        epsilon (arr/from-vec backend [1.0] [1 1 1 1])
        {:keys [previous-sample predicted-original-sample sigma]}
        (scheduler/ddim-step sample epsilon 0.25 0.5)
        x0 (/ (- 3.0 (Math/sqrt 0.75)) 0.5)
        expected-prev (+ (* (Math/sqrt 0.5) x0) (Math/sqrt 0.5))]
    (is (zero? sigma))
    (is (approx? x0 (first (arr/->vec predicted-original-sample))))
    (is (approx? expected-prev (first (arr/->vec previous-sample))))))

(deftest stochastic-ddim-requires-and-uses-noise
  (let [sample (arr/from-vec backend [1.0] [1])
        epsilon (arr/from-vec backend [0.2] [1])
        noise (arr/from-vec backend [0.5] [1])]
    (is (thrown? #?(:clj Exception :cljs js/Error)
                 (scheduler/ddim-step sample epsilon 0.4 0.6 {:eta 0.5})))
    (is (number? (:sigma (scheduler/ddim-step sample epsilon 0.4 0.6
                                               {:eta 0.5 :noise noise}))))))
