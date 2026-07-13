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

(deftest iterative-ddim-runs-cfg-over-descending-timesteps
  (let [sample (arr/from-vec backend [2.0] [1])
        uncond (arr/from-vec backend [0.1] [1])
        cond (arr/from-vec backend [0.3] [1])
        events (atom [])
        result (scheduler/ddim-sample
                {:sample sample
                 :alphas [0.9 0.7 0.5]
                 :timesteps [2 1]
                 :negative uncond :positive cond :cfg 2.0
                 :denoise-fn (fn [_ _ conditioning] conditioning)
                 :on-step #(swap! events conj %)})
        guided (scheduler/classifier-free-guidance uncond cond 2.0)
        step1 (:previous-sample (scheduler/ddim-step sample guided 0.5 0.7))
        expected (:previous-sample (scheduler/ddim-step step1 guided 0.7 1.0))]
    (is (= [2 0] (scheduler/select-timesteps 3 2)))
    (is (= [2 1] (mapv :timestep @events)))
    (is (approx? (first (arr/->vec expected))
                 (first (arr/->vec (:sample result)))))))

(deftest euler-discrete-integrates-the-sigma-schedule
  (let [sample (arr/from-vec backend [2.0] [1])
        uncond (arr/from-vec backend [0.1] [1])
        cond (arr/from-vec backend [0.3] [1])
        alphas [0.9 0.7 0.5]
        events (atom [])
        result (scheduler/euler-sample
                {:sample sample :alphas alphas :timesteps [2 1]
                 :negative uncond :positive cond :cfg 2.0
                 :denoise-fn (fn [_model-input _timestep conditioning]
                               conditioning)
                 :on-step #(swap! events conj %)})
        epsilon 0.5
        sigma2 (scheduler/alpha->sigma 0.5)
        sigma1 (scheduler/alpha->sigma 0.7)
        expected (+ 2.0 (* epsilon (- sigma1 sigma2)) (* epsilon (- 0.0 sigma1)))]
    (is (approx? 1.0 sigma2))
    (is (= [2 1] (mapv :timestep @events)))
    (is (zero? (:sigma-next (last @events))))
    (is (approx? expected (first (arr/->vec (:sample result)))))))
