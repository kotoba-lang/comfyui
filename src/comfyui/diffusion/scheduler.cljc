(ns comfyui.diffusion.scheduler
  "Real diffusion noise scheduling and deterministic DDIM stepping over
  num NDArrays. No model assumptions: a UNet/DiT supplies epsilon prediction;
  this namespace performs the mathematically defined latent transition."
  (:require [num.array :as arr]
            [num.tensor :as t]))

(defn linear-betas
  "Inclusive linear beta schedule with `steps` entries."
  [steps beta-start beta-end]
  (when-not (and (pos-int? steps) (< 0 beta-start beta-end 1))
    (throw (ex-info "invalid linear beta schedule"
                    {:steps steps :beta-start beta-start :beta-end beta-end})))
  (if (= steps 1)
    [(double beta-start)]
    (mapv (fn [i]
            (+ (double beta-start)
               (* (/ i (double (dec steps))) (- (double beta-end) beta-start))))
          (range steps))))

(defn alphas-cumprod [betas]
  (loop [remaining betas product 1.0 out []]
    (if-let [beta (first remaining)]
      (let [product (* product (- 1.0 (double beta)))]
        (recur (next remaining) product (conj out product)))
      out)))

(defn- scalar [like value]
  (arr/from-vec (:backend like) [(double value)] []))

(defn- scale [tensor value]
  (t/mul tensor (scalar tensor value)))

(defn add-noise
  "Forward process `sqrt(alpha)*clean + sqrt(1-alpha)*noise`."
  [clean noise alpha-cumprod]
  (when-not (= (:shape clean) (:shape noise))
    (throw (ex-info "clean/noise shapes differ" {:clean (:shape clean) :noise (:shape noise)})))
  (t/add (scale clean (Math/sqrt alpha-cumprod))
         (scale noise (Math/sqrt (- 1.0 alpha-cumprod)))))

(defn ddim-step
  "One DDIM epsilon-prediction step. Returns both `:previous-sample` and
  `:predicted-original-sample`. `noise` is used only when eta>0; eta=0 is the
  deterministic sampler used for reproducible verification."
  ([sample epsilon alpha alpha-prev]
   (ddim-step sample epsilon alpha alpha-prev {}))
  ([sample epsilon alpha alpha-prev {:keys [eta noise] :or {eta 0.0}}]
   (when-not (= (:shape sample) (:shape epsilon))
     (throw (ex-info "sample/epsilon shapes differ"
                     {:sample (:shape sample) :epsilon (:shape epsilon)})))
   (when-not (and (< 0 alpha 1.0) (< 0 alpha-prev) (<= alpha-prev 1.0) (<= 0 eta))
     (throw (ex-info "invalid DDIM coefficients"
                     {:alpha alpha :alpha-prev alpha-prev :eta eta})))
   (let [predicted-x0 (scale (t/sub sample (scale epsilon (Math/sqrt (- 1.0 alpha))))
                             (/ 1.0 (Math/sqrt alpha)))
         sigma (* eta
                  (Math/sqrt (* (/ (- 1.0 alpha-prev) (- 1.0 alpha))
                                (- 1.0 (/ alpha alpha-prev)))))
         direction-scale (Math/sqrt (max 0.0 (- 1.0 alpha-prev (* sigma sigma))))
         mean (t/add (scale predicted-x0 (Math/sqrt alpha-prev))
                     (scale epsilon direction-scale))
         previous (if (pos? sigma)
                    (do
                      (when-not noise
                        (throw (ex-info "DDIM eta>0 requires noise" {:eta eta})))
                      (t/add mean (scale noise sigma)))
                    mean)]
     {:previous-sample previous
      :predicted-original-sample predicted-x0
      :sigma sigma})))

(defn classifier-free-guidance
  "`unconditional + cfg * (conditional - unconditional)`."
  [unconditional conditional cfg]
  (when-not (= (:shape unconditional) (:shape conditional))
    (throw (ex-info "CFG epsilon shapes differ"
                    {:unconditional (:shape unconditional)
                     :conditional (:shape conditional)})))
  (t/add unconditional (scale (t/sub conditional unconditional) cfg)))

(defn select-timesteps
  "Select `steps` evenly-spaced training timestep indices, descending."
  [training-steps steps]
  (when-not (and (pos-int? training-steps) (pos-int? steps) (<= steps training-steps))
    (throw (ex-info "invalid inference timestep count"
                    {:training-steps training-steps :steps steps})))
  (if (= steps 1)
    [(dec training-steps)]
    (mapv (fn [i]
            (long (Math/round (* (- 1.0 (/ i (double (dec steps))))
                                 (dec training-steps)))))
          (range steps))))

(defn ddim-sample
  "Iterative classifier-free-guided DDIM sampling.

  `denoise-fn` receives `(sample timestep conditioning)` and returns predicted
  epsilon. `alphas` is the model training schedule; `timesteps` are descending
  indices into it. `noise-fn`, required only for eta>0, receives
  `(shape timestep)`. Returns the final latent and per-step audit history."
  [{:keys [sample alphas timesteps denoise-fn positive negative cfg eta noise-fn on-step]
    :or {cfg 1.0 eta 0.0}}]
  (when-not (and sample (seq alphas) (seq timesteps) (fn? denoise-fn))
    (throw (ex-info "DDIM sampler requires sample, alphas, timesteps, and denoise-fn" {})))
  (loop [current sample remaining (seq timesteps) history []]
    (if-let [timestep (first remaining)]
      (let [next-timestep (second remaining)
            alpha (double (nth alphas timestep))
            alpha-prev (if (some? next-timestep)
                         (double (nth alphas next-timestep))
                         1.0)
            unconditional (denoise-fn current timestep negative)
            conditional (denoise-fn current timestep positive)
            epsilon (classifier-free-guidance unconditional conditional cfg)
            noise (when (pos? eta)
                    (when-not noise-fn
                      (throw (ex-info "DDIM eta>0 requires noise-fn" {:eta eta})))
                    (noise-fn (:shape current) timestep))
            step (ddim-step current epsilon alpha alpha-prev {:eta eta :noise noise})
            event {:timestep timestep :alpha alpha :alpha-prev alpha-prev
                   :sigma (:sigma step)}]
        (when on-step (on-step event))
        (recur (:previous-sample step) (next remaining) (conj history event)))
      {:sample current :history history})))
