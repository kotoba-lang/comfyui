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

(defn scaled-linear-betas
  "Diffusers/CompVis `scaled_linear`: interpolate square roots of the beta
  endpoints, then square each value. Stable Diffusion checkpoints use this
  schedule rather than direct linear beta interpolation."
  [steps beta-start beta-end]
  (when-not (and (pos-int? steps) (< 0 beta-start beta-end 1))
    (throw (ex-info "invalid scaled-linear beta schedule"
                    {:steps steps :beta-start beta-start :beta-end beta-end})))
  (let [start (Math/sqrt beta-start) end (Math/sqrt beta-end)]
    (if (= steps 1)
      [(double beta-start)]
      (mapv (fn [i]
              (let [value (+ start (* (/ i (double (dec steps))) (- end start)))]
                (* value value)))
            (range steps)))))

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
  "Select Diffusers-compatible descending inference timesteps."
  ([training-steps steps] (select-timesteps training-steps steps {}))
  ([training-steps steps {:keys [spacing steps-offset]
                          :or {spacing "linspace" steps-offset 0}}]
   (when-not (and (pos-int? training-steps) (pos-int? steps) (<= steps training-steps))
     (throw (ex-info "invalid inference timestep count"
                     {:training-steps training-steps :steps steps})))
   (case spacing
     "linspace"
     (if (= steps 1)
       [(dec training-steps)]
       (mapv (fn [i]
               (long (Math/round (* (- 1.0 (/ i (double (dec steps))))
                                    (dec training-steps)))))
             (range steps)))
     "leading"
     (let [ratio (quot training-steps steps)]
       (mapv #(+ (long steps-offset) (* ratio %)) (reverse (range steps))))
     "trailing"
     (let [ratio (/ training-steps (double steps))]
       (mapv #(dec (long (Math/round (- training-steps (* % ratio)))))
             (range steps)))
     (throw (ex-info "unsupported timestep spacing" {:spacing spacing})))))

(defn ddim-sample
  "Iterative classifier-free-guided DDIM sampling.

  `denoise-fn` receives `(sample timestep conditioning)` and returns predicted
  epsilon. `alphas` is the model training schedule; `timesteps` are descending
  indices into it. `noise-fn`, required only for eta>0, receives
  `(shape timestep)`. Returns the final latent and per-step audit history."
  [{:keys [sample alphas timesteps denoise-fn positive negative cfg eta noise-fn on-step
           final-alpha]
    :or {cfg 1.0 eta 0.0 final-alpha 1.0}}]
  (when-not (and sample (seq alphas) (seq timesteps) (fn? denoise-fn))
    (throw (ex-info "DDIM sampler requires sample, alphas, timesteps, and denoise-fn" {})))
  (loop [current sample remaining (seq timesteps) history []]
    (if-let [timestep (first remaining)]
      (let [next-timestep (second remaining)
            alpha (double (nth alphas timestep))
            alpha-prev (if (some? next-timestep)
                         (double (nth alphas next-timestep))
                         (double final-alpha))
            unconditional (denoise-fn current timestep negative)
            conditional (denoise-fn current timestep positive)
            epsilon (classifier-free-guidance unconditional conditional cfg)
            noise (when (pos? eta)
                    (when-not noise-fn
                      (throw (ex-info "DDIM eta>0 requires noise-fn" {:eta eta})))
                    (noise-fn (:shape current) timestep))
            step (ddim-step current epsilon alpha alpha-prev {:eta eta :noise noise})
            event {:timestep timestep :alpha alpha :alpha-prev alpha-prev
                   :sigma (:sigma step) :epsilon epsilon
                   :sample (:previous-sample step)}]
        (when on-step (on-step event))
        (recur (:previous-sample step) (next remaining) (conj history event)))
      {:sample current :history history})))

(defn alpha->sigma
  "Convert cumulative alpha to the Euler/Karras noise level
  `sqrt((1-alpha)/alpha)`."
  [alpha]
  (when-not (< 0.0 alpha 1.0)
    (throw (ex-info "alpha must be in (0,1)" {:alpha alpha})))
  (Math/sqrt (/ (- 1.0 (double alpha)) (double alpha))))

(defn karras-sigmas
  "Construct k-diffusion's Karras noise schedule, including terminal zero."
  ([steps sigma-min sigma-max] (karras-sigmas steps sigma-min sigma-max 7.0))
  ([steps sigma-min sigma-max rho]
   (when-not (and (pos-int? steps) (< 0.0 sigma-min sigma-max) (pos? rho))
     (throw (ex-info "invalid Karras schedule"
                     {:steps steps :sigma-min sigma-min
                      :sigma-max sigma-max :rho rho})))
   (let [min-root (Math/pow sigma-min (/ 1.0 rho))
         max-root (Math/pow sigma-max (/ 1.0 rho))]
     (conj
      (mapv (fn [index]
              (let [ramp (if (= steps 1) 0.0 (/ index (double (dec steps))))]
                (Math/pow (+ max-root (* ramp (- min-root max-root))) rho)))
            (range steps))
      0.0))))

(defn sigma->timestep
  "Map a continuous sigma to a fractional discrete model timestep by linear
  interpolation in log-sigma space, matching k-diffusion DiscreteSchedule."
  [alphas sigma]
  (when-not (and (seq alphas) (pos? sigma))
    (throw (ex-info "sigma->timestep requires alphas and positive sigma"
                    {:sigma sigma})))
  (let [logs (mapv #(Math/log (alpha->sigma (double %))) alphas)
        target (Math/log sigma)
        last-index (dec (count logs))]
    (cond
      (<= target (first logs)) 0.0
      (>= target (last logs)) (double last-index)
      :else
      (let [low (first (filter #(<= (nth logs %) target (nth logs (inc %)))
                               (range last-index)))
            low-log (nth logs low) high-log (nth logs (inc low))
            weight (/ (- target low-log) (- high-log low-log))]
        (+ low weight)))))

(defn- resolve-sigmas [alphas timesteps sigmas]
  (if sigmas
    (do
      (when-not (and (= (inc (count timesteps)) (count sigmas))
                     (zero? (last sigmas))
                     (every? pos? (butlast sigmas)))
        (throw (ex-info "explicit sigma schedule must have one positive value per timestep plus zero"
                        {:timesteps (count timesteps) :sigmas sigmas})))
      (mapv double sigmas))
    (conj (mapv #(alpha->sigma (double (nth alphas %))) timesteps) 0.0)))

(defn euler-sample
  "Classifier-free-guided Euler discrete sampling for epsilon-prediction
  models. Model input is scaled by `1/sqrt(sigma^2+1)`; each step integrates
  the ODE derivative from the current training sigma to the next (final 0)."
  [{:keys [sample alphas timesteps sigmas denoise-fn positive negative cfg on-step]
    :or {cfg 1.0}}]
  (when-not (and sample (seq alphas) (seq timesteps) (fn? denoise-fn))
    (throw (ex-info "Euler sampler requires sample, alphas, timesteps, and denoise-fn" {})))
  (loop [current sample remaining (seq timesteps)
         sigma-remaining (seq (resolve-sigmas alphas timesteps sigmas)) history []]
    (if-let [timestep (first remaining)]
      (let [sigma (first sigma-remaining)
            sigma-next (second sigma-remaining)
            model-input (scale current (/ 1.0 (Math/sqrt (+ 1.0 (* sigma sigma)))))
            unconditional (denoise-fn model-input timestep negative)
            conditional (denoise-fn model-input timestep positive)
            epsilon (classifier-free-guidance unconditional conditional cfg)
            predicted-x0 (t/sub current (scale epsilon sigma))
            dt (- sigma-next sigma)
            next-sample (t/add current (scale epsilon dt))
            event {:timestep timestep :sigma sigma :sigma-next sigma-next
                   :predicted-original-sample predicted-x0}]
        (when on-step (on-step event))
        (recur next-sample (next remaining) (next sigma-remaining)
               (conj history event)))
      {:sample current :history history})))

(defn euler-ancestral-sample
  "Euler ancestral sampling. Each transition decomposes target sigma into a
  deterministic `sigma-down` step plus fresh noise at `sigma-up`; `eta=1`
  matches the standard ancestral schedule."
  [{:keys [sample alphas timesteps sigmas denoise-fn positive negative cfg eta noise-fn on-step]
    :or {cfg 1.0 eta 1.0}}]
  (when-not (and sample (seq alphas) (seq timesteps) (fn? denoise-fn)
                 (<= 0.0 eta))
    (throw (ex-info "Euler ancestral sampler received invalid arguments" {:eta eta})))
  (loop [current sample remaining (seq timesteps)
         sigma-remaining (seq (resolve-sigmas alphas timesteps sigmas)) history []]
    (if-let [timestep (first remaining)]
      (let [sigma (first sigma-remaining)
            sigma-next (second sigma-remaining)
            model-input (scale current (/ 1.0 (Math/sqrt (+ 1.0 (* sigma sigma)))))
            epsilon (classifier-free-guidance
                     (denoise-fn model-input timestep negative)
                     (denoise-fn model-input timestep positive) cfg)
            sigma-up (if (zero? sigma-next)
                       0.0
                       (min sigma-next
                            (* eta
                               (Math/sqrt
                                (max 0.0
                                     (/ (* sigma-next sigma-next
                                           (- (* sigma sigma)
                                              (* sigma-next sigma-next)))
                                        (* sigma sigma)))))))
            sigma-down (Math/sqrt (max 0.0 (- (* sigma-next sigma-next)
                                                (* sigma-up sigma-up))))
            mean (t/add current (scale epsilon (- sigma-down sigma)))
            noise (when (pos? sigma-up)
                    (when-not noise-fn
                      (throw (ex-info "Euler ancestral sampling requires noise-fn" {})))
                    (noise-fn (:shape current) timestep))
            next-sample (if noise (t/add mean (scale noise sigma-up)) mean)
            event {:timestep timestep :sigma sigma :sigma-next sigma-next
                   :sigma-up sigma-up :sigma-down sigma-down}]
        (when on-step (on-step event))
        (recur next-sample (next remaining) (next sigma-remaining)
               (conj history event)))
      {:sample current :history history})))

(defn dpmpp-2m-sample
  "Deterministic k-diffusion DPM-Solver++(2M) sampling.

  The denoiser predicts epsilon, so each call is converted to the denoised
  estimate `x0 = x - sigma*epsilon`. Non-final steps use the exponential
  first-order update, upgraded from the second step onward by the previous
  denoised estimate. The terminal sigma-zero step returns x0 exactly."
  [{:keys [sample alphas timesteps sigmas denoise-fn positive negative cfg on-step]
    :or {cfg 1.0}}]
  (when-not (and sample (seq alphas) (seq timesteps) (fn? denoise-fn))
    (throw (ex-info "DPM++ 2M sampler requires sample, alphas, timesteps, and denoise-fn"
                    {})))
  (loop [current sample remaining (seq timesteps)
         sigma-remaining (seq (resolve-sigmas alphas timesteps sigmas))
         old-denoised nil previous-time nil history []]
    (if-let [timestep (first remaining)]
      (let [sigma (first sigma-remaining)
            sigma-next (second sigma-remaining)
            model-input (scale current (/ 1.0 (Math/sqrt (+ 1.0 (* sigma sigma)))))
            epsilon (classifier-free-guidance
                     (denoise-fn model-input timestep negative)
                     (denoise-fn model-input timestep positive) cfg)
            denoised (t/sub current (scale epsilon sigma))
            time (- (Math/log sigma))
            next-sample
            (if (zero? sigma-next)
              denoised
              (let [next-time (- (Math/log sigma-next))
                    h (- next-time time)
                    effective-denoised
                    (if old-denoised
                      (let [h-last (- time previous-time)
                            r (/ h-last h)
                            previous-factor (/ 1.0 (* 2.0 r))]
                        (t/sub (scale denoised (+ 1.0 previous-factor))
                               (scale old-denoised previous-factor)))
                      denoised)
                    ratio (/ sigma-next sigma)
                    denoised-factor (- (Math/expm1 (- h)))]
                (t/add (scale current ratio)
                       (scale effective-denoised denoised-factor))))
            event {:timestep timestep :sigma sigma :sigma-next sigma-next
                   :order (if (and old-denoised (pos? sigma-next)) 2 1)
                   :predicted-original-sample denoised :sample next-sample}]
        (when on-step (on-step event))
        (recur next-sample (next remaining) (next sigma-remaining) denoised time
               (conj history event)))
      {:sample current :history history})))
