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
   (when-not (and (< 0 alpha 1.0) (< 0 alpha-prev 1.0) (<= 0 eta))
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
