(ns comfyui.nodes.diffusion-runtime
  "Executable JVM diffusion nodes. Unlike `comfyui.nodes.diffusion`'s shape
  contracts, every node in this pack performs real work: checkpoint catalog
  loading, latent allocation, or a DDIM transition."
  (:require [clojure.string :as str]
            [comfyui.diffusion.model :as diffusion-model]
            [comfyui.diffusion.scheduler :as scheduler]
            [comfyui.safetensors :as safe]
            [num.array :as arr])
  (:import [java.util Random]))

(defn- component [checkpoint kind prefixes]
  (let [names (filterv #(some (fn [prefix] (str/starts-with? % prefix)) prefixes)
                       (safe/tensor-names checkpoint))]
    {:comfyui/component kind
     :comfyui/checkpoint checkpoint
     :comfyui/tensor-names names
     :comfyui/read-tensor (fn [backend tensor-name]
                            (when-not (some #{tensor-name} names)
                              (throw (ex-info "tensor does not belong to checkpoint component"
                                              {:component kind :tensor tensor-name})))
                            (safe/read-tensor checkpoint backend tensor-name))}))

(defn pack
  "Build executable diffusion nodes.

  Options:
  - `:backend` num backend used for latent allocation
  - `:resolve-checkpoint` maps ComfyUI's ckpt_name to a filesystem path
  - `:model-spec` graph map, or `(fn [ckpt-name checkpoint] graph-map)`
  - `:alphas-cumprod` schedule vector, or a resolver function

  The MODEL/CLIP/VAE maps share one lazy SafeTensorFile. The host owns its
  lifecycle and closes `:comfyui/checkpoint` when the workflow/model unloads."
  [{:keys [backend resolve-checkpoint model-spec alphas-cumprod]
    :or {resolve-checkpoint identity}}]
  [{:type "CheckpointLoaderSimple"
    :category "loaders"
    :inputs {:ckpt_name {:type "STRING"}}
    :outputs [{:name "MODEL" :type "MODEL"}
              {:name "CLIP" :type "CLIP"}
              {:name "VAE" :type "VAE"}]
    :fn (fn [{:keys [ckpt_name]}]
          (let [checkpoint (safe/open-file (resolve-checkpoint ckpt_name))
                model-component (component checkpoint :model
                                           ["model.diffusion_model."
                                            "diffusion_model." "unet."])
                spec (if (fn? model-spec)
                       (model-spec ckpt_name checkpoint) model-spec)
                alphas (if (fn? alphas-cumprod)
                         (alphas-cumprod ckpt_name checkpoint) alphas-cumprod)
                executable-model
                (cond-> model-component
                  spec (assoc :comfyui/model-spec spec
                              :comfyui/denoise
                              (diffusion-model/compile-denoiser
                               model-component backend spec))
                  alphas (assoc :comfyui/alphas-cumprod (vec alphas)))]
            [executable-model
             (component checkpoint :clip
                        ["cond_stage_model." "text_encoder." "clip."])
             (component checkpoint :vae
                        ["first_stage_model." "vae."])]))}
   {:type "EmptyLatentImage"
    :category "latent"
    :inputs {:width {:type "INT" :default 512}
             :height {:type "INT" :default 512}
             :batch_size {:type "INT" :default 1}}
    :outputs [{:name "LATENT" :type "LATENT"}]
    :fn (fn [{:keys [width height batch_size]}]
          (when-not (and backend (zero? (mod width 8)) (zero? (mod height 8))
                         (pos? width) (pos? height) (pos? batch_size))
            (throw (ex-info "latent dimensions require backend and positive multiples of 8"
                            {:width width :height height :batch-size batch_size})))
          [(arr/zeros backend [batch_size 4 (quot height 8) (quot width 8)])])}
   {:type "DDIMStep"
    :category "sampling"
    :inputs {:sample {:type "LATENT"}
             :epsilon {:type "LATENT"}
             :alpha {:type "FLOAT"}
             :alpha_prev {:type "FLOAT"}
             :eta {:type "FLOAT" :default 0.0}
             :noise {:type "LATENT" :optional true}}
    :outputs [{:name "previous_sample" :type "LATENT"}
              {:name "predicted_original_sample" :type "LATENT"}]
   :fn (fn [{:keys [sample epsilon alpha alpha_prev eta noise]}]
          (let [{:keys [previous-sample predicted-original-sample]}
                (scheduler/ddim-step sample epsilon alpha alpha_prev
                                     {:eta eta :noise noise})]
            [previous-sample predicted-original-sample]))}
   {:type "KSampler"
    :category "sampling"
    :inputs {:model {:type "MODEL"}
             :positive {:type "CONDITIONING"}
             :negative {:type "CONDITIONING"}
             :latent_image {:type "LATENT"}
             :seed {:type "INT" :default 0}
             :steps {:type "INT" :default 20}
             :cfg {:type "FLOAT" :default 7.0}
             :sampler_name {:type "STRING" :default "ddim"}
             :scheduler {:type "STRING" :default "normal"}
             :denoise {:type "FLOAT" :default 1.0}}
    :outputs [{:name "LATENT" :type "LATENT"}]
    :fn (fn [{:keys [model positive negative latent_image seed steps cfg
                     sampler_name scheduler denoise]}]
          (when-not (and (= "ddim" sampler_name) (= "normal" scheduler)
                         (= 1.0 (double denoise)))
            (throw (ex-info "runtime KSampler currently supports ddim/normal/full-denoise"
                            {:sampler-name sampler_name :scheduler scheduler :denoise denoise})))
          (let [denoise-fn (:comfyui/denoise model)
                alphas (:comfyui/alphas-cumprod model)
                sample (if (and (map? latent_image) (contains? latent_image :samples))
                         (:samples latent_image)
                         latent_image)
                _ (when-not (and (fn? denoise-fn) (seq alphas) sample)
                    (throw (ex-info "MODEL lacks executable denoiser or alpha schedule"
                                    {:model-keys (keys model)})))
                random (Random. (long seed))
                noise-fn (fn [shape _]
                           (arr/from-vec (:backend sample)
                                         (repeatedly (arr/nelems shape) #(.nextGaussian random))
                                         shape))
                result (scheduler/ddim-sample
                        {:sample sample
                         :alphas alphas
                         :timesteps (scheduler/select-timesteps (count alphas) steps)
                         :denoise-fn denoise-fn
                         :positive positive :negative negative :cfg cfg
                         :eta 0.0 :noise-fn noise-fn})]
            [(:sample result)]))}])
