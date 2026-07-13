(ns comfyui.nodes.diffusion-runtime
  "Executable JVM diffusion nodes. Unlike `comfyui.nodes.diffusion`'s shape
  contracts, every node in this pack performs real work: checkpoint catalog
  loading, latent allocation, or a DDIM transition."
  (:require [clojure.string :as str]
            [comfyui.diffusion.scheduler :as scheduler]
            [comfyui.safetensors :as safe]
            [num.array :as arr]))

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

  The MODEL/CLIP/VAE maps share one lazy SafeTensorFile. The host owns its
  lifecycle and closes `:comfyui/checkpoint` when the workflow/model unloads."
  [{:keys [backend resolve-checkpoint] :or {resolve-checkpoint identity}}]
  [{:type "CheckpointLoaderSimple"
    :category "loaders"
    :inputs {:ckpt_name {:type "STRING"}}
    :outputs [{:name "MODEL" :type "MODEL"}
              {:name "CLIP" :type "CLIP"}
              {:name "VAE" :type "VAE"}]
    :fn (fn [{:keys [ckpt_name]}]
          (let [checkpoint (safe/open-file (resolve-checkpoint ckpt_name))]
            [(component checkpoint :model
                        ["model.diffusion_model." "diffusion_model." "unet."])
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
            [previous-sample predicted-original-sample]))}])
