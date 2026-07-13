(ns comfyui.nodes.diffusion-runtime
  "Executable JVM diffusion nodes. Unlike `comfyui.nodes.diffusion`'s shape
  contracts, every node in this pack performs real work: checkpoint catalog
  loading, latent allocation, or a DDIM transition."
  (:require [clojure.string :as str]
            [comfyui.diffusion.model :as diffusion-model]
            [comfyui.diffusion.scheduler :as scheduler]
            [comfyui.safetensors :as safe]
            [num.array :as arr]
            [num.tensor :as t])
  (:import [java.awt.image BufferedImage]
           [java.nio.file Files Paths]
           [java.util Random]
           [javax.imageio ImageIO]))

(defn- save-png-batch [images output-directory filename-prefix]
  (let [[batch height width channels :as shape] (:shape images)]
    (when-not (and (= 4 (count shape)) (= 3 channels))
      (throw (ex-info "SaveImage requires NHWC RGB images" {:shape shape})))
    (when-not output-directory
      (throw (ex-info "SaveImage requires :output-directory in runtime pack" {})))
    (let [directory (Paths/get (str output-directory) (make-array String 0))
          prefix (str/replace (or filename-prefix "ComfyUI") #"[^A-Za-z0-9._-]" "_")
          values (double-array (arr/->vec images))]
      (Files/createDirectories directory (make-array java.nio.file.attribute.FileAttribute 0))
      (mapv
       (fn [batch-index]
         (let [image (BufferedImage. width height BufferedImage/TYPE_INT_RGB)
               filename (format "%s_%05d.png" prefix batch-index)
               path (.resolve directory filename)]
           (dotimes [y height]
             (dotimes [x width]
               (let [base (+ (* batch-index height width channels)
                             (* y width channels) (* x channels))
                     byte (fn [channel]
                            (int (Math/round (* 255.0
                                               (max 0.0 (min 1.0
                                                             (aget values (+ base channel))))))))
                     rgb (bit-or (bit-shift-left (byte 0) 16)
                                 (bit-shift-left (byte 1) 8)
                                 (byte 2))]
                 (.setRGB image x y rgb))))
           (when-not (ImageIO/write image "png" (.toFile path))
             (throw (ex-info "no PNG ImageIO writer available" {})))
           {:filename filename :subfolder "" :type "output" :path (str path)}))
       (range batch)))))

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
  - `:vae-spec` decoder graph map, or resolver function
  - `:output-directory` destination for executable SaveImage PNG files
  - `:alphas-cumprod` schedule vector, or a resolver function

  The MODEL/CLIP/VAE maps share one lazy SafeTensorFile. The host owns its
  lifecycle and closes `:comfyui/checkpoint` when the workflow/model unloads."
  [{:keys [backend resolve-checkpoint model-spec vae-spec alphas-cumprod
           output-directory]
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
                decoder-spec (if (fn? vae-spec)
                               (vae-spec ckpt_name checkpoint) vae-spec)
                executable-model
                (cond-> model-component
                  spec (assoc :comfyui/model-spec spec
                              :comfyui/denoise
                              (diffusion-model/compile-denoiser
                               model-component backend spec))
                  alphas (assoc :comfyui/alphas-cumprod (vec alphas)))
                vae-component (component checkpoint :vae
                                         ["first_stage_model." "vae."])
                executable-vae
                (cond-> vae-component
                  decoder-spec
                  (assoc :comfyui/model-spec decoder-spec
                         :comfyui/decode
                         (diffusion-model/compile-decoder
                          vae-component backend decoder-spec)))]
            [executable-model
             (component checkpoint :clip
                        ["cond_stage_model." "text_encoder." "clip."])
             executable-vae]))}
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
   {:type "VAEDecode"
    :category "latent"
    :inputs {:samples {:type "LATENT"}
             :vae {:type "VAE"}}
    :outputs [{:name "IMAGE" :type "IMAGE"}]
    :fn (fn [{:keys [samples vae]}]
          (let [latent (if (and (map? samples) (contains? samples :samples))
                         (:samples samples) samples)
                decode (:comfyui/decode vae)]
            (when-not (and latent (fn? decode))
              (throw (ex-info "VAE lacks executable checkpoint decoder"
                              {:vae-keys (keys vae)})))
            (let [decoded (decode latent)
                  [batch channels height width :as shape] (:shape decoded)]
              (when-not (and (= 4 (count shape)) (= 3 channels))
                (throw (ex-info "VAE decoder must return NCHW RGB"
                                {:shape shape})))
              (let [nhwc (t/transpose decoded [0 2 3 1])
                    normalized
                    (arr/from-vec
                     (:backend nhwc)
                     (mapv (fn [value]
                             (max 0.0 (min 1.0 (* 0.5 (+ 1.0 value)))))
                           (arr/->vec nhwc))
                     [batch height width channels])]
                [normalized]))))}
   {:type "SaveImage"
    :category "image"
    :inputs {:images {:type "IMAGE"}
             :filename_prefix {:type "STRING" :default "ComfyUI"}}
    :outputs [{:name "UI" :type "UI"}]
    :fn (fn [{:keys [images filename_prefix]}]
          [{:images (save-png-batch images output-directory filename_prefix)}])}
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
          (when-not (and (contains? #{"ddim" "euler" "euler_ancestral"} sampler_name)
                         (= "normal" scheduler)
                         (= 1.0 (double denoise)))
            (throw (ex-info "runtime KSampler supports ddim/euler/euler_ancestral with normal/full-denoise"
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
                sampler-args
                {:sample sample
                 :alphas alphas
                 :timesteps (scheduler/select-timesteps (count alphas) steps)
                 :denoise-fn denoise-fn
                 :positive positive :negative negative :cfg cfg
                 :eta 0.0 :noise-fn noise-fn}
                result (case sampler_name
                         "ddim" (scheduler/ddim-sample sampler-args)
                         "euler" (scheduler/euler-sample sampler-args)
                         "euler_ancestral"
                         (scheduler/euler-ancestral-sample
                          (assoc sampler-args :eta 1.0)))]
            [(:sample result)]))}])
