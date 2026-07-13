(ns comfyui.nodes.diffusion-runtime-deno
  "Executable ComfyUI diffusion node pack for Deno WebGPU/Metal hosts."
  (:require [clojure.string :as str]
            [comfyui.clip.encoder :as clip]
            [comfyui.diffusion.model :as model]
            [comfyui.diffusion.scheduler :as scheduler]
            [comfyui.png-deno :as png]
            [comfyui.safetensors-deno :as safe]
            [num.array :as arr]
            [num.tensor :as t]))

(defn- component [checkpoint kind]
  {:comfyui/component kind
   :comfyui/checkpoint checkpoint
   :comfyui/read-tensor (fn [backend name]
                          (safe/read-tensor checkpoint backend name))})

(defn release-components! [components]
  (let [components (vec (remove nil? components))
        caches (->> components (mapcat vals) (filter fn?)
                    (keep #(-> % meta :comfyui/tensor-cache)) distinct vec)
        checkpoints (->> components (keep :comfyui/checkpoint) distinct vec)]
    (arr/release-all! (mapcat #(vals @%) caches))
    (doseq [cache caches] (reset! cache {}))
    (doseq [checkpoint checkpoints] (safe/close-file! checkpoint))
    nil))

(defn- safe-prefix [value]
  (let [value (str/replace (or value "ComfyUI") #"[^A-Za-z0-9._-]" "_")]
    (if (seq value) value "ComfyUI")))

(defn pack
  "Build real Deno nodes. `pipeline` contains paths plus inferred specs/alphas;
  `tokenize` and `noise-fn` are explicit host capabilities."
  [{:keys [backend pipeline tokenize noise-fn output-directory timesteps-fn]
    :or {timesteps-fn (fn [alphas steps]
                        (scheduler/select-timesteps (count alphas) steps))}}]
  [{:type "DiffusersPipelineLoader"
    :category "loaders"
    :inputs {:pipeline_name {:type "STRING"}}
    :outputs [{:name "MODEL" :type "MODEL"}
              {:name "CLIP" :type "CLIP"}
              {:name "VAE" :type "VAE"}]
    :fn
    (fn [_]
      (let [{:keys [unet text-encoder vae unet-spec clip-spec vae-spec alphas]}
            pipeline
            opened (atom [])
            open! (fn [path]
                    (let [checkpoint (safe/open-file path)]
                      (swap! opened conj checkpoint)
                      checkpoint))]
        (when-not (and backend unet-spec clip-spec vae-spec (seq alphas))
          (throw (ex-info "incomplete Deno Diffusers pipeline"
                          {:backend? (boolean backend) :unet? (boolean unet-spec)
                           :clip? (boolean clip-spec) :vae? (boolean vae-spec)
                           :alphas? (boolean (seq alphas))})))
        (try
          (let [unet-file (open! unet)
                text-file (open! text-encoder)
                vae-file (open! vae)
                model-component (component unet-file :model)
                clip-component (component text-file :clip)
                vae-component (component vae-file :vae)]
            [(assoc model-component :comfyui/model-spec unet-spec
                    :comfyui/alphas-cumprod alphas
                    :comfyui/denoise
                    (model/compile-denoiser model-component backend unet-spec))
             (assoc clip-component :comfyui/clip-spec clip-spec
                    :comfyui/encode
                    (clip/compile-encoder clip-component backend clip-spec))
             (assoc vae-component :comfyui/model-spec vae-spec
                    :comfyui/decode
                    (model/compile-decoder vae-component backend vae-spec))])
          (catch :default error
            (doseq [checkpoint @opened] (safe/close-file! checkpoint))
            (throw error)))))}
   {:type "CLIPTextEncode"
    :category "conditioning"
    :inputs {:clip {:type "CLIP"} :text {:type "STRING"}}
    :outputs [{:name "CONDITIONING" :type "CONDITIONING"}]
    :fn (fn [{:keys [clip text]}]
          (when-not (and (fn? tokenize) (fn? (:comfyui/encode clip)))
            (throw (ex-info "Deno CLIPTextEncode requires tokenizer and encoder" {})))
          [((:comfyui/encode clip) (assoc (tokenize text) :text text))])}
   {:type "EmptyLatentImage"
    :category "latent"
    :inputs {:width {:type "INT" :default 512}
             :height {:type "INT" :default 512}
             :batch_size {:type "INT" :default 1}}
    :outputs [{:name "LATENT" :type "LATENT"}]
    :fn (fn [{:keys [width height batch_size]}]
          (when-not (and (pos-int? width) (pos-int? height)
                         (zero? (mod width 8)) (zero? (mod height 8))
                         (pos-int? batch_size))
            (throw (ex-info "invalid latent dimensions"
                            {:width width :height height :batch batch_size})))
          [{:samples (arr/zeros backend
                                [batch_size 4 (quot height 8) (quot width 8)])}])}
   {:type "KSampler"
    :category "sampling"
    :inputs {:model {:type "MODEL"} :positive {:type "CONDITIONING"}
             :negative {:type "CONDITIONING"} :latent_image {:type "LATENT"}
             :seed {:type "INT" :default 0} :steps {:type "INT" :default 20}
             :cfg {:type "FLOAT" :default 7.0}
             :sampler_name {:type "STRING" :default "ddim"}
             :scheduler {:type "STRING" :default "normal"}
             :denoise {:type "FLOAT" :default 1.0}}
    :outputs [{:name "LATENT" :type "LATENT"}]
    :fn
    (fn [{:keys [model positive negative latent_image seed steps cfg
                 sampler_name scheduler denoise]}]
      (when-not (and (= "ddim" sampler_name) (= "normal" scheduler)
                     (= 1.0 (double denoise)) (fn? noise-fn))
        (throw (ex-info "Deno KSampler currently requires full-denoise DDIM/normal"
                        {:sampler sampler_name :scheduler scheduler :denoise denoise})))
      (let [empty (:samples latent_image)
            sample (noise-fn (:shape empty) seed backend)
            sampled (scheduler/ddim-sample
                     {:sample sample :alphas (:comfyui/alphas-cumprod model)
                      :timesteps (timesteps-fn (:comfyui/alphas-cumprod model) steps)
                      :denoise-fn (:comfyui/denoise model)
                      :positive positive :negative negative :cfg cfg
                      :final-alpha (first (:comfyui/alphas-cumprod model))})]
        (arr/release-all! [empty sample])
        [{:samples (:sample sampled) :history (:history sampled)}]))}
   {:type "VAEDecode"
    :category "latent"
    :inputs {:samples {:type "LATENT"} :vae {:type "VAE"}}
    :outputs [{:name "IMAGE" :type "IMAGE"}]
    :fn (fn [{:keys [samples vae]}]
          (let [decoded ((:comfyui/decode vae) (:samples samples))
                image (t/nchw-to-rgb-image decoded)]
            (arr/release! decoded)
            [image]))}
   {:type "SaveImage"
    :category "image" :output-node? true
    :inputs {:images {:type "IMAGE"}
             :filename_prefix {:type "STRING" :default "ComfyUI"}}
    :outputs [{:name "UI" :type "UI"}]
    :fn (fn [{:keys [images filename_prefix]}]
          (when-not (seq output-directory)
            (throw (ex-info "Deno SaveImage requires output-directory" {})))
          (let [[batch height width channels] (:shape images)]
            (when-not (= [1 height width 3] [batch height width channels])
              (throw (ex-info "Deno SaveImage currently requires one NHWC RGB image"
                              {:shape (:shape images)})))
            (-> (arr/->vec images)
                (.then #(png/encode-rgb % width height))
                (.then (fn [bytes]
                         (js/Deno.mkdirSync output-directory #js {:recursive true})
                         (let [filename (str (safe-prefix filename_prefix) "_00000.png")
                               path (str output-directory "/" filename)]
                           (js/Deno.writeFileSync path bytes)
                           [{:images [{:filename filename :subfolder ""
                                      :type "output" :path path
                                      :bytes (.-byteLength bytes)}]}]))))))}])
