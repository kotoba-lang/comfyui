(ns comfyui.diffusion.model
  "Checkpoint-backed diffusion graph execution.

  A graph is plain data and lowers directly to num tensor operations. Tensor
  names refer to lazy safetensors entries exposed by a MODEL component; each
  weight is decoded/uploaded once per compiled model and then cached. The op
  vocabulary covers the structural path of convolutional diffusion UNets:
  convolution/downsampling, GroupNorm, SiLU, saved residuals, nearest
  upsampling, skip concatenation/addition, timestep bias, and conditioning."
  (:require [num.array :as arr]
            [num.core :as nm]
            [num.tensor :as t]))

(def ^:private supported-ops
  #{:conv2d :groupnorm :silu :save :add-saved :cat-saved :upsample
    :add-conditioning :timestep-bias})

(defn- fail [message data]
  (throw (ex-info (str "comfyui.diffusion.model: " message) data)))

(defn- conditioning-tensor [conditioning]
  (cond
    (and (map? conditioning) (:tensor conditioning)) (:tensor conditioning)
    (and (map? conditioning) (:samples conditioning)) (:samples conditioning)
    (and (map? conditioning) (:shape conditioning) (:backend conditioning)) conditioning
    :else nil))

(defn compile-denoiser
  "Compile `spec` into `(fn [sample timestep conditioning] epsilon)`.

  Required spec key: `:layers`, a vector of op maps. `component` must expose
  `:comfyui/read-tensor`; weights are read lazily on `backend`. Output shape is
  required to equal sample shape, matching epsilon-prediction KSampler models."
  [component backend {:keys [layers] :as spec}]
  (when-not (and backend (fn? (:comfyui/read-tensor component))
                 (vector? layers) (seq layers))
    (fail "model spec requires backend, MODEL tensor reader, and non-empty :layers"
          {:spec spec :component-keys (keys component)}))
  (doseq [[index layer] (map-indexed vector layers)]
    (when-not (contains? supported-ops (:op layer))
      (fail "unsupported model op" {:index index :layer layer
                                     :supported supported-ops})))
  (let [cache (atom {})
        tensor! (fn [tensor-name]
                  (when tensor-name
                    (or (get @cache tensor-name)
                        (let [tensor ((:comfyui/read-tensor component) backend tensor-name)]
                          (swap! cache assoc tensor-name tensor)
                          tensor))))]
    (with-meta
      (fn [sample timestep conditioning]
        (let [initial {:value sample :saved {}}
              result
              (reduce
               (fn [{:keys [value saved] :as state} layer]
                 (case (:op layer)
                   :conv2d
                   (assoc state :value
                          (t/conv2d-nchw
                           value (tensor! (:weight layer)) (tensor! (:bias layer))
                           {:stride (or (:stride layer) 1)
                            :padding (or (:padding layer) 0)
                            :dilation (or (:dilation layer) 1)
                            :groups (or (:groups layer) 1)}))

                   :groupnorm
                   (assoc state :value
                          (t/group-norm-nchw value (:groups layer)
                                             (tensor! (:weight layer))
                                             (tensor! (:bias layer))
                                             (or (:eps layer) 1.0e-5)))

                   :silu (assoc state :value (t/silu value))

                   :save
                   (assoc-in state [:saved (:name layer)] value)

                   :add-saved
                   (let [other (get saved (:name layer))]
                     (when-not other
                       (fail "saved tensor not found" {:name (:name layer)}))
                     (when-not (= (:shape value) (:shape other))
                       (fail "residual add shape mismatch"
                             {:value (:shape value) :saved (:shape other)}))
                     (assoc state :value (nm/add value other)))

                   :cat-saved
                   (let [other (get saved (:name layer))]
                     (when-not other
                       (fail "saved tensor not found" {:name (:name layer)}))
                     (assoc state :value
                            (t/cat [value other] (or (:axis layer) 1))))

                   :upsample
                   (assoc state :value
                          (t/upsample-nearest2d value (or (:scale-factor layer) 2)))

                   :add-conditioning
                   (let [condition (conditioning-tensor conditioning)]
                     (when-not condition
                       (fail "conditioning op requires an NDArray or {:tensor NDArray}"
                             {:conditioning conditioning}))
                     (when-not (= (:shape value) (:shape condition))
                       (fail "conditioning shape mismatch"
                             {:value (:shape value) :conditioning (:shape condition)}))
                     (assoc state :value (nm/add value condition)))

                   :timestep-bias
                   (let [bias (* (double (or (:scale layer) 1.0)) (double timestep))
                         scalar (arr/from-vec (:backend value) [bias] [])]
                     (assoc state :value (t/add value scalar)))))
               initial layers)
              output (:value result)]
          (when-not (= (:shape sample) (:shape output))
            (fail "epsilon output shape must equal sample shape"
                  {:sample (:shape sample) :output (:shape output)}))
          output))
      {:comfyui/model-spec spec :comfyui/tensor-cache cache})))
