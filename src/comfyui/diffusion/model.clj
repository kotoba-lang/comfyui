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
    :scale :add-conditioning :cross-attention :timestep-bias :timestep-embedding})

(defn- fail [message data]
  (throw (ex-info (str "comfyui.diffusion.model: " message) data)))

(defn- conditioning-tensor [conditioning]
  (cond
    (and (map? conditioning) (:tensor conditioning)) (:tensor conditioning)
    (and (map? conditioning) (:samples conditioning)) (:samples conditioning)
    (and (map? conditioning) (:shape conditioning) (:backend conditioning)) conditioning
    :else nil))

(defn- linear [x weight bias]
  (let [output (t/matmul x (t/transpose weight))]
    (if bias (t/add output bias) output)))

(defn- cross-attention
  [value condition tensor! layer]
  (let [[batch channels height width :as input-shape] (:shape value)
        condition (conditioning-tensor condition)
        _ (when-not (and (= 4 (count input-shape)) condition
                         (= 3 (count (:shape condition)))
                         (= batch (first (:shape condition))))
            (fail "cross-attention requires NCHW and [N tokens context] conditioning"
                  {:value input-shape :conditioning (:shape condition)}))
        heads (long (:heads layer))
        _ (when-not (and (pos? heads) (zero? (mod channels heads)))
            (fail "attention heads must divide channels"
                  {:channels channels :heads heads}))
        sequence (* height width)
        head-dim (quot channels heads)
        x-sequence (t/reshape (t/transpose value [0 2 3 1])
                              [batch sequence channels])
        projection (fn [prefix source]
                     (linear source (tensor! (get layer (keyword (str prefix "-weight"))))
                             (tensor! (get layer (keyword (str prefix "-bias"))))))
        q (projection "query" x-sequence)
        k (projection "key" condition)
        v (projection "value" condition)
        tokens (second (:shape condition))
        split (fn [tensor length]
                (t/transpose (t/reshape tensor [batch length heads head-dim])
                             [0 2 1 3]))
        qh (split q sequence) kh (split k tokens) vh (split v tokens)
        scores (t/matmul qh (t/transpose kh [0 1 3 2]))
        scaled (nm/scal! (/ 1.0 (Math/sqrt head-dim)) scores)
        probabilities (t/softmax scaled)
        attended (t/matmul probabilities vh)
        merged (t/reshape (t/transpose attended [0 2 1 3])
                          [batch sequence channels])
        projected (linear merged (tensor! (:output-weight layer))
                          (tensor! (:output-bias layer)))
        nchw (t/transpose (t/reshape projected [batch height width channels])
                          [0 3 1 2])]
    (if (false? (:residual layer)) nchw (nm/add value nchw))))

(defn- timestep-embedding [value timestep tensor! layer]
  (let [first-weight (tensor! (:first-weight layer))
        first-bias (tensor! (:first-bias layer))
        second-weight (tensor! (:second-weight layer))
        second-bias (tensor! (:second-bias layer))
        embedding-dim (long (second (:shape first-weight)))
        half (quot embedding-dim 2)
        _ (when-not (and (even? embedding-dim) (> half 1))
            (fail "timestep embedding input dimension must be even and >= 4"
                  {:dimension embedding-dim}))
        frequencies (mapv #(Math/exp (* -1.0 (Math/log 10000.0)
                                           (/ % (dec half))))
                          (range half))
        angles (mapv #(* (double timestep) %) frequencies)
        embedding (arr/from-vec (:backend value)
                                (into (mapv #(Math/sin %) angles)
                                      (mapv #(Math/cos %) angles))
                                [1 embedding-dim])
        hidden (t/silu (linear embedding first-weight first-bias))
        projected (linear hidden second-weight second-bias)
        channels (second (:shape value))]
    (when-not (= [1 channels] (:shape projected))
      (fail "timestep projection must produce [1 C]"
            {:projected (:shape projected) :channels channels}))
    (t/add value (t/reshape projected [1 channels 1 1]))))

(defn- compile-graph
  [component backend {:keys [layers] :as spec} same-shape?]
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

                   :scale
                   (assoc state :value
                          (nm/scal! (double (:factor layer))
                                    (arr/from-vec (:backend value)
                                                  (arr/->vec value) (:shape value))))

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

                   :cross-attention
                   (assoc state :value
                          (cross-attention value conditioning tensor! layer))

                   :timestep-embedding
                   (assoc state :value
                          (timestep-embedding value timestep tensor! layer))

                   :timestep-bias
                   (let [bias (* (double (or (:scale layer) 1.0)) (double timestep))
                         scalar (arr/from-vec (:backend value) [bias] [])]
                     (assoc state :value (t/add value scalar)))))
               initial layers)
              output (:value result)]
          (when (and same-shape? (not= (:shape sample) (:shape output)))
            (fail "epsilon output shape must equal sample shape"
                  {:sample (:shape sample) :output (:shape output)}))
          output))
      {:comfyui/model-spec spec :comfyui/tensor-cache cache})))

(defn compile-denoiser
  "Compile `spec` into `(fn [sample timestep conditioning] epsilon)`.

  Required spec key: `:layers`, a vector of op maps. `component` must expose
  `:comfyui/read-tensor`; weights are read lazily on `backend`. Output shape is
  required to equal sample shape, matching epsilon-prediction KSampler models."
  [component backend spec]
  (compile-graph component backend spec true))

(defn compile-decoder
  "Compile a checkpoint-backed graph into `(fn [latent] image)`. Unlike a
  denoiser, a decoder may change channel and spatial dimensions."
  [component backend spec]
  (let [graph (compile-graph component backend spec false)]
    (with-meta (fn [latent] (graph latent 0 nil)) (meta graph))))
