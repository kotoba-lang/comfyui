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
    :scale :pad-right-bottom :take-channels :add-conditioning :cross-attention :timestep-bias :timestep-embedding
    :timestep-vector :sdxl-label-embedding :add-embedding :resblock
    :spatial-transformer :vae-attention})

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

(defn- sinusoidal-vector
  ([backend values embedding-dim]
   (sinusoidal-vector backend values embedding-dim false 1.0))
  ([backend values embedding-dim flip-sin-to-cos?]
   (sinusoidal-vector backend values embedding-dim flip-sin-to-cos? 1.0))
  ([backend values embedding-dim flip-sin-to-cos? frequency-shift]
  (when-not (and (even? embedding-dim) (>= embedding-dim 2))
    (fail "sinusoidal embedding dimension must be positive and even"
          {:dimension embedding-dim}))
  (let [half (quot embedding-dim 2)
        denominator (max 1.0 (- half (double frequency-shift)))
        frequencies (mapv #(Math/exp (* -1.0 (Math/log 10000.0)
                                     (/ % denominator)))
                          (range half))]
    (arr/from-vec
     backend
     (vec (mapcat (fn [value]
                    (let [angles (mapv #(* (double value) %) frequencies)
                          sin (mapv #(Math/sin %) angles)
                          cos (mapv #(Math/cos %) angles)]
                      (if flip-sin-to-cos? (concat cos sin) (concat sin cos))))
                  values))
     [1 (* (count values) embedding-dim)]))))

(defn- timestep-vector [value timestep tensor! layer]
  (let [first-weight (tensor! (:first-weight layer))
        input-dim (second (:shape first-weight))
        input (sinusoidal-vector (:backend value) [timestep] input-dim
                                 (boolean (:flip-sin-to-cos? layer))
                                 (double (or (:frequency-shift layer) 1.0)))]
    (linear (t/silu (linear input first-weight (tensor! (:first-bias layer))))
            (tensor! (:second-weight layer)) (tensor! (:second-bias layer)))))

(defn- sdxl-label-vector [value conditioning tensor! layer]
  (let [pooled (:pooled conditioning)
        time-ids (:time-ids conditioning)
        first-weight (tensor! (:first-weight layer))
        expected-input (second (:shape first-weight))
        pooled-width (when pooled (last (:shape pooled)))
        time-width (when pooled-width (- expected-input pooled-width))]
    (when-not (and pooled (= 2 (count (:shape pooled))) (= 1 (first (:shape pooled)))
                   (= 6 (count time-ids)) (pos? time-width) (zero? (mod time-width 6)))
      (fail "SDXL label embedding requires pooled [1 P] and six time IDs"
            {:pooled (some-> pooled :shape) :time-ids time-ids
             :expected-input expected-input}))
    (let [time-embedding (sinusoidal-vector (:backend value) time-ids
                                             (quot time-width 6))
          input (t/cat [pooled time-embedding] 1)]
      (linear (t/silu (linear input first-weight (tensor! (:first-bias layer))))
              (tensor! (:second-weight layer)) (tensor! (:second-bias layer))))))

(defn- add-embedding [value embedding tensor! layer]
  (when-not embedding
    (fail "add-embedding requires a timestep/label embedding" {:layer layer}))
  (let [projected (linear (t/silu embedding) (tensor! (:weight layer))
                          (tensor! (:bias layer)))
        channels (second (:shape value))]
    (when-not (= [1 channels] (:shape projected))
      (fail "embedding projection must produce one value per channel"
            {:projected (:shape projected) :value (:shape value)}))
    (t/add value (t/reshape projected [1 channels 1 1]))))

(defn- take-channels [value channels]
  (let [[_batch input-channels _height _width :as shape] (:shape value)]
    (when-not (and (= 4 (count shape)) (pos-int? channels)
                   (<= channels input-channels))
      (fail "take-channels requires NCHW and a valid channel count"
            {:shape shape :channels channels}))
    (t/slice-axis value 1 0 channels)))

(defn- pad-right-bottom [value]
  (let [shape (:shape value)]
    (when-not (= 4 (count shape))
      (fail "pad-right-bottom requires NCHW" {:shape shape}))
    (t/pad-right-bottom-nchw value)))

(defn- resblock [value embedding tensor! layer]
  (let [groups (or (:groups layer) 32)
        residual value
        hidden (-> value
                   (t/group-norm-nchw groups
                                      (tensor! (:in-norm-weight layer))
                                      (tensor! (:in-norm-bias layer)) 1.0e-5)
                   t/silu
                   (t/conv2d-nchw (tensor! (:in-conv-weight layer))
                                  (tensor! (:in-conv-bias layer))
                                  {:padding 1}))
        hidden (if (:embedding-weight layer)
                 (add-embedding hidden embedding tensor!
                                {:weight (:embedding-weight layer)
                                 :bias (:embedding-bias layer)})
                 hidden)
        hidden (-> hidden
                   (t/group-norm-nchw groups
                                      (tensor! (:out-norm-weight layer))
                                      (tensor! (:out-norm-bias layer)) 1.0e-5)
                   t/silu
                   (t/conv2d-nchw (tensor! (:out-conv-weight layer))
                                  (tensor! (:out-conv-bias layer))
                                  {:padding 1}))
        residual (if (:skip-weight layer)
                   (t/conv2d-nchw residual (tensor! (:skip-weight layer))
                                  (tensor! (:skip-bias layer)))
                   residual)]
    (when-not (= (:shape hidden) (:shape residual))
      (fail "ResBlock hidden/skip shapes differ"
            {:hidden (:shape hidden) :skip (:shape residual)}))
    (nm/add residual hidden)))

(defn- layer-norm [x weight bias eps]
  (let [shape (:shape x) hidden (long (last shape))
        rows (quot (arr/nelems shape) hidden)
        xs (double-array (arr/->vec x))
        ws (double-array (arr/->vec weight))
        bs (double-array (arr/->vec bias))
        out (double-array (arr/nelems shape))]
    (dotimes [row rows]
      (let [base (* row hidden)
            mean (/ (loop [i 0 sum 0.0]
                      (if (< i hidden) (recur (inc i) (+ sum (aget xs (+ base i)))) sum))
                    hidden)
            variance (/ (loop [i 0 sum 0.0]
                          (if (< i hidden)
                            (let [delta (- (aget xs (+ base i)) mean)]
                              (recur (inc i) (+ sum (* delta delta)))) sum))
                        hidden)
            inv-std (/ 1.0 (Math/sqrt (+ variance eps)))]
        (dotimes [i hidden]
          (aset out (+ base i)
                (+ (* (- (aget xs (+ base i)) mean) inv-std (aget ws i))
                   (aget bs i))))))
    (arr/from-vec (:backend x) (vec out) shape)))

(defn- sequence-attention [query context tensor! spec heads]
  (let [[batch query-length hidden] (:shape query)
        context-length (second (:shape context))
        head-dim (quot hidden heads)
        projection (fn [source weight-key bias-key]
                     (linear source (tensor! (get spec weight-key))
                             (tensor! (get spec bias-key))))
        split (fn [value length]
                (t/transpose (t/reshape value [batch length heads head-dim])
                             [0 2 1 3]))
        q (split (projection query :query-weight :query-bias) query-length)
        k (split (projection context :key-weight :key-bias) context-length)
        v (split (projection context :value-weight :value-bias) context-length)
        scores (nm/scal! (/ 1.0 (Math/sqrt head-dim))
                         (t/matmul q (t/transpose k [0 1 3 2])))
        probabilities (t/softmax scores)
        attended (t/matmul probabilities v)
        merged (t/reshape (t/transpose attended [0 2 1 3])
                          [batch query-length hidden])]
    (linear merged (tensor! (:output-weight spec)) (tensor! (:output-bias spec)))))

(defn- gelu [value]
  (* 0.5 value
     (+ 1.0 (Math/tanh (* 0.7978845608
                     (+ value (* 0.044715 value value value)))))))

(defn- geglu [x tensor! spec]
  (let [projected (linear x (tensor! (:project-weight spec))
                          (tensor! (:project-bias spec)))
        shape (:shape projected)
        doubled (last shape)
        inner (quot doubled 2)
        rows (quot (arr/nelems shape) doubled)
        values (vec (arr/->vec projected))
        gated (vec
               (mapcat
                (fn [row]
                  (let [base (* row doubled)]
                    (mapv (fn [i]
                            (let [value (nth values (+ base i))
                                  gate (nth values (+ base inner i))
                                  activated (gelu gate)]
                              (* value activated)))
                          (range inner))))
                (range rows)))
        hidden (arr/from-vec (:backend x) gated (assoc shape (dec (count shape)) inner))]
    (linear hidden (tensor! (:output-weight spec)) (tensor! (:output-bias spec)))))

(defn- spatial-transformer [value conditioning tensor! layer]
  (let [[batch channels height width] (:shape value)
        context (conditioning-tensor conditioning)
        _ (when-not context (fail "SpatialTransformer requires conditioning" {}))
        residual value
        normalized (t/group-norm-nchw value (or (:groups layer) 32)
                                            (tensor! (:norm-weight layer))
                                            (tensor! (:norm-bias layer)) 1.0e-6)
        projected (if (:linear-projection? layer)
                    (linear (t/reshape (t/transpose normalized [0 2 3 1])
                                       [batch (* height width) channels])
                            (tensor! (:proj-in-weight layer))
                            (tensor! (:proj-in-bias layer)))
                    (let [nchw (t/conv2d-nchw normalized (tensor! (:proj-in-weight layer))
                                              (tensor! (:proj-in-bias layer)))]
                      (t/reshape (t/transpose nchw [0 2 3 1])
                                 [batch (* height width) channels])))
        transformed
        (reduce
         (fn [x block]
           (let [heads (:heads block)
                 self-input (layer-norm x (tensor! (:norm1-weight block))
                                        (tensor! (:norm1-bias block)) 1.0e-5)
                 self-output (sequence-attention self-input self-input tensor!
                                                 (:self-attention block) heads)
                 x (nm/add x self-output)
                 cross-input (layer-norm x (tensor! (:norm2-weight block))
                                         (tensor! (:norm2-bias block)) 1.0e-5)
                 cross-output (sequence-attention cross-input context tensor!
                                                  (:cross-attention block) heads)
                 x (nm/add x cross-output)
                 ff-input (layer-norm x (tensor! (:norm3-weight block))
                                      (tensor! (:norm3-bias block)) 1.0e-5)]
             (nm/add x (geglu ff-input tensor! (:feed-forward block)))))
         projected (:blocks layer))
        output (if (:linear-projection? layer)
                 (let [sequence (linear transformed (tensor! (:proj-out-weight layer))
                                        (tensor! (:proj-out-bias layer)))]
                   (t/transpose (t/reshape sequence [batch height width channels])
                                [0 3 1 2]))
                 (t/conv2d-nchw
                  (t/transpose (t/reshape transformed [batch height width channels])
                               [0 3 1 2])
                  (tensor! (:proj-out-weight layer)) (tensor! (:proj-out-bias layer))))]
    (nm/add residual output)))

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

(defn- vae-attention [value tensor! layer]
  (let [[batch channels height width :as shape] (:shape value)
        normalized (t/group-norm-nchw
                    value (or (:groups layer) 32)
                    (tensor! (:norm-weight layer))
                    (tensor! (:norm-bias layer)) (or (:eps layer) 1.0e-6))
        sequence-length (* height width)
        sequence (t/reshape (t/transpose normalized [0 2 3 1])
                            [batch sequence-length channels])
        projection (fn [key]
                     (linear sequence
                             (tensor! (get layer (keyword (str key "-weight"))))
                             (tensor! (get layer (keyword (str key "-bias"))))))
        query (projection "query")
        key (projection "key")
        value-projection (projection "value")
        scores (nm/scal! (/ 1.0 (Math/sqrt channels))
                         (t/matmul query (t/transpose key [0 2 1])))
        attended (t/matmul (t/softmax scores) value-projection)
        projected (linear attended (tensor! (:output-weight layer))
                          (tensor! (:output-bias layer)))
        output (t/transpose (t/reshape projected [batch height width channels])
                            [0 3 1 2])]
    (when-not (= shape (:shape output))
      (fail "VAE attention changed tensor shape"
            {:input shape :output (:shape output)}))
    (nm/add value output)))

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
        (let [initial {:value sample :saved {} :embedding nil}
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
                   (assoc state :value (t/scale value (:factor layer)))

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

                   :timestep-vector
                   (assoc state :embedding
                          (timestep-vector value timestep tensor! layer))

                   :sdxl-label-embedding
                   (let [label (sdxl-label-vector value conditioning tensor! layer)]
                     (assoc state :embedding
                            (if-let [embedding (:embedding state)]
                              (nm/add embedding label)
                              label)))

                   :add-embedding
                   (assoc state :value
                          (add-embedding value (:embedding state) tensor! layer))

                   :resblock
                   (assoc state :value
                          (resblock value (:embedding state) tensor! layer))

                   :spatial-transformer
                   (assoc state :value
                          (spatial-transformer value conditioning tensor! layer))

                   :vae-attention
                   (assoc state :value (vae-attention value tensor! layer))

                   :pad-right-bottom
                   (assoc state :value (pad-right-bottom value))

                   :take-channels
                   (assoc state :value (take-channels value (:channels layer)))

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

(defn compile-encoder
  "Compile a checkpoint-backed image-to-latent graph. Encoder specs retain
  decoder layers separately under `:layers` and expose `:encoder-layers`."
  [component backend spec]
  (let [graph (compile-graph component backend
                             (assoc spec :layers (:encoder-layers spec)) false)]
    (with-meta (fn [image] (graph image 0 nil)) (meta graph))))
