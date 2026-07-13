(ns comfyui.clip.encoder
  "Checkpoint-backed OpenAI CLIP text transformer reference executor."
  (:require [num.array :as arr]
            [num.core :as nm]
            [num.tensor :as t]))

(defn- fail [message data]
  (throw (ex-info (str "comfyui.clip.encoder: " message) data)))

(defn- linear [x weight bias]
  (let [output (t/matmul x (t/transpose weight))]
    (if bias (t/add output bias) output)))

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
                      (if (< i hidden)
                        (recur (inc i) (+ sum (aget xs (+ base i)))) sum))
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

(defn- quick-gelu [x]
  (arr/from-vec (:backend x)
                (mapv (fn [value]
                        (* value (/ 1.0 (+ 1.0 (Math/exp (* -1.702 value))))))
                      (arr/->vec x))
                (:shape x)))

(defn- embeddings [backend input-ids token-weight position-weight]
  (let [[vocab hidden] (:shape token-weight)
        [positions position-hidden] (:shape position-weight)
        length (count input-ids)]
    (when-not (and (= hidden position-hidden) (<= length positions)
                   (every? #(and (integer? %) (<= 0 % (dec vocab))) input-ids))
      (fail "token IDs/embedding shapes are incompatible"
            {:tokens length :token-shape (:shape token-weight)
             :position-shape (:shape position-weight)}))
    (let [tokens (double-array (arr/->vec token-weight))
          position (double-array (arr/->vec position-weight))]
      (arr/from-vec
       backend
       (vec (for [index (range length) channel (range hidden)]
              (+ (aget tokens (+ (* (nth input-ids index) hidden) channel))
                 (aget position (+ (* index hidden) channel)))))
       [1 length hidden]))))

(defn- slice-first-axis [tensor start length]
  (let [[rows columns] (:shape tensor)]
    (when (> (+ start length) rows)
      (fail "projection slice exceeds tensor" {:shape (:shape tensor)
                                                :start start :length length}))
    (arr/from-vec (:backend tensor)
                  (subvec (vec (arr/->vec tensor)) (* start columns)
                          (* (+ start length) columns))
                  [length columns])))

(defn- slice-vector [tensor start length]
  (arr/from-vec (:backend tensor)
                (subvec (vec (arr/->vec tensor)) start (+ start length))
                [length]))

(defn- self-attention [x tensor! layer heads]
  (let [[batch length hidden] (:shape x)
        head-dim (quot hidden heads)
        fused? (:in-proj-weight layer)
        fused-weight (when fused? (tensor! (:in-proj-weight layer)))
        fused-bias (when fused? (tensor! (:in-proj-bias layer)))
        projection-index {"query" 0 "key" 1 "value" 2}
        project (fn [name]
                  (if fused?
                    (let [offset (* hidden (projection-index name))]
                      (linear x (slice-first-axis fused-weight offset hidden)
                              (slice-vector fused-bias offset hidden)))
                    (linear x (tensor! (get layer (keyword (str name "-weight"))))
                            (tensor! (get layer (keyword (str name "-bias")))))))
        split (fn [value]
                (t/transpose (t/reshape value [batch length heads head-dim])
                             [0 2 1 3]))
        q (split (project "query"))
        k (split (project "key"))
        v (split (project "value"))
        scores (nm/scal! (/ 1.0 (Math/sqrt head-dim))
                         (t/matmul q (t/transpose k [0 1 3 2])))
        causal (arr/from-vec (:backend x)
                             (vec (for [_batch (range batch) _head (range heads)
                                        row (range length) column (range length)]
                                    (if (<= column row) 0.0 -1.0e9)))
                             [batch heads length length])
        probabilities (t/softmax (t/add scores causal))
        attended (t/matmul probabilities v)
        merged (t/reshape (t/transpose attended [0 2 1 3])
                          [batch length hidden])]
    (linear merged (tensor! (:output-weight layer)) (tensor! (:output-bias layer)))))

(defn compile-encoder
  "Compile a CLIP transformer spec. Returns a function accepting a tokenizer
  result map and producing it with `:tensor [1,seq,hidden]` and `:pooled`."
  [component backend {:keys [token-embedding position-embedding layers
                             final-norm-weight final-norm-bias heads eps
                             return-penultimate? text-projection]
                      :or {eps 1.0e-5} :as spec}]
  (when-not (and backend (fn? (:comfyui/read-tensor component))
                 (seq layers) (pos-int? heads))
    (fail "encoder requires backend, tensor reader, layers, and heads" {:spec spec}))
  (let [cache (atom {})
        tensor! (fn [name]
                  (or (get @cache name)
                      (let [value ((:comfyui/read-tensor component) backend name)]
                        (swap! cache assoc name value) value)))]
    (with-meta
      (fn [{:keys [input-ids] :as tokenized}]
        (let [initial (embeddings backend input-ids
                                  (tensor! token-embedding) (tensor! position-embedding))
              block
              (fn [x layer]
                 (let [normalized (layer-norm x (tensor! (:norm1-weight layer))
                                              (tensor! (:norm1-bias layer)) eps)
                       attention (self-attention normalized tensor! layer heads)
                       residual (nm/add x attention)
                       normalized2 (layer-norm residual
                                               (tensor! (:norm2-weight layer))
                                               (tensor! (:norm2-bias layer)) eps)
                       mlp (linear (quick-gelu
                                    (linear normalized2
                                            (tensor! (:fc1-weight layer))
                                            (tensor! (:fc1-bias layer))))
                                   (tensor! (:fc2-weight layer))
                                   (tensor! (:fc2-bias layer)))]
                   (nm/add residual mlp)))
              states (vec (reductions block initial layers))
              hidden (if return-penultimate?
                       (nth states (dec (count layers)))
                       (peek states))
              output (layer-norm hidden (tensor! final-norm-weight)
                                 (tensor! final-norm-bias) eps)
              [_ length hidden-size] (:shape output)
              eos-index (max 0 (dec (or (first (keep-indexed
                                                (fn [index mask]
                                                  (when (zero? mask) index))
                                                (:attention-mask tokenized)))
                                        length)))
              values (vec (arr/->vec output))
              pooled-base (arr/from-vec backend
                                          (subvec values (* eos-index hidden-size)
                                                  (* (inc eos-index) hidden-size))
                                          [1 hidden-size])
              pooled (if text-projection
                       (t/matmul pooled-base (tensor! text-projection))
                       pooled-base)]
          (assoc tokenized :tensor output :pooled pooled)))
      {:comfyui/clip-spec spec :comfyui/tensor-cache cache})))

(defn compile-dual-encoder
  "Compile two CLIP encoders, concatenate token features, and retain the
  second encoder's pooled embedding for SDXL added conditioning."
  [component backend {:keys [encoders] :as spec}]
  (when-not (= 2 (count encoders))
    (fail "dual encoder requires exactly two encoder specs" {:spec spec}))
  (let [[encode-l encode-g] (mapv #(compile-encoder component backend %) encoders)]
    (with-meta
      (fn [tokenized]
        (let [left (encode-l tokenized)
              right (encode-g tokenized)
              left-tensor (:tensor left)
              right-tensor (:tensor right)]
          (when-not (= (subvec (:shape left-tensor) 0 2)
                       (subvec (:shape right-tensor) 0 2))
            (fail "dual encoder batch/token dimensions differ"
                  {:left (:shape left-tensor) :right (:shape right-tensor)}))
          (assoc tokenized
                 :tensor (t/cat [left-tensor right-tensor] 2)
                 :pooled (:pooled right)
                 :clip-l left
                 :clip-g right)))
      {:comfyui/clip-spec spec})))
