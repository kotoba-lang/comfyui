(ns comfyui.png-deno
  "Minimal dependency-free RGB8 PNG encoder for Deno/browser hosts.")

(def ^:private signature
  (js/Uint8Array. #js [137 80 78 71 13 10 26 10]))

(defn- crc32 [bytes]
  (let [crc (reduce
             (fn [crc byte]
               (loop [value (bit-xor crc byte) bit 0]
                 (if (= bit 8)
                   value
                   (recur (if (odd? value)
                            (bit-xor (unsigned-bit-shift-right value 1) 0xedb88320)
                            (unsigned-bit-shift-right value 1))
                          (inc bit)))))
             -1 (js/Array.from bytes))]
    (unsigned-bit-shift-right (bit-xor crc -1) 0)))

(defn- concat-bytes [parts]
  (let [length (reduce + (map #(.-byteLength %) parts))
        output (js/Uint8Array. length)]
    (loop [remaining parts offset 0]
      (if-let [part (first remaining)]
        (do (.set output part offset)
            (recur (next remaining) (+ offset (.-byteLength part))))
        output))))

(defn- png-chunk [type data]
  (let [type-bytes (.encode (js/TextEncoder.) type)
        crc-input (concat-bytes [type-bytes data])
        output (js/Uint8Array. (+ 12 (.-byteLength data)))
        view (js/DataView. (.-buffer output))]
    (.setUint32 view 0 (.-byteLength data) false)
    (.set output type-bytes 4)
    (.set output data 8)
    (.setUint32 view (+ 8 (.-byteLength data)) (crc32 crc-input) false)
    output))

(defn- ihdr [width height]
  (let [data (js/Uint8Array. 13)
        view (js/DataView. (.-buffer data))]
    (.setUint32 view 0 width false)
    (.setUint32 view 4 height false)
    (aset data 8 8)  ; bit depth
    (aset data 9 2)  ; truecolor RGB
    data))

(defn- scanlines [values width height]
  (let [row-bytes (* width 3)
        output (js/Uint8Array. (* height (inc row-bytes)))]
    (dotimes [row height]
      (let [destination (+ (* row (inc row-bytes)) 1)
            source (* row row-bytes)]
        (dotimes [column row-bytes]
          (let [value (nth values (+ source column))
                byte (Math/round (* 255.0 (max 0.0 (min 1.0 value))))]
            (aset output (+ destination column) byte)))))
    output))

(defn encode-rgb
  "Return Promise<Uint8Array> for flat RGB values in [0,1]."
  [values width height]
  (when-not (and (pos-int? width) (pos-int? height)
                 (= (count values) (* width height 3)))
    (throw (ex-info "RGB data length does not match PNG dimensions"
                    {:count (count values) :width width :height height})))
  (let [raw (scanlines values width height)
        compressed-stream (-> (js/Blob. #js [raw])
                              (.stream)
                              (.pipeThrough (js/CompressionStream. "deflate")))]
    (-> (.arrayBuffer (js/Response. compressed-stream))
        (.then (fn [buffer]
                 (concat-bytes
                  [signature
                   (png-chunk "IHDR" (ihdr width height))
                   (png-chunk "IDAT" (js/Uint8Array. buffer))
                   (png-chunk "IEND" (js/Uint8Array. 0))]))))))

(defn dimensions [png]
  (when-not (and (>= (.-byteLength png) 33)
                 (every? true? (map = (js/Array.from signature)
                                    (take 8 (js/Array.from png)))))
    (throw (ex-info "invalid PNG signature/header" {})))
  (let [view (js/DataView. (.-buffer png) (.-byteOffset png) (.-byteLength png))]
    [(.getUint32 view 16 false) (.getUint32 view 20 false)]))
