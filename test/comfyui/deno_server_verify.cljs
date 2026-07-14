(ns comfyui.deno-server-verify
  "Exercise the ComfyUI-compatible prompt API through an actual TCP listener."
  (:require [comfyui.node :as node]
            [comfyui.server :as server]
            [comfyui.std :as std]))

(defn- post-json [url body]
  (js/fetch url #js {:method "POST"
                      :headers #js {"content-type" "application/json"}
                      :body (js/JSON.stringify (clj->js body))}))

(defn- json [response] (.json response))

(defn -main [& _]
  (let [active (atom 0) peak (atom 0) completed (atom [])
        output-directory "/tmp/comfyui-deno-server-output"
        _ (js/Deno.mkdirSync output-directory #js {:recursive true})
        _ (js/Deno.writeFileSync (str output-directory "/result.png")
                                 (js/Uint8Array. #js [1 2 3 4]))
        slow {:type "SlowPreview" :category "output" :output-node? true
              :inputs {:value {:type "INT"}}
              :outputs [{:name "value" :type "INT"}]
              :fn (fn [{:keys [value]}]
                    (swap! active inc)
                    (swap! peak max @active)
                    (js/Promise.
                     (fn [resolve _]
                       (js/setTimeout
                        #(do (swap! active dec) (swap! completed conj value)
                             (resolve [value])) 40))))}
        registry (node/registry (conj std/all slow))
        service (server/server {:registry registry :cache nil
                                :output-directory output-directory})
        listener (server/serve! service {:hostname "127.0.0.1" :port 0})
        base (str "http://127.0.0.1:" (.-port (.-addr listener)))
        workflow (fn [value]
                   {"value" {:class_type "PrimitiveInt" :inputs {:value value}}
                    "out" {:class_type "SlowPreview"
                           :inputs {:value ["value" 0]}}})]
    (-> (js/Promise.all
         #js [(post-json (str base "/prompt") {:prompt (workflow 3)})
              (post-json (str base "/prompt") {:prompt (workflow 7)})])
        (.then (fn [responses]
                 (js/Promise.all #js [(json (aget responses 0))
                                      (json (aget responses 1))])))
        (.then
         (fn [submitted]
           (let [first-id (.-prompt_id (aget submitted 0))
                 second-id (.-prompt_id (aget submitted 1))]
             (-> (js/fetch (str base "/queue"))
                 (.then json)
                 (.then
                  (fn [queue]
                    (let [queued? (= 2 (+ (.-length (.-queue_running queue))
                                          (.-length (.-queue_pending queue))))]
                      (js/Promise.
                       (fn [resolve _]
                         (js/setTimeout #(resolve [first-id second-id queued?]) 120))))))))))
        (.then
         (fn [[first-id second-id queued?]]
           (-> (js/Promise.all
                #js [(js/fetch (str base "/history/" first-id))
                     (js/fetch (str base "/history/" second-id))
                     (js/fetch (str base "/object_info"))
                     (js/fetch (str base "/view?filename=result.png"))
                     (js/fetch (str base "/view?filename=..%2Fcomfyui-deno-outside.png"))
                     (post-json (str base "/prompt")
                                {:prompt {"bad" {:class_type "Missing"
                                                  :inputs {}}}})])
               (.then
                (fn [responses]
                  (-> (js/Promise.all
                       #js [(json (aget responses 0)) (json (aget responses 1))
                            (json (aget responses 2))
                            (-> (.arrayBuffer (aget responses 3))
                                (.then #(.-byteLength %)))
                            (json (aget responses 4))
                            (json (aget responses 5))])
                      (.then
                       (fn [bodies]
                         (let [first-history (aget bodies 0)
                               second-history (aget bodies 1)
                               info (aget bodies 2)
                               view-bytes (aget bodies 3)
                               view-error (aget bodies 4)
                               error (aget bodies 5)
                               first-entry (aget first-history first-id)
                               second-entry (aget second-history second-id)
                               ok? (and queued? (= 1 @peak) (= [3 7] @completed)
                                        (= 3 (.. first-entry -outputs -out -value))
                                        (= 7 (.. second-entry -outputs -out -value))
                                        (some? (aget info "SlowPreview"))
                                        (= 4 view-bytes)
                                        (some? (.-error view-error))
                                        (= 400 (.-status (aget responses 5)))
                                        (some? (.-error error)))]
                           (println "ComfyUI /prompt → serialized execution → /history:"
                                    (if ok? "passed" "failed"))
                           (println "/queue and /object_info contracts:"
                                    (if (and queued? (some? (aget info "SlowPreview")))
                                      "passed" "failed"))
                           (println "confined /view artifact serving:"
                                    (if (and (= 4 view-bytes)
                                             (some? (.-error view-error)))
                                      "passed" "failed"))
                           (.shutdown listener)
                           (when-not ok?
                             (throw (js/Error. "ComfyUI server verification failed"))))))))))))
        (.then #(js/Deno.exit 0))
        (.catch (fn [error]
                  (println "ERROR:" (or (.-stack error) (str error)))
                  (.shutdown listener)
                  (js/Deno.exit 1))))))

(set! *main-cli-fn* -main)
