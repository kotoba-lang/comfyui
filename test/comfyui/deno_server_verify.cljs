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
        failing {:type "FailPreview" :category "output" :output-node? true
                 :inputs {} :outputs [{:name "value" :type "INT"}]
                 :fn (fn [_] (throw (js/Error. "intentional node failure")))}
        registry (node/registry (into std/all [slow failing]))
        service (server/server {:registry registry :cache nil
                                :output-directory output-directory})
        listener (server/serve! service {:hostname "127.0.0.1" :port 0})
        base (str "http://127.0.0.1:" (.-port (.-addr listener)))
        messages (atom [])
        websocket (js/WebSocket.
                   (str "ws://127.0.0.1:" (.-port (.-addr listener))
                        "/ws?clientId=deno-verifier"))
        ws-ready (js/Promise.
                  (fn [resolve reject]
                    (set! (.-onopen websocket) resolve)
                    (set! (.-onerror websocket)
                          #(reject (js/Error. "WebSocket connection failed")))))
        _ (set! (.-onmessage websocket)
                (fn [event]
                  (swap! messages conj
                         (js->clj (js/JSON.parse (.-data event))
                                  :keywordize-keys true))))
        workflow (fn [value]
                   {"value" {:class_type "PrimitiveInt" :inputs {:value value}}
                    "out" {:class_type "SlowPreview"
                           :inputs {:value ["value" 0]}}})
        failing-workflow {"fail" {:class_type "FailPreview" :inputs {}}}]
    (-> ws-ready
        (.then
         (fn [_]
           (js/Promise.all
            #js [(post-json (str base "/prompt") {:prompt (workflow 3)})
                 (post-json (str base "/prompt") {:prompt (workflow 7)})
                 (post-json (str base "/prompt") {:prompt failing-workflow})])))
        (.then (fn [responses]
                 (js/Promise.all #js [(json (aget responses 0))
                                      (json (aget responses 1))
                                      (json (aget responses 2))])))
        (.then
         (fn [submitted]
           (let [first-id (.-prompt_id (aget submitted 0))
                 second-id (.-prompt_id (aget submitted 1))
                 failed-id (.-prompt_id (aget submitted 2))]
             (-> (js/fetch (str base "/queue"))
                 (.then json)
                 (.then
                  (fn [queue]
                    (let [queued? (= 3 (+ (.-length (.-queue_running queue))
                                          (.-length (.-queue_pending queue))))]
                      (js/Promise.
                       (fn [resolve _]
                         (js/setTimeout
                          #(resolve [first-id second-id failed-id queued?]) 140))))))))))
        (.then
         (fn [[first-id second-id failed-id queued?]]
           (-> (js/Promise.all
                #js [(js/fetch (str base "/history/" first-id))
                     (js/fetch (str base "/history/" second-id))
                     (js/fetch (str base "/history/" failed-id))
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
                            (json (aget responses 3))
                            (-> (.arrayBuffer (aget responses 4))
                                (.then #(.-byteLength %)))
                            (json (aget responses 5))
                            (json (aget responses 6))])
                      (.then
                       (fn [bodies]
                         (let [first-history (aget bodies 0)
                               second-history (aget bodies 1)
                               failed-history (aget bodies 2)
                               info (aget bodies 3)
                               view-bytes (aget bodies 4)
                               view-error (aget bodies 5)
                               error (aget bodies 6)
                               first-entry (aget first-history first-id)
                               second-entry (aget second-history second-id)
                               failed-entry (aget failed-history failed-id)
                               events @messages
                               executing-nodes
                               (->> events
                                    (filter #(= "executing" (:type %)))
                                    (keep #(get-in % [:data :node]))
                                    vec)
                               started-ids
                               (->> events (filter #(= "execution_start" (:type %)))
                                    (map #(get-in % [:data :prompt_id])) vec)
                               succeeded-ids
                               (->> events (filter #(= "execution_success" (:type %)))
                                    (map #(get-in % [:data :prompt_id])) vec)
                               failed-events
                               (->> events (filter #(= "execution_error" (:type %)))
                                    (map :data) vec)
                               websocket-ok?
                               (and (= "deno-verifier"
                                       (get-in (first events) [:data :sid]))
                                    (= [first-id second-id failed-id] started-ids)
                                    (= ["value" "out" "value" "out" "fail"]
                                       executing-nodes)
                                    (= [first-id second-id] succeeded-ids)
                                    (= [{:prompt_id failed-id
                                         :node_id "fail"
                                         :exception_message "intentional node failure"}]
                                       failed-events)
                                    (some #(and (= "status" (:type %))
                                                (zero? (get-in % [:data :status
                                                                 :exec_info
                                                                 :queue_remaining])))
                                          events))
                               ok? (and queued? (= 1 @peak) (= [3 7] @completed)
                                        (= 3 (.. first-entry -outputs -out -value))
                                        (= 7 (.. second-entry -outputs -out -value))
                                        (= "error" (.. failed-entry -status -status_str))
                                        (= "fail"
                                           (.-node_id
                                            (aget (aget (.. failed-entry -status -messages)
                                                        0) 1)))
                                        (some? (aget info "SlowPreview"))
                                        (= 4 view-bytes)
                                        (some? (.-error view-error))
                                        (= 400 (.-status (aget responses 6)))
                                        (some? (.-error error))
                                        websocket-ok?)]
                           (when-not websocket-ok?
                             (println "WebSocket events:" (pr-str events)))
                           (println "ComfyUI /prompt → serialized execution → /history:"
                                    (if ok? "passed" "failed"))
                           (println "/queue and /object_info contracts:"
                                    (if (and queued? (some? (aget info "SlowPreview")))
                                      "passed" "failed"))
                           (println "confined /view artifact serving:"
                                    (if (and (= 4 view-bytes)
                                             (some? (.-error view-error)))
                                      "passed" "failed"))
                           (println "WebSocket status and execution events:"
                                    (if websocket-ok? "passed" "failed"))
                           (.close websocket)
                           (.shutdown listener)
                           (when-not ok?
                             (throw (js/Error. "ComfyUI server verification failed"))))))))))))
        (.then #(js/Deno.exit 0))
        (.catch (fn [error]
                  (println "ERROR:" (or (.-stack error) (str error)))
                  (.close websocket)
                  (.shutdown listener)
                  (js/Deno.exit 1))))))

(set! *main-cli-fn* -main)
