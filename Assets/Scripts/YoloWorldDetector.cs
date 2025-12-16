using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using TMPro;

public class YoloWorldDetector : MonoBehaviour
{
    [Header("Model Settings")]
    [Tooltip("Your YOLOv8n ONNX as a Sentis ModelAsset")]
    public ModelAsset modelAsset;

    [Tooltip("Model training size - MUST match your ONNX model (e.g. 640)")]
    public int inputSize = 640;   // keep 640 for your current model

    [Range(0.01f, 0.99f)]
    public float confidenceThreshold = 0.4f;

    [Range(0.1f, 0.9f)]
    public float iouThreshold = 0.5f;

    [Tooltip("Maximum labels to show at once")]
    public int maxDetections = 5;

    [Header("Performance")]
    [Tooltip("Run YOLO every N frames (1–3 recommended on HoloLens)")]
    public int runEveryNFrames = 2;

    [Tooltip("Reuse last detections when skipping frames")]
    public bool interpolateDetections = true;

    [Header("Simple Text HUD (world-space TMPs)")]
    public TextMeshPro statusText;
    public TextMeshPro fpsText;
    public TextMeshPro debugText;

    [Header("World-Space Label Settings")]
    public float labelDepth = 2.0f;
    public float labelViewportYOffset = 0.06f;
    public float textScale = 0.002f;

    // Your 8 classes
    private readonly string[] classNames = {
        "Tools", "Housing", "Big Screw", "Small Screw",
        "V-Lock Plate", "Plunger", "Halical Spring", "Cover"
    };

    // Sentis
    private Worker worker;
    private RenderTexture preprocessTexture;

    // Camera
    private WebCamTexture webCamTex;
    private bool cameraInitialized = false;
    private int cameraWidth = 896;
    private int cameraHeight = 504;

    // Labels
    private WorldLabel[] worldLabels;
    private Transform labelParent;

    // Stats
    private float inferenceMS;
    private float smoothFPS;
    private int frameCounter;
    private float totalFrameTime;

    // Detection caching
    private readonly List<Det> lastDetections = new List<Det>();

    private bool loggedOutputShape = false;

    // ---------- helper types ----------

    private class Det
    {
        public int cls;
        public float p;
        public Rect r;

        public Det(int cls, float p, Rect r)
        {
            this.cls = cls;
            this.p = p;
            this.r = r;
        }
    }

    private class WorldLabel
    {
        public GameObject root;
        public Transform t;
        public TextMeshPro text;
        public LineRenderer line;
        public GameObject bgQuad;
        private Renderer bgRenderer;

        public bool IsValid =>
            root != null && t != null && text != null && line != null;

        public WorldLabel(Transform parent, int layer, float textScale)
        {
            Create(parent, layer, textScale);
        }

        public void Create(Transform parent, int layer, float textScale)
        {
            if (parent == null) return;
            if (root != null) Object.Destroy(root);

            root = new GameObject("DetectionLabel");
            t = root.transform;
            t.SetParent(parent, false);
            t.localPosition = Vector3.zero;
            t.localRotation = Quaternion.identity;
            t.localScale = Vector3.one * Mathf.Max(textScale, 0.0001f);
            root.layer = layer;

            // Background quad
            bgQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            bgQuad.name = "Background";
            bgQuad.transform.SetParent(t, false);
            bgQuad.transform.localPosition = Vector3.zero;
            bgQuad.transform.localRotation = Quaternion.identity;
            bgQuad.transform.localScale = new Vector3(200f, 40f, 1f);
            bgQuad.layer = layer;

            bgRenderer = bgQuad.GetComponent<Renderer>();
            if (bgRenderer != null)
            {
                bgRenderer.material = new Material(Shader.Find("Unlit/Color"));
                bgRenderer.material.color = new Color(0, 0, 0, 0.7f);
            }

            var collider = bgQuad.GetComponent<Collider>();
            if (collider != null) Object.Destroy(collider);

            // Text
            var textObj = new GameObject("Text");
            textObj.transform.SetParent(t, false);
            textObj.transform.localPosition = new Vector3(0, 0, -0.1f);
            textObj.transform.localRotation = Quaternion.identity;
            textObj.transform.localScale = Vector3.one;
            textObj.layer = layer;

            text = textObj.AddComponent<TextMeshPro>();
            text.fontSize = 36;
            text.color = Color.white;
            text.alignment = TextAlignmentOptions.Center;
            text.overflowMode = TextOverflowModes.Overflow;

            var rectTransform = text.rectTransform;
            rectTransform.sizeDelta = new Vector2(200f, 40f);
            rectTransform.pivot = new Vector2(0.5f, 0.5f);

            text.outlineWidth = 0.2f;
            text.outlineColor = Color.black;

            // Line renderer
            var lineObj = new GameObject("Stem");
            lineObj.transform.SetParent(parent, false);
            lineObj.layer = layer;

            line = lineObj.AddComponent<LineRenderer>();
            line.useWorldSpace = true;
            line.positionCount = 2;
            line.startWidth = 0.005f;
            line.endWidth = 0.002f;
            line.numCapVertices = 2;
            line.numCornerVertices = 2;

            var shader = Shader.Find("Sprites/Default");
            if (shader == null) shader = Shader.Find("Unlit/Color");
            if (shader != null) line.material = new Material(shader);

            Clear();
        }

        public void Set(Vector3 worldLabelPos, Vector3 worldPointPos, string txt, Color col)
        {
            if (!IsValid) return;

            t.position = worldLabelPos;
            var cam = Camera.main;
            if (cam != null)
                t.rotation = Quaternion.LookRotation(t.position - cam.transform.position);

            if (text.text != txt) text.text = txt;

            if (bgRenderer != null)
            {
                Color bgCol = new Color(col.r * 0.3f, col.g * 0.3f, col.b * 0.3f, 0.8f);
                if (bgRenderer.material.color != bgCol)
                    bgRenderer.material.color = bgCol;
            }

            line.enabled = true;
            line.startColor = col;
            line.endColor = col;
            line.SetPosition(0, worldPointPos);
            line.SetPosition(1, worldLabelPos);
            root.SetActive(true);
        }

        public void Clear()
        {
            if (!IsValid) return;
            root.SetActive(false);
            line.enabled = false;
        }
    }

    // ---------- lifecycle ----------

    private void Start()
    {
        Application.targetFrameRate = 60;
        QualitySettings.vSyncCount = 0;

        labelDepth = Mathf.Max(labelDepth, 0.5f);

        var cam = Camera.main;
        if (cam != null)
        {
            labelParent = cam.transform;
        }
        else
        {
            labelParent = this.transform;
            Debug.LogError("[YOLO] No Camera.main found – labels will be attached to this transform.");
        }

        StartCoroutine(Init());
    }

    private IEnumerator Init()
    {
        UpdateStatus("Loading model...");

        if (modelAsset == null)
        {
            UpdateStatus("ERROR: modelAsset not assigned");
            yield break;
        }

        if (worker == null)
        {
            var runtimeModel = ModelLoader.Load(modelAsset);
            worker = new Worker(runtimeModel, BackendType.CPU);
        }

        // RT MUST match model input (e.g. 640x640)
        int modelSize = inputSize;
        if (preprocessTexture == null)
        {
            preprocessTexture = new RenderTexture(modelSize, modelSize, 0, RenderTextureFormat.ARGB32);
            preprocessTexture.Create();
        }

        if (worldLabels == null)
        {
            int camLayer = Camera.main != null ? Camera.main.gameObject.layer : 0;
            worldLabels = new WorldLabel[maxDetections];
            for (int i = 0; i < maxDetections; i++)
                worldLabels[i] = new WorldLabel(labelParent, camLayer, textScale);
        }

        yield return StartCameraCoroutine();

        if (cameraInitialized)
        {
            UpdateStatus("Camera READY - Running YOLO");
            frameCounter = 0;
            loggedOutputShape = false;
            StartCoroutine(DetectLoop());
        }
        else
        {
            UpdateStatus("ERROR: Camera failed to initialize");
        }
    }

    private IEnumerator StartCameraCoroutine()
    {
        UpdateStatus("Starting camera...");

        if (webCamTex != null)
        {
            webCamTex.Stop();
            Destroy(webCamTex);
            webCamTex = null;
        }

        var devices = WebCamTexture.devices;
        if (devices == null || devices.Length == 0)
        {
            UpdateStatus("No camera devices found");
            if (debugText != null) debugText.text = "No camera devices found";
            cameraInitialized = false;
            yield break;
        }

        // Prefer back camera (QC Back Camera) when present
        WebCamDevice chosen = devices[0];
        for (int i = 0; i < devices.Length; i++)
        {
            var d = devices[i];
            if (!d.isFrontFacing && d.name.ToLower().Contains("back"))
            {
                chosen = d;
                break;
            }
        }

        webCamTex = new WebCamTexture(chosen.name, cameraWidth, cameraHeight, 30);
        webCamTex.wrapMode = TextureWrapMode.Clamp;
        webCamTex.filterMode = FilterMode.Bilinear;
        webCamTex.Play();

        // wait until camera reports a valid size
        float timeout = 5f;
        float t = 0f;
        while (webCamTex.width <= 16 && t < timeout)
        {
            t += Time.deltaTime;
            yield return null;
        }

        if (webCamTex.width <= 16)
        {
            UpdateStatus("Camera start timeout");
            cameraInitialized = false;
            yield break;
        }

        cameraWidth = webCamTex.width;
        cameraHeight = webCamTex.height;

        if (debugText != null)
        {
            debugText.text =
                $"Camera: {chosen.name}\nfront={chosen.isFrontFacing}\n" +
                $"Resolution: {cameraWidth}x{cameraHeight}";
        }

        UpdateStatus($"Camera started: {chosen.name} {cameraWidth}x{cameraHeight}");
        cameraInitialized = true;
    }

    private IEnumerator DetectLoop()
    {
        int modelInputSize = inputSize;

        while (true)
        {
            frameCounter++;
            float frameStart = Time.realtimeSinceStartup;

            bool shouldRunInference = frameCounter % Mathf.Max(1, runEveryNFrames) == 0;

            if (cameraInitialized && webCamTex != null && webCamTex.didUpdateThisFrame && shouldRunInference)
            {
                float inferenceStart = Time.realtimeSinceStartup;

                try
                {
                    // copy camera → RT
                    Graphics.Blit(webCamTex, preprocessTexture);

                    using (var input = new Tensor<float>(new TensorShape(1, 3, modelInputSize, modelInputSize)))
                    {
                        var tt = new TextureTransform().SetDimensions(modelInputSize, modelInputSize);
                        TextureConverter.ToTensor(preprocessTexture, input, tt);

                        worker.Schedule(input);

                        using (var gpuOut = worker.PeekOutput() as Tensor<float>)
                        using (var cpuOut = gpuOut.ReadbackAndClone() as Tensor<float>)
                        {
                            if (!loggedOutputShape)
                            {
                                loggedOutputShape = true;
                                string shapeStr = cpuOut.shape.ToString();
                                Debug.Log("[YOLO] Output shape: " + shapeStr);
                                if (debugText != null)
                                    debugText.text = "Out shape: " + shapeStr;
                            }

                            var dets = ParseYOLO(cpuOut, modelInputSize);
                            var finalDets = NMS(dets);

                            lastDetections.Clear();
                            for (int i = 0; i < finalDets.Count; i++)
                                lastDetections.Add(finalDets[i]);

                            UpdateWorldLabels(finalDets);

                            if (debugText != null && finalDets.Count > 0)
                            {
                                float maxScore = 0f;
                                for (int i = 0; i < finalDets.Count; i++)
                                    if (finalDets[i].p > maxScore) maxScore = finalDets[i].p;

                                debugText.text =
                                    $"Camera: {cameraWidth}x{cameraHeight}\n" +
                                    $"Det: {finalDets.Count} | Max:{maxScore:F2}";
                            }
                            else if (debugText != null && finalDets.Count == 0)
                            {
                                debugText.text =
                                    $"Camera: {cameraWidth}x{cameraHeight}\nNo detections";
                            }
                        }
                    }

                    inferenceMS = (Time.realtimeSinceStartup - inferenceStart) * 1000f;
                }
                catch (System.SystemException ex)
                {
                    // Sentis / tensor / memory / index errors land here
                    string msg = "[YOLO ERROR] " + ex.GetType().Name + ": " + ex.Message;
                    Debug.LogError(msg);
                    if (debugText != null)
                    {
                        debugText.color = Color.red;
                        debugText.text = msg;
                    }
                    UpdateStatus("ERROR in DetectLoop (see debug text)");
                    // exit loop so we don't keep spamming errors
                    yield break;
                }
            }
            else if (interpolateDetections && lastDetections.Count > 0)
            {
                UpdateWorldLabels(lastDetections);
            }

            totalFrameTime = (Time.realtimeSinceStartup - frameStart) * 1000f;
            smoothFPS = Mathf.Lerp(smoothFPS, 1f / Mathf.Max(Time.deltaTime, 0.0001f), 0.1f);

            if (fpsText != null)
            {
                fpsText.text = $"FPS:{smoothFPS:F1}\n" +
                               $"Inf:{inferenceMS:F0}ms\n" +
                               $"Frame:{totalFrameTime:F0}ms";
            }

            yield return null;
        }
    }

    private void OnDestroy()
    {
        CleanupResources();
    }

    private void CleanupResources()
    {
        StopAllCoroutines();

        if (webCamTex != null)
        {
            webCamTex.Stop();
            Destroy(webCamTex);
            webCamTex = null;
        }

        if (preprocessTexture != null)
        {
            preprocessTexture.Release();
            Destroy(preprocessTexture);
            preprocessTexture = null;
        }

        if (worker != null)
        {
            worker.Dispose();
            worker = null;
        }

        lastDetections.Clear();
        cameraInitialized = false;
    }

    // ---------- YOLO parsing & NMS ----------

    private List<Det> ParseYOLO(Tensor<float> o, int actualSize)
    {
        var dets = new List<Det>(maxDetections * 2);

        int d1 = o.shape[1];
        int d2 = o.shape[2];
        bool rowsAreDetections = d1 > d2;
        int N = rowsAreDetections ? d1 : d2;

        for (int i = 0; i < N; i++)
        {
            float cx = rowsAreDetections ? o[0, i, 0] : o[0, 0, i];
            float cy = rowsAreDetections ? o[0, i, 1] : o[0, 1, i];
            float w = rowsAreDetections ? o[0, i, 2] : o[0, 2, i];
            float h = rowsAreDetections ? o[0, i, 3] : o[0, 3, i];

            cx /= actualSize;
            cy /= actualSize;
            w /= actualSize;
            h /= actualSize;

            if (w <= 0f || h <= 0f) continue;

            float bestScore = 0f;
            int bestClass = -1;

            for (int c = 0; c < classNames.Length; c++)
            {
                float p = rowsAreDetections ? o[0, i, 4 + c] : o[0, 4 + c, i];
                if (p > bestScore)
                {
                    bestScore = p;
                    bestClass = c;
                }
            }

            if (bestScore > confidenceThreshold && bestClass >= 0)
            {
                var r = new Rect(cx - w / 2f, cy - h / 2f, w, h);
                dets.Add(new Det(bestClass, bestScore, r));

                if (dets.Count > maxDetections * 3)
                    break;
            }
        }

        return dets;
    }

    private List<Det> NMS(List<Det> dets)
    {
        var result = new List<Det>(maxDetections);
        int n = dets.Count;
        if (n == 0) return result;

        bool[] removed = new bool[n];

        for (int k = 0; k < maxDetections; k++)
        {
            int bestIndex = -1;
            float bestScore = 0f;

            for (int i = 0; i < n; i++)
            {
                if (removed[i]) continue;
                if (dets[i].p > bestScore)
                {
                    bestScore = dets[i].p;
                    bestIndex = i;
                }
            }

            if (bestIndex == -1) break;

            var best = dets[bestIndex];
            result.Add(best);
            removed[bestIndex] = true;

            for (int i = 0; i < n; i++)
            {
                if (removed[i]) continue;
                if (dets[i].cls == best.cls && IoU(dets[i].r, best.r) > iouThreshold)
                    removed[i] = true;
            }
        }

        return result;
    }

    private float IoU(Rect a, Rect b)
    {
        float x1 = Mathf.Max(a.x, b.x);
        float y1 = Mathf.Max(a.y, b.y);
        float x2 = Mathf.Min(a.xMax, b.xMax);
        float y2 = Mathf.Min(a.yMax, b.yMax);

        float inter = Mathf.Max(0f, x2 - x1) * Mathf.Max(0f, y2 - y1);
        float union = a.width * a.height + b.width * b.height - inter;
        return (union > 0f) ? inter / union : 0f;
    }

    // ---------- label update (THIS is UpdateWorldLabels) ----------

    private void UpdateWorldLabels(List<Det> dets)
    {
        if (worldLabels == null || labelParent == null) return;
        var cam = Camera.main;
        if (cam == null) return;

        int camLayer = cam.gameObject.layer;

        // Clear all labels first
        for (int i = 0; i < worldLabels.Length; i++)
        {
            if (worldLabels[i] == null || !worldLabels[i].IsValid)
                worldLabels[i] = new WorldLabel(labelParent, camLayer, textScale);
            else
                worldLabels[i].Clear();
        }

        int count = dets.Count;
        if (count > worldLabels.Length) count = worldLabels.Length;

        for (int i = 0; i < count; i++)
        {
            var d = dets[i];
            var label = worldLabels[i];

            float cx = d.r.x + d.r.width * 0.5f;
            float cy = d.r.y + d.r.height * 0.5f;

            float vxCenter = cx;
            float vyCenter = 1f - cy;

            float vxLabel = vxCenter;
            float vyLabel = vyCenter + labelViewportYOffset;

            vxCenter = Mathf.Clamp01(vxCenter);
            vyCenter = Mathf.Clamp01(vyCenter);
            vxLabel = Mathf.Clamp01(vxLabel);
            vyLabel = Mathf.Clamp01(vyLabel);

            Vector3 worldPointPos = cam.ViewportToWorldPoint(
                new Vector3(vxCenter, vyCenter, labelDepth));
            Vector3 worldLabelPos = cam.ViewportToWorldPoint(
                new Vector3(vxLabel, vyLabel, labelDepth));

            string txt = $"{classNames[d.cls]} {(d.p * 100f):F0}%";
            Color col = Color.HSVToRGB((float)d.cls / classNames.Length, 0.9f, 1f);

            label.Set(worldLabelPos, worldPointPos, txt, col);
        }
    }

    // ---------- MRTK Restart callback ----------

    public void OnRestartButtonClicked()
    {
        Debug.Log("[YOLO] MRTK Restart pressed");
        UpdateStatus("Restarting...");

        CleanupResources();

        if (worldLabels != null)
        {
            for (int i = 0; i < worldLabels.Length; i++)
                if (worldLabels[i] != null && worldLabels[i].IsValid)
                    worldLabels[i].Clear();
        }

        frameCounter = 0;
        lastDetections.Clear();
        StartCoroutine(Init());
    }

    private void UpdateStatus(string s)
    {
        if (statusText != null) statusText.text = s;
        Debug.Log("[YOLO] " + s);
    }
}
