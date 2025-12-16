using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Sentis;
using TMPro;
using UnityEngine.UI;

public class YoloWorldDetector : MonoBehaviour
{
    [Header("Model Settings")]
    public ModelAsset modelAsset;
    public int inputSize = 640;

    [Range(0.01f, 0.99f)]
    public float confidenceThreshold = 0.4f;

    [Range(0.1f, 0.9f)]
    public float iouThreshold = 0.5f;

    [Tooltip("Maximum labels to show at once")]
    public int maxDetections = 5;

    [Header("Performance Optimization")]
    [Tooltip("Run YOLO every N frames (2-3 recommended for HoloLens)")]
    public int runEveryNFrames = 2;

    [Tooltip("Lower resolution = faster inference (uses 320x320 instead of 640x640)")]
    public bool useHalfResolution = false;

    [Tooltip("Reuse last detections when skipping frames")]
    public bool interpolateDetections = true;

    [Header("HoloLens Camera Settings")]
    [Tooltip("Max seconds to wait for camera before giving up")]
    public float cameraTimeout = 10f;

    [Header("Text / Debug (optional)")]
    public TextMeshProUGUI statusText;
    public TextMeshProUGUI fpsText;
    public TextMeshProUGUI debugText;

    [Header("UI Controls (optional sliders)")]
    public Slider confSlider;
    public Slider iouSlider;

    [Header("World-Space Label Settings")]
    [Tooltip("Distance in meters from camera for labels & stems")]
    public float labelDepth = 2.0f;

    [Tooltip("Vertical offset (viewport units) to place label above box centre")]
    public float labelViewportYOffset = 0.06f;

    [Tooltip("Local scale for label root (0.001 to match your canvas)")]
    public float textScale = 0.001f;

    // Custom classes for your model
    private readonly string[] classNames = {
        "Tools", "Housing", "Big Screw", "Small Screw",
        "V-Lock Plate", "Plunger", "Halical Spring", "Cover"
    };

    // Sentis
    private Worker worker;
    private WebCamTexture webCam;
    private RenderTexture preprocessTexture;

    // Labels
    private WorldLabel[] worldLabels;
    private Transform labelParent; // camera transform

    // Stats
    private float inferenceMS;
    private float smoothFPS;
    private int frameCounter;
    private float totalFrameTime;

    // Detection caching (for interpolation on skipped frames)
    private List<Det> lastDetections = new List<Det>();
    private float lastInferenceTime;

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

        public Det Clone()
        {
            return new Det(cls, p, r);
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

            if (root != null)
                Object.Destroy(root);

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
            bgQuad.transform.localScale = new Vector3(200, 40, 1);
            bgQuad.layer = layer;

            bgRenderer = bgQuad.GetComponent<Renderer>();
            if (bgRenderer != null)
            {
                bgRenderer.material = new Material(Shader.Find("Unlit/Color"));
                bgRenderer.material.color = new Color(0, 0, 0, 0.7f);
            }

            var collider = bgQuad.GetComponent<Collider>();
            if (collider != null)
                Object.Destroy(collider);

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
            text.textWrappingMode = TextWrappingModes.NoWrap;
            text.overflowMode = TextOverflowModes.Overflow;

            var rectTransform = text.rectTransform;
            rectTransform.sizeDelta = new Vector2(200, 40);
            rectTransform.pivot = new Vector2(0.5f, 0.5f);

            text.outlineWidth = 0.2f;
            text.outlineColor = Color.black;

            // Line renderer (stem)
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
            if (shader != null)
                line.material = new Material(shader);

            Clear();
        }

        public void Set(Vector3 worldLabelPos, Vector3 worldPointPos,
                        string txt, Color col)
        {
            if (!IsValid) return;

            t.position = worldLabelPos;

            var cam = Camera.main;
            if (cam != null)
            {
                t.rotation = Quaternion.LookRotation(t.position - cam.transform.position);
            }

            if (text.text != txt)
                text.text = txt;

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
            Debug.Log($"[YOLO] Camera layer={cam.gameObject.layer}, mask={cam.cullingMask}");
        }
        else
        {
            Debug.LogError("[YOLO] No Camera.main found!");
            labelParent = this.transform;
        }

        if (confSlider != null)
        {
            confSlider.minValue = 0.01f;
            confSlider.maxValue = 0.99f;
            confSlider.value = confidenceThreshold;
        }
        if (iouSlider != null)
        {
            iouSlider.minValue = 0.1f;
            iouSlider.maxValue = 0.9f;
            iouSlider.value = iouThreshold;
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

        var runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, BackendType.CPU);   // CPU for HL2

        UpdateStatus("Starting camera...");

        var cams = WebCamTexture.devices;
        if (cams.Length == 0)
        {
            UpdateStatus("ERROR: No camera devices found");
            Debug.LogError("[YOLO] No WebCam devices. Check Capabilities (WebCam) in Player Settings.");
            yield break;
        }

        // Pick world-facing camera
        WebCamDevice selectedCam = cams[0];
        foreach (var c in cams)
        {
            if (!c.isFrontFacing || c.name.ToLower().Contains("rgb"))
            {
                selectedCam = c;
                break;
            }
        }

        Debug.Log($"[YOLO] Selected camera: {selectedCam.name}");

        webCam = new WebCamTexture(selectedCam.name);
        webCam.Play();

        // Wait until camera reports a sane resolution
        float start = Time.realtimeSinceStartup;
        while ((webCam.width < 64 || webCam.height < 64) &&
               Time.realtimeSinceStartup - start < cameraTimeout)
        {
            UpdateStatus($"Waiting for camera... {webCam.width}x{webCam.height}");
            yield return null;
        }

        if (webCam.width < 64 || webCam.height < 64)
        {
            UpdateStatus("ERROR: Camera did not start");
            Debug.LogError($"[YOLO] Camera failed to start. width={webCam.width}, height={webCam.height}");
            yield break;
        }

        Debug.Log($"[YOLO] Camera ready: {webCam.width}x{webCam.height}");

        int modelSize = useHalfResolution ? 320 : inputSize;
        preprocessTexture = new RenderTexture(modelSize, modelSize, 0, RenderTextureFormat.ARGB32);
        preprocessTexture.Create();

        Debug.Log($"[YOLO] RenderTexture: {modelSize}x{modelSize}");

        int camLayer = Camera.main != null ? Camera.main.gameObject.layer : 0;
        worldLabels = new WorldLabel[maxDetections];
        for (int i = 0; i < maxDetections; i++)
            worldLabels[i] = new WorldLabel(labelParent, camLayer, textScale);

        UpdateStatus("Ready - Running YOLO");
        StartCoroutine(DetectLoop());
    }

    private IEnumerator DetectLoop()
    {
        frameCounter = 0;

        while (true)
        {
            frameCounter++;
            float frameStart = Time.realtimeSinceStartup;

            bool shouldRunInference = frameCounter % Mathf.Max(1, runEveryNFrames) == 0;

            if (webCam != null && webCam.isPlaying && shouldRunInference)
            {
                float inferenceStart = Time.realtimeSinceStartup;

                Graphics.Blit(webCam, preprocessTexture);

                int modelInputSize = useHalfResolution ? 320 : inputSize;

                using (var input = new Tensor<float>(new TensorShape(1, 3, modelInputSize, modelInputSize)))
                {
                    var tt = new TextureTransform().SetDimensions(modelInputSize, modelInputSize);
                    TextureConverter.ToTensor(preprocessTexture, input, tt);

                    worker.Schedule(input);

                    using (var gpuOut = worker.PeekOutput() as Tensor<float>)
                    using (var cpuOut = gpuOut.ReadbackAndClone() as Tensor<float>)
                    {
                        var dets = ParseYOLO(cpuOut, modelInputSize);
                        var finalDets = NMS(dets);

                        lastDetections = finalDets.Select(d => d.Clone()).ToList();
                        lastInferenceTime = Time.realtimeSinceStartup;

                        UpdateWorldLabels(finalDets);

                        if (debugText != null)
                        {
                            float maxScore = finalDets.Count > 0 ? finalDets.Max(d => d.p) : 0f;
                            debugText.text =
                                $"Det: {finalDets.Count} | MaxConf: {maxScore:F2}\n" +
                                $"Cam: {webCam.width}x{webCam.height}\n" +
                                $"Frame: {frameCounter}";
                        }
                    }
                }

                inferenceMS = (Time.realtimeSinceStartup - inferenceStart) * 1000f;
            }
            else if (interpolateDetections && lastDetections.Count > 0)
            {
                // reuse last detections when we skip frames
                UpdateWorldLabels(lastDetections);
            }

            totalFrameTime = (Time.realtimeSinceStartup - frameStart) * 1000f;
            smoothFPS = Mathf.Lerp(smoothFPS, 1f / Mathf.Max(Time.deltaTime, 0.0001f), 0.1f);

            if (fpsText != null)
            {
                fpsText.text = $"FPS: {smoothFPS:F1}\n" +
                               $"Inf: {inferenceMS:F0}ms\n" +
                               $"Frame: {totalFrameTime:F0}ms";
            }

            yield return null;
        }
    }

    private void OnDestroy()
    {
        StopAllCoroutines();
        worker?.Dispose();

        if (webCam != null)
        {
            webCam.Stop();
            Destroy(webCam);
        }

        if (preprocessTexture != null)
        {
            preprocessTexture.Release();
            Destroy(preprocessTexture);
        }
    }

    // ---------- YOLO parsing & NMS ----------

    private List<Det> ParseYOLO(Tensor<float> o, int actualSize)
    {
        var dets = new List<Det>(maxDetections * 2);

        int d1 = o.shape[1];
        int d2 = o.shape[2];
        bool rowsAreDetections = d1 > d2;
        int N = rowsAreDetections ? d1 : d2;

        int checkedCount = 0;
        int maxChecks = N;

        for (int i = 0; i < maxChecks; i++)
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

                checkedCount++;
                if (checkedCount > maxDetections * 3)
                    break;
            }
        }

        return dets;
    }

    private List<Det> NMS(List<Det> dets)
    {
        if (dets.Count == 0) return dets;

        var result = new List<Det>(maxDetections);
        var list = dets.OrderByDescending(d => d.p).ToList();

        while (list.Count > 0 && result.Count < maxDetections)
        {
            var best = list[0];
            list.RemoveAt(0);
            result.Add(best);

            for (int i = list.Count - 1; i >= 0; i--)
            {
                if (list[i].cls == best.cls && IoU(list[i].r, best.r) > iouThreshold)
                {
                    list.RemoveAt(i);
                }
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

    // ---------- label update ----------

    private void UpdateWorldLabels(List<Det> dets)
    {
        if (worldLabels == null || labelParent == null) return;
        var cam = Camera.main;
        if (cam == null) return;

        int camLayer = cam.gameObject.layer;

        for (int i = 0; i < worldLabels.Length; i++)
        {
            if (worldLabels[i] == null || !worldLabels[i].IsValid)
                worldLabels[i] = new WorldLabel(labelParent, camLayer, textScale);
            else
                worldLabels[i].Clear();
        }

        int count = Mathf.Min(dets.Count, worldLabels.Length);

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

            Vector3 worldPointPos = cam.ViewportToWorldPoint(
                new Vector3(vxCenter, vyCenter, labelDepth));
            Vector3 worldLabelPos = cam.ViewportToWorldPoint(
                new Vector3(vxLabel, vyLabel, labelDepth));

            string txt = $"{classNames[d.cls]} {(d.p * 100f):F0}%";
            Color col = Color.HSVToRGB((float)d.cls / classNames.Length, 0.9f, 1f);

            label.Set(worldLabelPos, worldPointPos, txt, col);
        }
    }

    // ---------- UI callbacks ----------

    public void OnConfSliderChanged(float v) => confidenceThreshold = v;
    public void OnIoUSliderChanged(float v) => iouThreshold = v;

    private void UpdateStatus(string s)
    {
        if (statusText != null) statusText.text = s;
        Debug.Log("[YOLO] " + s);
    }
}
