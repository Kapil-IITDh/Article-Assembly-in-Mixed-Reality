using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Sentis;
using TMPro;

public class YoloCameraRunner : MonoBehaviour
{
    [Header("Inputs")]
    [Tooltip("Script that provides the camera Texture (your CameraCapture).")]
    public CameraCapture cameraSource;

    [Tooltip("Your YOLO ONNX model asset (Sentis ModelAsset).")]
    public ModelAsset modelAsset;

    [Header("YOLO Settings")]
    [Tooltip("Model input size (must match training, usually 640).")]
    public int inputSize = 640;

    [Range(0.01f, 0.99f)]
    public float confidenceThreshold = 0.4f;   // slightly relaxed

    [Range(0.1f, 0.9f)]
    public float iouThreshold = 0.5f;

    [Tooltip("Maximum labels to show at once.")]
    public int maxDetections = 5;

    [Tooltip("Run YOLO every N frames (1–3 helps reduce flicker & load).")]
    public int runEveryNFrames = 1;

    [Header("Label Settings")]
    [Tooltip("Distance from camera where labels will be placed (in meters).")]
    public float labelDepth = 2.0f;

    [Tooltip("Vertical offset (viewport units) so label appears above box centre.")]
    public float labelViewportYOffset = 0.06f;

    [Tooltip("Scale of the world-space text (0.001 = small, 0.003 = medium).")]
    public float textScale = 0.002f;

    [Header("Optional Debug UI")]
    public TextMeshPro statusText;

    // Your 8 classes
    private readonly string[] classNames =
    {
        "Tools", "Housing", "Big Screw", "Small Screw",
        "V-Lock Plate", "Plunger", "Helical Spring", "Cover"
    };

    // Sentis
    private Model model;
    private Worker worker;
    private Tensor<float> inputTensor;

    // Label pool
    private WorldLabel[] worldLabels;
    private Transform labelParent;

    // Temporal smoothing
    private List<Detection> lastDetections = new List<Detection>();
    private int framesSinceLastDetections = 0;

    [Tooltip("How many empty frames before labels disappear.")]
    public int holdFrames = 8;

    private int frameCounter = 0;

    // ---- helper types ----

    private class Detection
    {
        public int cls;
        public float score;
        public Rect rect;   // normalized [0,1] x,y,w,h

        public Detection(int cls, float score, Rect rect)
        {
            this.cls = cls;
            this.score = score;
            this.rect = rect;
        }

        public Detection Clone()
        {
            return new Detection(cls, score, rect);
        }
    }

    private class WorldLabel
    {
        public GameObject root;
        public Transform t;
        public TextMeshPro text;

        public bool IsValid => root != null && t != null && text != null;

        public WorldLabel(Transform parent, int layer, float textScale)
        {
            Create(parent, layer, textScale);
        }

        private void Create(Transform parent, int layer, float textScale)
        {
            root = new GameObject("DetectionLabel");
            t = root.transform;
            t.SetParent(parent, false);
            t.localPosition = Vector3.zero;
            t.localRotation = Quaternion.identity;
            t.localScale = Vector3.one * Mathf.Max(textScale, 0.0001f);
            root.layer = layer;

            var textObj = new GameObject("Text");
            textObj.transform.SetParent(t, false);
            textObj.transform.localPosition = Vector3.zero;
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

            Clear();
        }

        public void Set(Vector3 worldPos, string txt, Color col)
        {
            if (!IsValid) return;

            t.position = worldPos;
            var cam = Camera.main;
            if (cam != null)
            {
                // billboard towards camera
                t.rotation = Quaternion.LookRotation(t.position - cam.transform.position);
            }

            text.text = txt;
            text.color = col;
            root.SetActive(true);
        }

        public void Clear()
        {
            if (root != null)
                root.SetActive(false);
        }
    }

    // ---- lifecycle ----

    private void Start()
    {
        if (cameraSource == null)
        {
            Debug.LogError("[YoloCameraRunner] cameraSource not assigned!");
            UpdateStatus("ERROR: cameraSource not set");
            enabled = false;
            return;
        }

        if (modelAsset == null)
        {
            Debug.LogError("[YoloCameraRunner] modelAsset not assigned!");
            UpdateStatus("ERROR: modelAsset not set");
            enabled = false;
            return;
        }

        labelDepth = Mathf.Max(labelDepth, 0.5f);
        runEveryNFrames = Mathf.Max(1, runEveryNFrames);
        holdFrames = Mathf.Max(1, holdFrames);

        // Set label parent to Camera.main so labels move with the camera
        var cam = Camera.main;
        if (cam != null)
        {
            labelParent = cam.transform;
        }
        else
        {
            labelParent = this.transform;
            Debug.LogWarning("[YoloCameraRunner] No Camera.main found. Labels will attach to this object.");
        }

        // Load model + worker
        model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.CPU); // you can try GPUCompute later

        // Create reusable input tensor: NCHW = (1, 3, H, W)
        inputTensor = new Tensor<float>(new TensorShape(1, 3, inputSize, inputSize));

        // Create label pool
        int layer = cam != null ? cam.gameObject.layer : 0;
        worldLabels = new WorldLabel[maxDetections];
        for (int i = 0; i < maxDetections; i++)
        {
            worldLabels[i] = new WorldLabel(labelParent, layer, textScale);
        }

        UpdateStatus("YOLO ready");
        Debug.Log("[YoloCameraRunner] Initialized.");
    }

    private void Update()
    {
        if (worker == null) return;

        frameCounter++;

        Texture frame = cameraSource.GetFrame();
        if (frame == null) return;

        bool shouldRun = (frameCounter % runEveryNFrames) == 0;

        if (shouldRun)
        {
            // Resize + convert to tensor
            var tt = new TextureTransform().SetDimensions(inputSize, inputSize);
            TextureConverter.ToTensor(frame, inputTensor, tt);

            // Run inference
            worker.Schedule(inputTensor);

            // Get output on GPU and then bring to CPU
            using (var gpuOut = worker.PeekOutput() as Tensor<float>)
            using (var cpuOut = gpuOut.ReadbackAndClone() as Tensor<float>)
            {
                var dets = DecodeDetections(cpuOut, inputSize);
                var finalDets = ApplyNMS(dets);

                if (finalDets.Count > 0)
                {
                    // store + reset timer
                    lastDetections = finalDets.Select(d => d.Clone()).ToList();
                    framesSinceLastDetections = 0;

                    ApplyWorldLabels(finalDets);
                }
                else
                {
                    framesSinceLastDetections++;
                    HandleNoDetections();
                }
            }
        }
        else
        {
            // no fresh inference this frame – reuse last detections if any
            if (lastDetections.Count > 0)
            {
                framesSinceLastDetections++;
                HandleNoDetections();
            }
            else
            {
                ClearAllLabels();
                UpdateStatus("No detections");
            }
        }
    }

    private void OnDestroy()
    {
        worker?.Dispose();
        inputTensor?.Dispose();
    }

    // ---- YOLO decoding ----
    // Output shape: [1, 12, 8400]
    // channels = 12 = 4 box + 1 objConf + 7 class scores (8 classes)
    private List<Detection> DecodeDetections(Tensor<float> t, int modelSize)
    {
        var dets = new List<Detection>();

        int C = t.shape[1]; // 12
        int N = t.shape[2]; // 8400

        for (int i = 0; i < N; i++)
        {
            float cx = t[0, 0, i];
            float cy = t[0, 1, i];
            float w = t[0, 2, i];
            float h = t[0, 3, i];

            // assuming pixel coords – convert to [0,1] (if model already outputs 0..1, this just rescales)
            cx /= modelSize;
            cy /= modelSize;
            w /= modelSize;
            h /= modelSize;

            if (w <= 0f || h <= 0f) continue;

            float objConf = t[0, 4, i];
            if (objConf < 0.1f) // tiny gate, not the main threshold
                continue;

            int bestCls = -1;
            float bestScore = 0f;

            for (int c = 0; c < classNames.Length; c++)
            {
                float p = t[0, 5 + c, i];
                if (p > bestScore)
                {
                    bestScore = p;
                    bestCls = c;
                }
            }

            // Make it less strict: use class score directly as final score
            float finalScore = bestScore; // instead of objConf * bestScore

            if (bestCls < 0 || finalScore < confidenceThreshold)
                continue;

            float x = cx - w * 0.5f;
            float y = cy - h * 0.5f;
            Rect r = new Rect(x, y, w, h); // normalized

            dets.Add(new Detection(bestCls, finalScore, r));
        }

        return dets;
    }

    private List<Detection> ApplyNMS(List<Detection> dets)
    {
        if (dets.Count == 0) return dets;

        var sorted = dets.OrderByDescending(d => d.score).ToList();
        var result = new List<Detection>(maxDetections);

        while (sorted.Count > 0 && result.Count < maxDetections)
        {
            var best = sorted[0];
            sorted.RemoveAt(0);
            result.Add(best);

            for (int i = sorted.Count - 1; i >= 0; i--)
            {
                if (sorted[i].cls == best.cls && IoU(sorted[i].rect, best.rect) > iouThreshold)
                {
                    sorted.RemoveAt(i);
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

        if (union <= 0f) return 0f;
        return inter / union;
    }

    // ---- World-space label placement ----

    private void ApplyWorldLabels(List<Detection> dets)
    {
        if (worldLabels == null || labelParent == null) return;
        var cam = Camera.main;
        if (cam == null) return;

        ClearAllLabels();

        int count = Mathf.Min(dets.Count, worldLabels.Length);

        for (int i = 0; i < count; i++)
        {
            var d = dets[i];
            var label = worldLabels[i];

            if (!label.IsValid) continue;

            // centre of box (normalized 0..1)
            float cx = d.rect.x + d.rect.width * 0.5f;
            float cy = d.rect.y + d.rect.height * 0.5f;

            // YOLO is usually top-left origin, Unity viewport is bottom-left
            float vxCenter = cx;
            float vyCenter = 1f - cy;

            float vxLabel = vxCenter;
            float vyLabel = vyCenter + labelViewportYOffset;

            Vector3 worldLabelPos = cam.ViewportToWorldPoint(
                new Vector3(vxLabel, vyLabel, labelDepth)
            );

            string txt = $"{classNames[d.cls]} {(d.score * 100f):F0}%";
            Color col = Color.HSVToRGB((float)d.cls / classNames.Length, 0.9f, 1f);

            label.Set(worldLabelPos, txt, col);
        }

        if (dets.Count > 0)
        {
            UpdateStatus($"Detections: {dets.Count}");
        }
        else
        {
            UpdateStatus("No detections");
        }
    }

    private void ClearAllLabels()
    {
        if (worldLabels == null) return;
        for (int i = 0; i < worldLabels.Length; i++)
            worldLabels[i].Clear();
    }

    private void HandleNoDetections()
    {
        if (lastDetections.Count > 0 && framesSinceLastDetections <= holdFrames)
        {
            // Reuse last good detections to keep labels stable
            ApplyWorldLabels(lastDetections);
        }
        else
        {
            ClearAllLabels();
            UpdateStatus("No detections");
        }
    }

    // ---- UI helper ----

    private void UpdateStatus(string msg)
    {
        if (statusText != null)
            statusText.text = msg;
    }
}
