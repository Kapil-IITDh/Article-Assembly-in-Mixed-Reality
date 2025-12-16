using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using TMPro;

public class YoloHL2Manager : MonoBehaviour
{
    [Header("Model")]
    public Unity.InferenceEngine.ModelAsset modelAsset;
    public int inputSize = 640;

    [Header("Detection Settings")]
    [Range(0.1f, 0.9f)] public float confidenceThreshold = 0.5f;
    [Range(0.1f, 0.9f)] public float iouThreshold = 0.5f;
    public float detectionInterval = 0.1f; // How often to run detection

    [Header("UI")]
    public TextMeshProUGUI statusText;
    public GameObject labelPrefab;

    [Header("Label Settings")]
    public float labelDistance = 2f;
    public float labelScale = 0.005f;
    public float labelLifetime = 3f; // How long labels stay after last detection
    public float positionTolerance = 0.3f; // Merge labels within this distance

    [Header("Performance")]
    public bool useAsyncDetection = true;

    private YoloDetector detector;
    private WebCamTexture webCamTexture;
    private Dictionary<string, LabelInfo> activeLabels = new Dictionary<string, LabelInfo>();
    private bool isInitialized = false;
    private bool isDetecting = false;
    private int frameCount = 0;
    private float lastDetectionTime = 0f;

    private class LabelInfo
    {
        public GameObject gameObject;
        public float lastSeenTime;
        public Vector3 targetPosition;
        public string className;
        public float confidence;
    }

    void Start()
    {
        Debug.Log("[YOLO] Application started");
        StartCoroutine(Initialize());
    }

    void Update()
    {
        // Smooth label movement every frame
        UpdateLabelPositions();

        // Run detection at controlled intervals
        if (isInitialized && !isDetecting && Time.time - lastDetectionTime >= detectionInterval)
        {
            lastDetectionTime = Time.time;
            if (useAsyncDetection)
            {
                StartCoroutine(DetectAndDisplayAsync());
            }
            else
            {
                DetectAndDisplay();
            }
        }
    }

    IEnumerator Initialize()
    {
        UpdateStatus("Starting HoloLens 2...");
        yield return new WaitForSeconds(1f);

        UpdateStatus("Initializing camera...");
        yield return StartCoroutine(InitializeCamera());

        if (webCamTexture == null || !webCamTexture.isPlaying)
        {
            UpdateStatus("ERROR: Camera failed!");
            yield break;
        }

        UpdateStatus("Loading YOLO model...");
        yield return new WaitForSeconds(1f);

        detector = new YoloDetector(modelAsset, inputSize, confidenceThreshold, iouThreshold);
        if (!detector.Initialize())
        {
            UpdateStatus("ERROR: Model failed!");
            yield break;
        }

        isInitialized = true;
        UpdateStatus("READY! Point at objects...");
        Debug.Log("[YOLO] System initialized successfully");
    }

    IEnumerator InitializeCamera()
    {
        WebCamDevice[] devices = WebCamTexture.devices;

        if (devices.Length == 0)
        {
            Debug.LogError("[CAMERA] No cameras found");
            yield break;
        }

        string selectedCamera = devices[0].name;
        foreach (var device in devices)
        {
            if (device.name.Contains("RGB") || device.name.Contains("Photo"))
            {
                selectedCamera = device.name;
                break;
            }
        }

        webCamTexture = new WebCamTexture(selectedCamera, 640, 360, 15);
        webCamTexture.Play();

        float timeout = 10f;
        while (!webCamTexture.isPlaying && timeout > 0)
        {
            timeout -= 0.1f;
            yield return new WaitForSeconds(0.1f);
        }

        if (!webCamTexture.isPlaying)
        {
            Debug.LogError("[CAMERA] Failed to start");
            yield break;
        }

        yield return new WaitForSeconds(1f);
        Debug.Log($"[CAMERA] Started: {webCamTexture.width}x{webCamTexture.height}");
    }

    void DetectAndDisplay()
    {
        if (!isInitialized || webCamTexture == null || !webCamTexture.isPlaying || isDetecting)
            return;

        isDetecting = true;
        var detections = detector.Detect(webCamTexture);
        ProcessDetections(detections);
        isDetecting = false;
    }

    IEnumerator DetectAndDisplayAsync()
    {
        if (!isInitialized || webCamTexture == null || !webCamTexture.isPlaying || isDetecting)
            yield break;

        isDetecting = true;

        // Run detection
        var detections = detector.Detect(webCamTexture);

        yield return null; // Allow frame to process

        ProcessDetections(detections);

        isDetecting = false;
    }

    void ProcessDetections(List<YoloDetector.Detection> detections)
    {
        float currentTime = Time.time;
        frameCount++;

        // Track which labels are still active
        var seenLabels = new HashSet<string>();

        foreach (var detection in detections)
        {
            Vector3 position = CalculatePosition(detection);

            // Find if we have an existing label nearby for same class
            string existingKey = FindNearbyLabel(detection.className, position);

            if (existingKey != null)
            {
                // Update existing label
                UpdateExistingLabel(existingKey, detection, position, currentTime);
                seenLabels.Add(existingKey);
            }
            else
            {
                // Create new label
                string newKey = CreateUniqueKey(detection.className);
                CreateLabel(detection.className, detection.confidence, position, newKey, currentTime);
                seenLabels.Add(newKey);
            }
        }

        // Remove expired labels
        RemoveExpiredLabels(currentTime);

        UpdateStatus($"Tracking: {activeLabels.Count} | FPS: {frameCount / Time.time:F1}");
    }

    string FindNearbyLabel(string className, Vector3 position)
    {
        foreach (var kvp in activeLabels)
        {
            if (kvp.Value.className == className)
            {
                float distance = Vector3.Distance(kvp.Value.targetPosition, position);
                if (distance < positionTolerance)
                {
                    return kvp.Key;
                }
            }
        }
        return null;
    }

    void UpdateExistingLabel(string key, YoloDetector.Detection detection, Vector3 position, float currentTime)
    {
        if (!activeLabels.ContainsKey(key))
            return;

        var labelInfo = activeLabels[key];
        labelInfo.lastSeenTime = currentTime;
        labelInfo.targetPosition = position;
        labelInfo.confidence = detection.confidence;

        // Update text
        if (labelInfo.gameObject != null)
        {
            UpdateLabelText(labelInfo.gameObject, detection.className, detection.confidence);
        }
    }

    void UpdateLabelPositions()
    {
        if (Camera.main == null) return;

        foreach (var kvp in activeLabels)
        {
            if (kvp.Value.gameObject == null) continue;

            // Smooth position interpolation
            kvp.Value.gameObject.transform.position = Vector3.Lerp(
                kvp.Value.gameObject.transform.position,
                kvp.Value.targetPosition,
                Time.deltaTime * 5f // Smooth movement
            );

            // Always face camera
            kvp.Value.gameObject.transform.LookAt(Camera.main.transform);
            kvp.Value.gameObject.transform.Rotate(0, 180, 0);
        }
    }

    string CreateUniqueKey(string className)
    {
        // Generate unique key that won't change based on position
        return $"{className}_{Time.time}_{UnityEngine.Random.Range(0, 10000)}";
    }

    void CreateLabel(string className, float confidence, Vector3 position, string key, float currentTime)
    {
        if (labelPrefab == null)
        {
            Debug.LogError("[LABEL] Label prefab is NULL!");
            return;
        }

        GameObject label = Instantiate(labelPrefab, position, Quaternion.identity);

        // Hide mesh renderers
        foreach (var mr in label.GetComponentsInChildren<MeshRenderer>(true))
        {
            mr.enabled = false;
        }

        // Configure text
        ConfigureLabelText(label, className, confidence);

        // Face camera
        if (Camera.main != null)
        {
            label.transform.LookAt(Camera.main.transform);
            label.transform.Rotate(0, 180, 0);
        }

        label.SetActive(true);

        // Store label info
        activeLabels[key] = new LabelInfo
        {
            gameObject = label,
            lastSeenTime = currentTime,
            targetPosition = position,
            className = className,
            confidence = confidence
        };

        Debug.Log($"[LABEL] Created '{className}' at {position}");
    }

    void ConfigureLabelText(GameObject label, string className, float confidence)
    {
        string labelText = $"{className}\n{confidence * 100:F0}%";

        // Try TextMeshProUGUI
        TextMeshProUGUI tmpUI = label.GetComponentInChildren<TextMeshProUGUI>(true);
        if (tmpUI != null)
        {
            tmpUI.text = labelText;
            tmpUI.fontSize = 60;
            tmpUI.color = Color.green;
            tmpUI.alignment = TextAlignmentOptions.Center;
            tmpUI.fontStyle = FontStyles.Bold;
            tmpUI.gameObject.SetActive(true);

            Canvas canvas = tmpUI.GetComponentInParent<Canvas>();
            if (canvas != null)
            {
                canvas.renderMode = RenderMode.WorldSpace;
                canvas.transform.localScale = Vector3.one * labelScale;
            }
            return;
        }

        // Try TextMeshPro 3D
        TextMeshPro tmp3D = label.GetComponentInChildren<TextMeshPro>(true);
        if (tmp3D != null)
        {
            tmp3D.text = labelText;
            tmp3D.fontSize = 1f;
            tmp3D.color = Color.green;
            tmp3D.alignment = TextAlignmentOptions.Center;
            tmp3D.fontStyle = FontStyles.Bold;
            tmp3D.gameObject.SetActive(true);
            return;
        }

        // Try TextMesh
        TextMesh textMesh = label.GetComponentInChildren<TextMesh>(true);
        if (textMesh != null)
        {
            textMesh.text = labelText;
            textMesh.fontSize = 100;
            textMesh.color = Color.green;
            textMesh.anchor = TextAnchor.MiddleCenter;
            textMesh.gameObject.SetActive(true);
            textMesh.transform.localScale = Vector3.one * 0.01f;
            return;
        }
    }

    void UpdateLabelText(GameObject label, string className, float confidence)
    {
        string labelText = $"{className}\n{confidence * 100:F0}%";

        TextMeshProUGUI tmpUI = label.GetComponentInChildren<TextMeshProUGUI>(true);
        if (tmpUI != null)
        {
            tmpUI.text = labelText;
            return;
        }

        TextMeshPro tmp3D = label.GetComponentInChildren<TextMeshPro>(true);
        if (tmp3D != null)
        {
            tmp3D.text = labelText;
            return;
        }

        TextMesh textMesh = label.GetComponentInChildren<TextMesh>(true);
        if (textMesh != null)
        {
            textMesh.text = labelText;
        }
    }

    void RemoveExpiredLabels(float currentTime)
    {
        var keysToRemove = new List<string>();

        foreach (var kvp in activeLabels)
        {
            // Remove labels not seen for labelLifetime seconds
            if (currentTime - kvp.Value.lastSeenTime > labelLifetime)
            {
                if (kvp.Value.gameObject != null)
                {
                    Destroy(kvp.Value.gameObject);
                }
                keysToRemove.Add(kvp.Key);
            }
        }

        foreach (var key in keysToRemove)
        {
            activeLabels.Remove(key);
        }
    }

    Vector3 CalculatePosition(YoloDetector.Detection detection)
    {
        if (Camera.main == null)
            return Vector3.zero;

        float centerX = detection.box.x + detection.box.width / 2f;
        float centerY = detection.box.y + detection.box.height / 2f;

        Vector2 viewportPoint = new Vector2(centerX, 1f - centerY);
        Ray ray = Camera.main.ViewportPointToRay(viewportPoint);

        // Try spatial mesh raycast
        RaycastHit hit;
        if (Physics.Raycast(ray, out hit, 10f))
        {
            return hit.point;
        }

        // Place at fixed distance
        return ray.origin + ray.direction * labelDistance;
    }

    void UpdateStatus(string message)
    {
        if (statusText != null)
            statusText.text = message;
    }

    void OnDestroy()
    {
        detector?.Dispose();

        if (webCamTexture != null && webCamTexture.isPlaying)
            webCamTexture.Stop();

        foreach (var labelInfo in activeLabels.Values)
        {
            if (labelInfo.gameObject != null)
                Destroy(labelInfo.gameObject);
        }
        activeLabels.Clear();
    }
}
