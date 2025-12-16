using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Main detector class
public class YoloDetector
{
    public struct Detection
    {
        public string className;
        public float confidence;
        public Rect box;
    }

    private Unity.InferenceEngine.Model model;
    private Unity.InferenceEngine.Worker worker;
    private int inputSize;
    private float confidenceThreshold;
    private float iouThreshold;

    private readonly string[] classNames = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    public YoloDetector(Unity.InferenceEngine.ModelAsset modelAsset, int inputSize, float confidenceThreshold, float iouThreshold)
    {
        this.inputSize = inputSize;
        this.confidenceThreshold = confidenceThreshold;
        this.iouThreshold = iouThreshold;

        try
        {
            model = Unity.InferenceEngine.ModelLoader.Load(modelAsset);
            worker = new Unity.InferenceEngine.Worker(model, Unity.InferenceEngine.BackendType.GPUCompute);
            Debug.Log("[DETECTOR] Initialized successfully");
        }
        catch (Exception e)
        {
            Debug.LogError($"[DETECTOR] Failed: {e.Message}");
        }
    }

    public bool Initialize()
    {
        return worker != null;
    }

    public List<Detection> Detect(WebCamTexture webCamTexture)
    {
        List<Detection> detections = new List<Detection>();
        Unity.InferenceEngine.Tensor<float> inputTensor = null;
        Unity.InferenceEngine.Tensor<float> outputTensor = null;
        Unity.InferenceEngine.Tensor<float> cpuTensor = null;

        try
        {
            inputTensor = PreprocessImage(webCamTexture);
            if (inputTensor == null) return detections;

            worker.Schedule(inputTensor);
            outputTensor = worker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;
            if (outputTensor == null) return detections;

            cpuTensor = outputTensor.ReadbackAndClone();
            detections = ProcessDetections(cpuTensor);
        }
        catch (Exception e)
        {
            Debug.LogError($"[DETECTOR] Detection failed: {e.Message}");
        }
        finally
        {
            inputTensor?.Dispose();
            cpuTensor?.Dispose();
        }
        return detections;
    }

    private Unity.InferenceEngine.Tensor<float> PreprocessImage(WebCamTexture webCamTexture)
    {
        if (webCamTexture == null || !webCamTexture.isPlaying) return null;

        Color32[] pixels = webCamTexture.GetPixels32();
        int width = webCamTexture.width;
        int height = webCamTexture.height;

        var tensor = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(1, 3, inputSize, inputSize));

        float scaleX = (float)width / inputSize;
        float scaleY = (float)height / inputSize;

        for (int y = 0; y < inputSize; y++)
        {
            for (int x = 0; x < inputSize; x++)
            {
                int srcX = Mathf.Clamp((int)(x * scaleX), 0, width - 1);
                int srcY = Mathf.Clamp((int)(y * scaleY), 0, height - 1);
                int pixelIndex = srcY * width + srcX;

                Color32 pixel = pixels[pixelIndex];

                tensor[0, 0, y, x] = pixel.r / 255f;
                tensor[0, 1, y, x] = pixel.g / 255f;
                tensor[0, 2, y, x] = pixel.b / 255f;
            }
        }
        return tensor;
    }

    private List<Detection> ProcessDetections(Unity.InferenceEngine.Tensor<float> output)
    {
        var detections = new List<Detection>();
        var shape = output.shape;

        if (shape.rank != 3) return detections;

        int features = shape[1];
        int predictions = shape[2];
        bool transposed = features > predictions;

        if (transposed)
        {
            int temp = features;
            features = predictions;
            predictions = temp;
        }

        if (features != 85) return detections; // expect 85: 4 box, 1 objectness, 80 classes

        for (int i = 0; i < predictions; i++)
        {
            float cx, cy, w, h, objectness;
            if (transposed)
            {
                cx = output[0, i, 0];
                cy = output[0, i, 1];
                w = output[0, i, 2];
                h = output[0, i, 3];
                objectness = 1f / (1f + Mathf.Exp(-output[0, i, 4]));
            }
            else
            {
                cx = output[0, 0, i];
                cy = output[0, 1, i];
                w = output[0, 2, i];
                h = output[0, 3, i];
                objectness = 1f / (1f + Mathf.Exp(-output[0, 4, i]));
            }

            float maxConf = 0f;
            int bestClass = -1;

            for (int c = 0; c < 80; c++)
            {
                float conf = transposed ? output[0, i, 5 + c] : output[0, 5 + c, i];
                conf = 1f / (1f + Mathf.Exp(-conf)); // sigmoid for class
                float classScore = objectness * conf;
                if (classScore > maxConf)
                {
                    maxConf = classScore;
                    bestClass = c;
                }
            }

            if (maxConf > confidenceThreshold && bestClass >= 0)
            {
                detections.Add(new Detection
                {
                    className = classNames[bestClass],
                    confidence = maxConf,
                    box = new Rect(
                        (cx - w / 2f) / inputSize,
                        (cy - h / 2f) / inputSize,
                        w / inputSize,
                        h / inputSize
                    )
                });
            }
        }

        return ApplyNMS(detections);
    }

    private List<Detection> ApplyNMS(List<Detection> detections)
    {
        detections.Sort((a, b) => b.confidence.CompareTo(a.confidence));
        var result = new List<Detection>();

        foreach (var det in detections)
        {
            bool keep = true;
            foreach (var kept in result)
            {
                if (det.className == kept.className && IoU(det.box, kept.box) > iouThreshold)
                {
                    keep = false;
                    break;
                }
            }
            if (keep) result.Add(det);
        }
        return result;
    }

    private float IoU(Rect a, Rect b)
    {
        float xA = Mathf.Max(a.xMin, b.xMin);
        float yA = Mathf.Max(a.yMin, b.yMin);
        float xB = Mathf.Min(a.xMax, b.xMax);
        float yB = Mathf.Min(a.yMax, b.yMax);

        float inter = Mathf.Max(0, xB - xA) * Mathf.Max(0, yB - yA);
        float union = a.width * a.height + b.width * b.height - inter;

        return union > 0 ? inter / union : 0;
    }

    public void Dispose()
    {
        worker?.Dispose();
    }
}

// Runner component to call above detector out-of-band from Update()
public class YoloDetectorRunner : MonoBehaviour
{
    public YoloDetector detector;
    public WebCamTexture webcam;

    private void Start()
    {
        StartCoroutine(DetectionCoroutine());
    }

    private IEnumerator DetectionCoroutine()
    {
        while (true)
        {
            if (webcam != null && webcam.isPlaying && detector != null && detector.Initialize())
            {
                var detections = detector.Detect(webcam);
                // handle detections here
            }
            yield return new WaitForSeconds(0.1f); // Run every 100ms, adjust as needed
        }
    }

    private void OnDestroy()
    {
        detector?.Dispose();
    }
}
