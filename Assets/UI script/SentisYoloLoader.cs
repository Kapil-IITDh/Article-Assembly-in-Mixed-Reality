using UnityEngine;
using Unity.Sentis;

public class SentisYoloLoader : MonoBehaviour
{
    // Drag your .onnx ModelAsset here in the Inspector
    public ModelAsset modelAsset;

    private Model runtimeModel;
    private Worker worker;

    void Start()
    {
        if (modelAsset == null)
        {
            Debug.LogError("❌ [SentisYoloLoader] No ModelAsset assigned!");
            return;
        }

        // 1. Load runtime model from asset
        runtimeModel = ModelLoader.Load(modelAsset);

        // 2. Create a worker
        //    For now, use CPU to avoid any GPU / compute-shader issues.
        //    Once it works, we can switch to BackendType.GPUCompute.
        worker = new Worker(runtimeModel, BackendType.CPU);
        // worker = new Worker(runtimeModel, BackendType.GPUCompute); // later, if needed

        Debug.Log("✅ [SentisYoloLoader] Model loaded and Worker created successfully.");
    }

    private void OnDestroy()
    {
        if (worker != null)
        {
            worker.Dispose();
            worker = null;
        }
    }
}
