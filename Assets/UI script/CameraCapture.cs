using UnityEngine;

public class CameraCapture : MonoBehaviour
{
    public Renderer displayRenderer;   // RawImage or Quad
    private WebCamTexture camTexture;

    void Start()
    {
        // Use default camera
        camTexture = new WebCamTexture();

        // Assign to material on a Quad/Plane
        if (displayRenderer != null)
        {
            displayRenderer.material.mainTexture = camTexture;
        }

        camTexture.Play();

        Debug.Log($"📷 Camera started: {camTexture.width} x {camTexture.height}");
    }

    public Texture GetFrame()
    {
        return camTexture;
    }

    void OnDestroy()
    {
        camTexture?.Stop();
    }
}
