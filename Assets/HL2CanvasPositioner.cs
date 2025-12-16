using UnityEngine;

public class HL2CanvasPositioner : MonoBehaviour
{
    public float distanceFromCamera = 1.5f;
    public bool followCamera = true;

    private Canvas canvas;

    void Start()
    {
        canvas = GetComponent<Canvas>();

        // Force world space
        if (canvas != null)
        {
            canvas.renderMode = RenderMode.WorldSpace;
        }

        // Initial position
        PositionCanvas();
    }

    void Update()
    {
        if (followCamera)
        {
            PositionCanvas();
        }
    }

    void PositionCanvas()
    {
        if (Camera.main == null) return;

        // Position canvas directly in front of camera
        Vector3 targetPosition = Camera.main.transform.position + Camera.main.transform.forward * distanceFromCamera;
        transform.position = targetPosition;

        // Make it face the camera
        transform.LookAt(Camera.main.transform);
        transform.Rotate(0, 180, 0);

        // Ensure proper scale
        transform.localScale = new Vector3(0.001f, 0.001f, 0.001f);
    }
}
