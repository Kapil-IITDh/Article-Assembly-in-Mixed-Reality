using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class HL2CameraTestRawImage : MonoBehaviour
{
    [Header("UI")]
    public RawImage display;          // RawImage where camera feed will show
    public TextMeshProUGUI debugText; // Text area for logs

    private WebCamTexture camTex;

    private void Start()
    {
        StartCoroutine(InitCamera());
    }

    private IEnumerator InitCamera()
    {
        // 1) List devices
        var devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            SetDebug("No WebCam devices.\nCheck Capabilities (WebCam) + app permission.");
            yield break;
        }

        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.AppendLine("Devices:");
        for (int i = 0; i < devices.Length; i++)
        {
            sb.AppendLine($"[{i}] {devices[i].name}  front={devices[i].isFrontFacing}");
        }
        SetDebug(sb.ToString());

        // 2) Pick a device
        WebCamDevice chosen = devices[0];
        foreach (var d in devices)
        {
            if (!d.isFrontFacing)
            {
                chosen = d;   // prefer back/world camera
                break;
            }
        }

        // 3) Start camera
        camTex = new WebCamTexture(chosen.name, 896, 504, 30);
        camTex.Play();

        float startTime = Time.realtimeSinceStartup;
        bool gotFrame = false;

        while (Time.realtimeSinceStartup - startTime < 10f) // 10 sec timeout
        {
            if (camTex.didUpdateThisFrame && camTex.width > 16 && camTex.height > 16)
            {
                gotFrame = true;
                break;
            }

            SetDebug(
                $"Starting camera:\n{chosen.name}\n" +
                $"isPlaying={camTex.isPlaying}\n" +
                $"w={camTex.width}, h={camTex.height}\n" +
                $"t={(Time.realtimeSinceStartup - startTime):F1}s"
            );

            yield return null;
        }

        if (!gotFrame)
        {
            SetDebug(
                $"Camera started but no frames.\n" +
                $"isPlaying={camTex.isPlaying}\n" +
                $"w={camTex.width}, h={camTex.height}\n\n" +
                "Check HoloLens Settings > Privacy > Camera\n" +
                "and ensure this app is allowed."
            );
            yield break;
        }

        // 4) Attach texture to RawImage
        if (display != null)
        {
            display.texture = camTex;
            // optional: stretch to full rect
            display.rectTransform.sizeDelta = Vector2.zero;
        }

        SetDebug(
            $"Using: {chosen.name} (front={chosen.isFrontFacing})\n" +
            $"Resolution: {camTex.width}x{camTex.height}"
        );
    }

    private void OnDestroy()
    {
        if (camTex != null)
        {
            camTex.Stop();
            Destroy(camTex);
        }
    }

    private void SetDebug(string msg)
    {
        if (debugText != null)
            debugText.text = msg;
    }
}
