using UnityEngine;
using System.Collections;
using MixedReality.Toolkit.UX;
using UnityEngine.EventSystems;

public class MultiAnimatorStepController : MonoBehaviour
{
    [Header("Animators")]
    public Animator[] animators = new Animator[4];

    [Header("Animation State")]
    public string animStateName;

    [Header("Segment Durations Per Animator")]
    public float[][] segmentDurations = new float[4][];

    [Header("MRTK Pressable Buttons")]
    public PressableButton nextButton;
    public PressableButton doneButton;

    private int[] currentSegments = new int[4];
    private bool isWaitingForInput = false;

    void Start()
    {
        // Example segment durations
        segmentDurations[0] = new float[] { 4f, 4f, 1f, 3f };
        segmentDurations[1] = new float[] { 4f, 4f, 1f, 3f };
        segmentDurations[2] = new float[] { 4f, 4f, 1f, 3f };
        segmentDurations[3] = new float[] { 4f, 4f, 1f, 3f };

        // Hide UI at start
        nextButton.gameObject.SetActive(false);
        doneButton.gameObject.SetActive(false);

        // MRTK3 PressableButton event
        nextButton.OnClicked.AddListener(OnNextButtonClicked);

        // Add mouse click support (UNITY EDITOR ONLY)
#if UNITY_EDITOR
        nextButton.gameObject.AddComponent<HybridClickHelper>().Init(OnNextButtonClicked);
#endif

        // Initialize animators
        for (int i = 0; i < animators.Length; i++)
        {
            currentSegments[i] = 0;
            animators[i].Play(animStateName);
            animators[i].speed = 1;
        }

        StartCoroutine(PlaySegments());
    }

    IEnumerator PlaySegments()
    {
        while (true)
        {
            float maxSegmentTime = 0f;

            for (int i = 0; i < animators.Length; i++)
            {
                if (currentSegments[i] < segmentDurations[i].Length)
                {
                    float seg = segmentDurations[i][currentSegments[i]];
                    if (seg > maxSegmentTime)
                        maxSegmentTime = seg;
                }
            }

            // Wait for longest segment
            yield return new WaitForSeconds(maxSegmentTime);

            // Pause animations
            foreach (var a in animators)
                a.speed = 0;

            // Show button
            isWaitingForInput = true;
            nextButton.gameObject.SetActive(true);

            // Wait until Next clicked
            yield return new WaitUntil(() => !isWaitingForInput);

            // Advance all animators
            bool anyLeft = false;
            for (int i = 0; i < currentSegments.Length; i++)
            {
                if (currentSegments[i] + 1 < segmentDurations[i].Length)
                {
                    currentSegments[i]++;
                    anyLeft = true;
                }
            }

            // End if finished
            if (!anyLeft)
                break;

            // Resume animation
            foreach (var a in animators)
                a.speed = 1;
        }

        // Final UI
        nextButton.gameObject.SetActive(false);
        doneButton.gameObject.SetActive(true);

        Debug.Log("All animation segments completed.");
    }

    void OnNextButtonClicked()
    {
        if (isWaitingForInput)
        {
            isWaitingForInput = false;
            nextButton.gameObject.SetActive(false);
        }
    }
}
