using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using TMPro;

public class AssemblyManager : MonoBehaviour
{
    [Header("Part Prefabs")]
    public GameObject part1Prefab; // Body4
    public GameObject part2Prefab; // Body2
    public GameObject part3Prefab; // Body3

    [Header("Animation Clips")]
    public AnimationClip[] part1Animations;
    public AnimationClip[] part2Animations;
    public AnimationClip[] part3Animations;

    [Header("UI")]
    public GameObject popupPanel;
    public Button nextButton;
    public TextMeshProUGUI popupText;

    // Spawned instances and animators
    private GameObject part1, part2, part3;
    private Animator animPart1, animPart2, animPart3;

    private bool userClickedNext = false;

    void Start()
    {
        popupPanel.SetActive(false);
        nextButton.onClick.AddListener(OnNextButtonClicked);
        StartCoroutine(FullAssemblyWorkflow());
    }

    IEnumerator FullAssemblyWorkflow()
    {
        Debug.Log("=== STARTING ASSEMBLY WORKFLOW ===");

        Debug.Log("PHASE 1: Spawning all parts at random locations...");
        yield return new WaitForSeconds(1f);

        Debug.Log("PHASE 2: Moving parts to final visible positions with specified scale and rotation...");

        yield return StartCoroutine(MoveToFinalTransform(part1, new Vector3(0.04f, 0f, 0.2f), Quaternion.Euler(0f, 0f, 0f), new Vector3(100f, 100f, 100f), 2f));
        yield return StartCoroutine(MoveToFinalTransform(part3, new Vector3(-0.6f, 0f, 0.7f), Quaternion.Euler(180f, -90f, 0f), new Vector3(100f, 100f, 100f), 2f));
        yield return StartCoroutine(MoveToFinalTransform(part2, new Vector3(-2f, 1f, 8f), Quaternion.Euler(90f, 90f, 0f), new Vector3(1000f, 1000f, 1000f), 2f));

        yield return new WaitForSeconds(0.5f);

        Debug.Log("PHASE 3: Playing animations for each part...");
        yield return StartCoroutine(PlayAnimationClips(animPart1, part1Animations));
        yield return StartCoroutine(PlayAnimationClips(animPart3, part3Animations));
        yield return StartCoroutine(PlayAnimationClips(animPart2, part2Animations));

        Debug.Log("PHASE 4: Assembly complete, waiting for user...");
        ShowPopup("Assembly Complete! All parts are in place.");
        yield return new WaitUntil(() => userClickedNext);
        userClickedNext = false;

        Debug.Log("=== ASSEMBLY WORKFLOW FINISHED ===");
    }

    Vector3 GetRandomSpawnPosition()
    {
        float range = 10f; // customize as needed
        return new Vector3(
            Random.Range(-range, range),
            Random.Range(0.5f, 3f),
            Random.Range(-range, range)
        );
    }

    Quaternion GetRandomRotation()
    {
        return Quaternion.Euler(
            Random.Range(0f, 360f),
            Random.Range(0f, 360f),
            Random.Range(0f, 360f)
        );
    }

    IEnumerator MoveToFinalTransform(GameObject part, Vector3 targetPos, Quaternion targetRot, Vector3 targetScale, float duration)
    {
        Vector3 startPos = part.transform.position;
        Quaternion startRot = part.transform.rotation;
        Vector3 startScale = part.transform.localScale;

        float elapsed = 0f;

        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.SmoothStep(0f, 1f, elapsed / duration);

            part.transform.position = Vector3.Lerp(startPos, targetPos, t);
            part.transform.rotation = Quaternion.Slerp(startRot, targetRot, t);
            part.transform.localScale = Vector3.Lerp(startScale, targetScale, t);

            yield return null;
        }

        part.transform.position = targetPos;
        part.transform.rotation = targetRot;
        part.transform.localScale = targetScale;
    }

    IEnumerator PlayAnimationClips(Animator animator, AnimationClip[] clips)
    {
        if (clips == null || clips.Length == 0)
            yield break;

        RuntimeAnimatorController originalController = animator.runtimeAnimatorController;
        AnimatorOverrideController overrideController = new AnimatorOverrideController(originalController);

        foreach (var clip in clips)
        {
            overrideController["BaseClip"] = clip; // replace "BaseClip" with the placeholder name in your animator
            animator.runtimeAnimatorController = overrideController;
            animator.Play("BaseStateName", 0, 0); // replace "BaseStateName" with your animator's base state name

            yield return new WaitForSeconds(clip.length);
        }

        animator.runtimeAnimatorController = originalController;
    }

    void ShowPopup(string message)
    {
        popupText.text = message;
        popupPanel.SetActive(true);
        Debug.Log($"Popup shown: {message}");
    }

    void OnNextButtonClicked()
    {
        Debug.Log("User clicked Next button");
        popupPanel.SetActive(false);
        userClickedNext = true;
    }
}
