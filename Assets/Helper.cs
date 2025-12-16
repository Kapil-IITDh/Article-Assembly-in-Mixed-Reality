using UnityEngine;
using UnityEngine.EventSystems;
using System;

public class HybridClickHelper : MonoBehaviour, IPointerClickHandler
{
    private Action callback;

    public void Init(Action cb)
    {
        callback = cb;
    }

    public void OnPointerClick(PointerEventData eventData)
    {
        callback?.Invoke();
    }
}
