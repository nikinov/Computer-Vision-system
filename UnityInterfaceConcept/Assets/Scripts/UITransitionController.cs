using System;
using System.Collections;
using System.Collections.Generic;
using DG.Tweening;
using UnityEngine;

public class UITransitionController : MonoBehaviour
{
    [SerializeField] private float fadeTime;
    [SerializeField] private int startPanel;
    
    [SerializeField] private CanvasGroup blackPanel;
    [SerializeField] private List<CanvasGroup> mainPanels;

    private int _currentPanel;
    private bool _panelSwaEnable = true;
    
    // Start is called before the first frame update
    void Start()
    {
        _currentPanel = startPanel;
        BlackTransition(false);

        for (int i = 0; i < mainPanels.Count; i++)
        {
            if (i != startPanel)
                PanelChange(mainPanels[i], 0, 0);
        }
    }

    public void Next()
    {
        StartCoroutine(WaitForFirstPanel(true));
    }

    public void Back()
    {
        StartCoroutine(WaitForFirstPanel(false));
    }
    
    public void Skip(int panelIdx)
    {
        StartCoroutine(WaitForFirstPanel(false, skip: panelIdx));
    }

    public void BlackTransition(bool mode, Action callOnEnd=null)
    {
        if (mode)
            PanelChange(blackPanel, 1, callOnEnd: callOnEnd);
        else
            PanelChange(blackPanel, 0, callOnEnd: callOnEnd);
    }

    void PanelChange(CanvasGroup panel, int fadeEnd, float fadeT=-1f, Action callOnEnd=null)
    {
        if (fadeT == -1f)
            fadeT = fadeTime;
        if (fadeEnd > 0 && !panel.gameObject.activeSelf)
            panel.gameObject.SetActive(true);
        if (fadeEnd == 0)
            StartCoroutine(WaitForDisable(panel.gameObject, fadeT, callOnEnd));
        panel.DOFade(fadeEnd, fadeT);
    }

    IEnumerator WaitForDisable(GameObject obj, float waitTime, Action callOnEnd)
    {
        yield return new WaitForSeconds(waitTime);
        obj.SetActive(false);
        if (callOnEnd != null)
            callOnEnd.DynamicInvoke();
    }

    IEnumerator WaitForFirstPanel(bool add, int skip=-1)
    {
        if (_panelSwaEnable)
        {
            _panelSwaEnable = false;
            PanelChange(mainPanels[_currentPanel], 0);
            yield return new WaitForSeconds(fadeTime);
            if (skip != -1)
                _currentPanel = skip;
            else if (add)
                _currentPanel += 1;
            else
                _currentPanel -= 1;
            PanelChange(mainPanels[_currentPanel], 1);
            _panelSwaEnable = true;
        }
    }
}
