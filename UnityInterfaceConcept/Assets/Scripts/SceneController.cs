using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneController : MonoBehaviour
{
    private string _newSceneName;
    
    public void Quit()
    {
        Application.Quit();
    }

    public void OpenScene(string sceneName)
    {
        _newSceneName = sceneName;
        gameObject.GetComponent<UITransitionController>().BlackTransition(true, callOnEnd: LoadSc);
    }

    private void LoadSc()
    {
        SceneManager.LoadScene(_newSceneName);
    }
}
