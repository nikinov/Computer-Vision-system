using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;

public class TypeSimController : MonoBehaviour
{
    [SerializeField] private float typeSpeed;
    [SerializeField] private float delay;
    [SerializeField] private bool repeatOnStart;
    [SerializeField] private TextMeshProUGUI typeText;
    [SerializeField] private List<string> messages;

    private int _currentMessage;

    private void Start()
    {
        if (repeatOnStart)
            StartCoroutine(waiteForTyping(messages[_currentMessage]));
    }

    IEnumerator waiteForTyping(string message)
    {
        foreach (char c in message)
        {
            typeText.text = typeText.text.Insert(typeText.text.Length - 1, c.ToString());
            yield return new WaitForSeconds(typeSpeed);
        }
        yield return new WaitForSeconds(delay);
        foreach (char c in typeText.text)
        {
            if (typeText.text.Length - 1 == 0)
                continue;
            typeText.text = typeText.text.Remove(typeText.text.Length - 2, 1);
            yield return new WaitForSeconds(typeSpeed);
        }
        yield return new WaitForSeconds(delay);
        
        if (_currentMessage < messages.Count - 1)
            _currentMessage += 1;
        else
            _currentMessage = 0;
        StartCoroutine(waiteForTyping(messages[_currentMessage]));
    }
}
