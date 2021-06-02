using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovingBackground : MonoBehaviour
{
    [SerializeField] private float speed;
    [SerializeField] private float xLimit;
    // Update is called once per frame
    void Update()
    {
        transform.Translate(new Vector3(-1, 0) * (Time.deltaTime * speed));
        if (transform.position.x <= xLimit)
            transform.position = new Vector3(-xLimit, 0);
    }
}
