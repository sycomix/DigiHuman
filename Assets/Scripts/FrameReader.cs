using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;

[Serializable] 
public struct BodyPart
{
    public float x;
    public float y;
    public float z;
    public float visibility;
}


[Serializable]
public class PoseJson
{
    public BodyPart[] predictions;
    public float width;
    public float height;
}

[Serializable] 
public struct BodyPartVector
{
    public Vector3 position;
    public float visibility;
}

[Serializable]
public class PoseJsonVector
{
    public BodyPartVector[] predictions;
    public float width;
    public float height;
}

public class FrameReader : MonoBehaviour
{
    public VideoPlayer videoPlayer;
    public TextAsset jsonTest;

    public float fraction = 1.2f;
    public float fractionX = 1.2f;
    public float fractionY = 1.2f;
    public float fractionZ = 1.2f;

    public float nextFrameTime = 0.1f;
    public PoseJson poseJson;
    public PoseJsonVector poseJsonVector;
    public PoseJsonVector poseJsonVectorNew;

    private void Awake()
    {
        poseJson = GetBodyParts(jsonTest.text);
        poseJsonVector = GetBodyPartsVector(jsonTest.text);
    }

    private string path = "C:\\Danial\\Projects\\Danial\\DigiHuman\\Backend\\json\\";
    private int index = 0;
    private float timer = 0;
    private void Update()
    {
        timer += Time.deltaTime;
        if (timer > nextFrameTime)
        {
            try
            {
                StreamReader reader = new StreamReader(path + "" + index + ".json");
                poseJsonVector = poseJsonVectorNew;
                poseJsonVectorNew = GetBodyPartsVector(reader.ReadToEnd());
                timer = 0;
                videoPlayer.frame = index;
                videoPlayer.Play();
                videoPlayer.Pause();
                reader.Close();
            }
            catch (FileNotFoundException e)
            {
                
            }
            index += 1;
        }
        else
        {
            try
            {
                for (int i = 0; i < poseJsonVector.predictions.Length; i++)
                {
                    poseJsonVector.predictions[i].position = Vector3.Lerp(
                        poseJsonVector.predictions[i].position, poseJsonVectorNew.predictions[i].position,
                        timer / nextFrameTime);
                }
            }
            catch (Exception e)
            {

            }
        }
    }
    private void Start() {
        videoPlayer.Prepare();
    }
    
    
    public PoseJson GetBodyParts(string text)
    {
        return JsonUtility.FromJson<PoseJson>(text);
    }
    public PoseJsonVector GetBodyPartsVector(string text)
    {
        
        PoseJson poseJson = JsonUtility.FromJson<PoseJson>(text);
        int len = poseJson.predictions.Length;
        PoseJsonVector poseJsonVector = new PoseJsonVector();
        poseJsonVector.predictions = new BodyPartVector[len];

        for (int i = 0; i < len; i++)
        {
            poseJsonVector.predictions[i].position = fraction * new Vector3(-poseJson.predictions[i].x*fractionX,
                -poseJson.predictions[i].y*fractionY,poseJson.predictions[i].z*fractionZ);
            poseJsonVector.predictions[i].visibility = poseJson.predictions[i].visibility;
            
        }

        return poseJsonVector;
    }
}
