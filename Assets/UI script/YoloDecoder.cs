using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;

public static class YoloDecoder
{
    // YOUR 8 CUSTOM CLASSES
    public static readonly string[] classNames = {
        "Tools", "Housing", "Big Screw", "Small Screw",
        "V-Lock Plate", "Plunger", "Helical Spring", "Cover"
    };

    public class Detection
    {
        public int cls;
        public float score;
        public Rect rect;

        public Detection(int cls, float score, Rect rect)
        {
            this.cls = cls;
            this.score = score;
            this.rect = rect;
        }
    }

    public static List<Detection> Decode(Tensor<float> t, float confThresh)
    {
        var dets = new List<Detection>();

        // t.shape == [1, 12, 8400]
        int C = t.shape[1];     // 12
        int N = t.shape[2];     // 8400

        for (int i = 0; i < N; i++)
        {
            float cx = t[0, 0, i];
            float cy = t[0, 1, i];
            float w = t[0, 2, i];
            float h = t[0, 3, i];

            float objConf = t[0, 4, i];
            if (objConf < confThresh)
                continue;

            // find top class
            int bestCls = -1;
            float bestScore = 0f;

            // class scores start at channel 5
            for (int c = 0; c < classNames.Length; c++)
            {
                float p = t[0, 5 + c, i];
                if (p > bestScore)
                {
                    bestScore = p;
                    bestCls = c;
                }
            }

            float finalScore = objConf * bestScore;
            if (finalScore < confThresh)
                continue;

            // YOLO coords are normalized: convert to rect
            float x = cx - w * 0.5f;
            float y = cy - h * 0.5f;
            Rect r = new Rect(x, y, w, h);

            dets.Add(new Detection(bestCls, finalScore, r));
        }

        return dets;
    }
}
