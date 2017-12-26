//
// Created by dilin on 11/10/17.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include "BGSDetector.h"
#include "HOGDetector.h"

#define FPS 25

using namespace std;
using namespace cv;

int main(int argc, const char * argv[])
{
    if(argc<1)
    {
        cout << "Please specify an input video." << endl;
        return -1;
    }

    VideoCapture cap(argv[1]);
    Mat img;
    BGSDetector detector(30,
                         BGS_MOVING_AVERAGE,
                         false,
                         "/home/dilin/fyp/Optimization/Detector/pca_coeff.xml",
                         true);
//    HOGDetector detector;

    namedWindow("Detections");

    while(cap.isOpened())
    {
        cap >> img;

        if(!img.data)
            break;

        vector<Rect> detections = detector.detect(img);

        for(int i=0;i<detections.size();i++)
        {
            rectangle(img, detections[i].tl(), detections[i].br()
                    , cv::Scalar(0,255,0), 2);
        }

        imshow("Detections",img);

        if(waitKey(1)>0)
            break;
    }

    cout << "Training the detector..." << endl;
    detector.trainDetector();
    cout << "Training completed!" << endl;

    cap.release();
    cap.open(argv[1]);
    BGSDetector detector2(30,
                         BGS_MOVING_AVERAGE,
                         false,
                         "/home/dilin/fyp/Optimization/Detector/pca_coeff.xml",
                         false);


    while(cap.isOpened())
    {
        cap >> img;

        if(!img.data)
            break;

        vector<Rect> detections = detector2.detect(img);

        for(int i=0;i<detections.size();i++)
        {
            rectangle(img, detections[i].tl(), detections[i].br()
                    , cv::Scalar(0,255,0), 2);
        }

        imshow("Detections",img);

        if(waitKey(1000/FPS)>0)
            break;
    }

    return 0;
}

