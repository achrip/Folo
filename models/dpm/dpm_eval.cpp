#include <opencv2/opencv.hpp>
#include <opencv2/dpm.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace cv::dpm;
using namespace std;

vector<string> loadImagesFromDirectory(const string& directoryPath) {
    vector<string> filenames;
    glob(directoryPath + "/*.png", filenames, false); // Adjust the extension as needed
    return filenames;
}

void evaluateDPM(const vector<string>& hasPersonImages, 
                 const vector<string>& noPersonImages, 
                 const Ptr<DPMDetector>& detector) {

    int truePositives = 0;
    int falsePositives = 0;
    int trueNegatives = 0;
    int falseNegatives = 0;

    for (const string& filename : hasPersonImages) {
        Mat image = imread(filename);
        if (image.empty()) {
            cerr << "Could not load image: " << filename << endl;
            continue;
        }

        vector<DPMDetector::ObjectDetection> detections;
        detector->detect(image, detections);

        if (!detections.empty()) {
            truePositives++;
        } else {
            falseNegatives++;
        }
    }

    for (const string& filename : noPersonImages) {
        Mat image = imread(filename);
        if (image.empty()) {
            cerr << "Could not load image: " << filename << endl;
            continue;
        }

        vector<DPMDetector::ObjectDetection> detections;
        detector->detect(image, detections);

        if (!detections.empty()) {
            falsePositives++;
        } else {
            trueNegatives++;
        }
    }

    float precision = static_cast<float>(truePositives) / (truePositives + falsePositives);
    float recall = static_cast<float>(truePositives) / (truePositives + falseNegatives);
    float accuracy = static_cast<float>(truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives);

    cout << "Precision: " << precision << endl;
    cout << "Recall: " << recall << endl;
    cout << "Accuracy: " << accuracy << endl;
    cout << "True Positive: " << truePositives << endl; 
    cout << "False Positive: " << falsePositives << endl; 
    cout << "True Negative: " << trueNegatives << endl; 
    cout << "False Negative: " << falseNegatives << endl; 
}

int main() {
    string hasPersonDir = "../../dataset/has_person/"; 
    string noPersonDir = "../../dataset/no_person/";

    vector<string> hasPersonImages = loadImagesFromDirectory(hasPersonDir);
    vector<string> noPersonImages = loadImagesFromDirectory(noPersonDir);

    // Load your trained DPM model using OpenCV
    Ptr<DPMDetector> detector = DPMDetector::create(vector<string>(1, "../inriaperson.xml"));

    evaluateDPM(hasPersonImages, noPersonImages, detector);

    return 0;
}
