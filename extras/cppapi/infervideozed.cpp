#include <iostream>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <math.h>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<boost/algorithm/string.hpp>
#include <experimental/string_view>

#include <cuda_runtime.h>
#include "../../csrc/engine.h"

using namespace std;
using namespace cv;
using namespace boost::algorithm;

int main(int argc, char *argv[]) {
    
	if (argc != 3) {
		cerr << "Usage: " << argv[0] << " engine.plan label_map.txt" << endl;
		return 1;
	}
    
       // convert label map
    cout << "Loading label map..." << endl;
    std::string line;
    std::vector<std::string> mylist;   
    try
    {
        std::ifstream f(argv[2]);

        if(!f)
        {
            std::cerr << "ERROR: Cannot open label text file!" << std::endl;
            exit(1);
        }

        while (std::getline(f,line))
        {
            mylist.push_back(line); 
            std::cout << mylist.back() << std::endl;  
        }
        
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Exception: '" << ex.what() << "'!" << std::endl;
        exit(1);
    }

	cout << "Loading engine..." << endl;
	auto engine = retinanet::Engine(argv[1]);
	VideoCapture cap(0);

	if (!cap.isOpened()){
		cerr << "Could not read " << argv[2] << endl;
		return 1;
	}
    
    // Set the video resolution to HD720 (2560*720)
    cap.set(CAP_PROP_FRAME_WIDTH, 1280*2);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);    
    cap.set(CAP_PROP_FPS, 60);

	Mat frame;
	Mat resized_frame;
	Mat inferred_frame;
	int count=1;
    
	auto inputSize = engine.getInputSize();
	// Create device buffers
	void *data_d, *scores_d, *boxes_d, *classes_d;
	auto num_det = engine.getMaxDetections();
	cudaMalloc(&data_d, 3 * inputSize[0] * inputSize[1] * sizeof(float));
	cudaMalloc(&scores_d, num_det * sizeof(float));
	cudaMalloc(&boxes_d, num_det * 4 * sizeof(float));
	cudaMalloc(&classes_d, num_det * sizeof(float));

	unique_ptr<float[]> scores(new float[num_det]);
	unique_ptr<float[]> boxes(new float[num_det * 4]);
	unique_ptr<float[]> classes(new float[num_det]);

	vector<float> mean {0.485, 0.456, 0.406};
	vector<float> std {0.229, 0.224, 0.225};

	vector<uint8_t> blues {0,63,127,191,255,0}; //colors for bonuding boxes
	vector<uint8_t> greens {0,255,191,127,63,0};
	vector<uint8_t> reds {191,255,0,0,63,127};

	int channels = 3;
	vector<float> img;
	vector<float> data (channels * inputSize[0] * inputSize[1]);
    
    //Begin fps counter
    long frameCounter = 0;
    std::time_t timeBegin = std::time(0);
    int tick = 0;
    std::string fps = "fps: ";
    
	while(1)
	{
		cap >> frame;
		if (frame.empty()){
			cout << "Finished inference!" << endl;
			break;
		}
        
        frame = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
		cv::resize(frame, resized_frame, Size(inputSize[0],   inputSize[1]));
		cv::Mat pixels;
		resized_frame.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);

		img.assign((float*)pixels.datastart, (float*)pixels.dataend);

		for (int c = 0; c < channels; c++) {
			for (int j = 0, hw = inputSize[0] * inputSize[1]; j < hw; j++) {
				data[c * hw + j] = (img[channels * j + 2 - c] - mean[c]) / std[c];
			}
		}

		// Copy image to device
        auto start = std::chrono::high_resolution_clock::now();
		size_t dataSize = data.size() * sizeof(float);
		cudaMemcpy(data_d, data.data(), dataSize, cudaMemcpyHostToDevice);

		//Do inference
		count++;
		vector<void *> buffers = { data_d, scores_d, boxes_d, classes_d };
		engine.infer(buffers);

		cudaMemcpy(scores.get(), scores_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
		cudaMemcpy(boxes.get(), boxes_d, sizeof(float) * num_det * 4, cudaMemcpyDeviceToHost);
		cudaMemcpy(classes.get(), classes_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
       
        //Calculate inference time
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        long long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
     
        std::stringstream strstream;
        strstream <<  "Inference time: " << milliseconds;
        std::string time=strstream.str();   

        
        // Get back the bounding boxes
		for (int i = 0; i < num_det; i++) {
			if (scores[i] >= 0.5f) {
				float x1 = boxes[i*4+0];
				float y1 = boxes[i*4+1];
				float x2 = boxes[i*4+2];
				float y2 = boxes[i*4+3];
				int cls=classes[i];
                auto score = round(scores[i]*100)/100;
    
                //Prepare class and score string
                std::stringstream ss;
                std::string classstr = mylist[classes[i]];
                trim(classstr);
                ss << classstr <<": " << score;
                std::string s = ss.str();
                
				// Draw bounding box, score and class on image
				cv::rectangle(resized_frame, Point(x1, y1), Point(x2, y2), cv::Scalar(blues[cls], greens[cls], reds[cls]));
                cv::putText(resized_frame, s, cv::Point(x1, y1 -4), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255),0.75, cv::LINE_AA); 
            }
		}
        
        frameCounter++; 
        std::time_t timeNow = std::time(0) - timeBegin;

        if (timeNow - tick >= 1)
        {
            fps="fps: ";
            tick++;
            std::stringstream strstream;
            strstream << frameCounter;
            std::string fpsres=strstream.str();   
            fps += fpsres;
            frameCounter = 0;

        } 
        
        cv::resize(resized_frame, inferred_frame, Size(640, 480));
        cv::putText(inferred_frame, time, cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255),0.75, cv::LINE_AA); 
        cv::putText(inferred_frame, fps, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255),0.75, cv::LINE_AA); 
        cv::imshow("Retinanet - Resnet18", inferred_frame);
        cv::waitKey(1);

	} 
	cap.release();
	cudaFree(data_d);
	cudaFree(scores_d);
	cudaFree(boxes_d);
	cudaFree(classes_d);
	return 0;
}
