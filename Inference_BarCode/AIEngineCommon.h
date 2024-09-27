#ifndef AIENGINECOMMON_H
#define AIENGINECOMMON_H

#include <vector>

#define AIENGINE_NO_ERROR 0


typedef struct CoordinateVOC 
{
    float xmin;  // ���Ͻǵ�x������
    float ymin;  // ���Ͻǵ�y������
    float xmax;  // ���½ǵ�x������
    float ymax;  // ���½ǵ�y������
} Coordinate_VOC;

typedef enum 
{
    PREP_TYPE_FULL_IMG_RESIZE = 1,   // ȫͼresize���������Ϊ����ͼƬ����Ҫʱ��ԭͼ������
    PREP_TYPE_ROI_DEP_ON_COORD = 0,      // ����ROI����ִ����ͼ���ϸ����Ƽ�������С���ó�������������С
} prep_type_t;

struct ImageInfo 
{
    Coordinate_VOC roi_coord;   // [����]ROI����
    void* img_data_pt;          // [����]ͼƬ�洢��ַ
    int img_height;             // [����]ͼ��߶�
    int img_width;              // [����]ͼ����
    prep_type_t prep_type;      // [����]ǰ��������
};

struct DetectionBBoxInfo 
{
    float xmin;   // ���߿����Ͻǵ�x������
    float ymin;   // ���߿����Ͻǵ�y������
    float xmax;   // ���߿����½ǵ�x������
    float ymax;   // ���߿����½ǵ�y������
    float score;  // ���Ŷ�
    int classID;  // ���ID
};


// ��¼ʱ����Ľṹ��
struct TimeStamps
{
    std::chrono::high_resolution_clock::time_point preprocess_start;
    std::chrono::high_resolution_clock::time_point preprocess_end;
    std::chrono::high_resolution_clock::time_point inference_start;
    std::chrono::high_resolution_clock::time_point inference_end;
    std::chrono::high_resolution_clock::time_point postprocess_start;
    std::chrono::high_resolution_clock::time_point postprocess_end;
};



//��һ������
int Inference(MNN::Interpreter* barcodeNet, MNN::Session* barcodeSession, MNN::Tensor* barcodeInputTensor, MNN::Tensor* barcodeOutputTensor, cv::Mat& image, std::vector<DetectionBBoxInfo>& det_results, TimeStamps& timestamps);

//�ڶ�������
int SecondInference(MNN::Interpreter* barcodeNet, MNN::Session* barcodeSession, MNN::Tensor* barcodeInputTensor, MNN::Tensor* barcodeOutputTensor, const cv::Mat& image, const std::vector<DetectionBBoxInfo>& det_results, TimeStamps& timestamps);

//�������ʱ��
void OutputElapsedTime(const TimeStamps& timestamps);

//ǰ����
int Input_Image(const ImageInfo* image_info, MNN::Tensor* inputTensor);

// ����
int Output_Detection(MNN::Tensor* outputTensor, std::vector<DetectionBBoxInfo>* det_results, const ImageInfo* image_info, MNN::Tensor* inputTensor);

//�Ǽ���ֵ�����㷨
void NonMaximumSuppression(std::vector<DetectionBBoxInfo>& boxes, float iouThreshold);

//IOU�㷨
float CalculateIOU(const DetectionBBoxInfo& bbox1, const DetectionBBoxInfo& bbox2);

// ���Ƽ�������Ŷ�
void DrawDetections(cv::Mat& img, const std::vector<DetectionBBoxInfo>& detections);

#endif // AIENGINECOMMON_H
