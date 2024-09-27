#ifndef AIENGINECOMMON_H
#define AIENGINECOMMON_H

#include <vector>

#define AIENGINE_NO_ERROR 0


typedef struct CoordinateVOC 
{
    float xmin;  // 左上角点x轴坐标
    float ymin;  // 左上角点y轴坐标
    float xmax;  // 右下角点x轴坐标
    float ymax;  // 右下角点y轴坐标
} Coordinate_VOC;

typedef enum 
{
    PREP_TYPE_FULL_IMG_RESIZE = 1,   // 全图resize，检测区域为整张图片，必要时对原图做缩放
    PREP_TYPE_ROI_DEP_ON_COORD = 0,      // 根据ROI坐标执行挖图，严格限制检测区域大小不得超过输入张量大小
} prep_type_t;

struct ImageInfo 
{
    Coordinate_VOC roi_coord;   // [输入]ROI坐标
    void* img_data_pt;          // [输入]图片存储地址
    int img_height;             // [输入]图像高度
    int img_width;              // [输入]图像宽度
    prep_type_t prep_type;      // [输入]前处理类型
};

struct DetectionBBoxInfo 
{
    float xmin;   // 检测边框左上角点x轴坐标
    float ymin;   // 检测边框左上角点y轴坐标
    float xmax;   // 检测边框右下角点x轴坐标
    float ymax;   // 检测边框右下角点y轴坐标
    float score;  // 置信度
    int classID;  // 类别ID
};


// 记录时间戳的结构体
struct TimeStamps
{
    std::chrono::high_resolution_clock::time_point preprocess_start;
    std::chrono::high_resolution_clock::time_point preprocess_end;
    std::chrono::high_resolution_clock::time_point inference_start;
    std::chrono::high_resolution_clock::time_point inference_end;
    std::chrono::high_resolution_clock::time_point postprocess_start;
    std::chrono::high_resolution_clock::time_point postprocess_end;
};



//第一次推理
int Inference(MNN::Interpreter* barcodeNet, MNN::Session* barcodeSession, MNN::Tensor* barcodeInputTensor, MNN::Tensor* barcodeOutputTensor, cv::Mat& image, std::vector<DetectionBBoxInfo>& det_results, TimeStamps& timestamps);

//第二次推理
int SecondInference(MNN::Interpreter* barcodeNet, MNN::Session* barcodeSession, MNN::Tensor* barcodeInputTensor, MNN::Tensor* barcodeOutputTensor, const cv::Mat& image, const std::vector<DetectionBBoxInfo>& det_results, TimeStamps& timestamps);

//输出推理时间
void OutputElapsedTime(const TimeStamps& timestamps);

//前处理
int Input_Image(const ImageInfo* image_info, MNN::Tensor* inputTensor);

// 后处理
int Output_Detection(MNN::Tensor* outputTensor, std::vector<DetectionBBoxInfo>* det_results, const ImageInfo* image_info, MNN::Tensor* inputTensor);

//非极大值抑制算法
void NonMaximumSuppression(std::vector<DetectionBBoxInfo>& boxes, float iouThreshold);

//IOU算法
float CalculateIOU(const DetectionBBoxInfo& bbox1, const DetectionBBoxInfo& bbox2);

// 绘制检测框和置信度
void DrawDetections(cv::Mat& img, const std::vector<DetectionBBoxInfo>& detections);

#endif // AIENGINECOMMON_H
