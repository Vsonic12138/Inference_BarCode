#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"

#include "AIEngineCommon.h"  // 包含了PDF中描述的AIEngine相关定义

TimeStamps timestamps;  // 全局声明，或者在函数内声明


// 输出处理时间的函数
// 输出处理时间的函数
void OutputElapsedTime(const TimeStamps& timestamps)
{
    // 计算耗时并转换为毫秒（带有小数点后两位）
    auto preprocess_time = std::chrono::duration<double, std::milli>(timestamps.preprocess_end - timestamps.preprocess_start).count();
    auto inference_time = std::chrono::duration<double, std::milli>(timestamps.inference_end - timestamps.inference_start).count();
    auto postprocess_time = std::chrono::duration<double, std::milli>(timestamps.postprocess_end - timestamps.postprocess_start).count();

    // 输出带有小数点后两位的时间
    std::cout << std::fixed << std::setprecision(2); // 设置小数点精度为2位
    std::cout << "Preprocessing Time: " << preprocess_time << " ms" << std::endl;
    std::cout << "Inference Time: " << inference_time << " ms" << std::endl;
    std::cout << "Postprocessing Time: " << postprocess_time << " ms" << std::endl;
}

//IOU算法
float CalculateIOU(const DetectionBBoxInfo& bbox1, const DetectionBBoxInfo& bbox2) 
{
    // 计算交集坐标
    int inter_xmin = std::max(bbox1.xmin, bbox2.xmin);
    int inter_ymin = std::max(bbox1.ymin, bbox2.ymin);
    int inter_xmax = std::min(bbox1.xmax, bbox2.xmax);
    int inter_ymax = std::min(bbox1.ymax, bbox2.ymax);

    // 计算交集面积
    int inter_width = std::max(0, inter_xmax - inter_xmin);
    int inter_height = std::max(0, inter_ymax - inter_ymin);
    int inter_area = inter_width * inter_height;

    // 计算每个框的面积
    int bbox1_area = (bbox1.xmax - bbox1.xmin) * (bbox1.ymax - bbox1.ymin);
    int bbox2_area = (bbox2.xmax - bbox2.xmin) * (bbox2.ymax - bbox2.ymin);

    // 计算并集面积
    int union_area = bbox1_area + bbox2_area - inter_area;

    // 计算IOU
    float iou = static_cast<float>(inter_area) / union_area;
    return iou;
}

void NonMaximumSuppression(std::vector<DetectionBBoxInfo>& boxes, float iouThreshold)
{
    // 按置信度从高到低排序
    std::sort(boxes.begin(), boxes.end(), [](const DetectionBBoxInfo& a, const DetectionBBoxInfo& b)
        {
            return a.score > b.score;
        });

    // 标记要抑制的框
    std::vector<char> suppressed(boxes.size(), 0); // 使用char类型避免std::vector<bool>的特殊行为

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (suppressed[i]) continue; // 如果已经被抑制，跳过

        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (CalculateIOU(boxes[i], boxes[j]) > iouThreshold)
            {
                suppressed[j] = 1; // 抑制当前框
            }
        }
    }

    // 删除被抑制的框
    auto it = std::remove_if(boxes.begin(), boxes.end(), [&](const DetectionBBoxInfo& box)
        {
            size_t idx = &box - &boxes[0]; // 计算当前元素的索引
            return suppressed[idx]; // 根据index来判断是否删除
        });

    boxes.erase(it, boxes.end());

    // 调试输出经过NMS筛选后的框数量
    //std::cerr << "经过NMS过滤后的二维码数量: " << boxes.size() << std::endl;
}



// 绘制检测框和置信度
void DrawDetections(cv::Mat& img, const std::vector<DetectionBBoxInfo>& detections)
{
    for (const auto& bbox : detections)
    {
        if (bbox.score > 0.6) // 只显示置信度大于0.6的结果
        {
            // 绘制检测框
            cv::rectangle(img, cv::Point(bbox.xmin, bbox.ymin), cv::Point(bbox.xmax, bbox.ymax), cv::Scalar(0, 255, 0), 2);

            // 绘制置信度文本
            std::string label = cv::format("%.2f", bbox.score);
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.5;
            int thickness = 1;
            cv::Scalar color = cv::Scalar(0, 255, 0); // 文字颜色
            int baseline = 0;

            // 计算文本框的大小
            cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);
            // 文本放在框的左下角
            cv::Point textOrg(bbox.xmin, bbox.ymax + textSize.height);

            // 确保文本不会超出图像边界
            if (textOrg.y + textSize.height > img.rows)
            {
                textOrg.y = bbox.ymax - baseline;
            }

            // 绘制文本
            cv::putText(img, label, textOrg, fontFace, fontScale, color, thickness, 8);
        }
    }
}



//前处理
int Input_Image(const ImageInfo* image_info, MNN::Tensor* inputTensor) 
{
    if (!image_info || !inputTensor) {
        return -1;  // 错误：输入的参数为空
    }

    timestamps.preprocess_start = std::chrono::high_resolution_clock::now(); // 前处理开始时间戳

    // 获取输入张量的维度
    auto tensorDims = inputTensor->shape();
    int tensorHeight = tensorDims[2];
    int tensorWidth = tensorDims[3];

    // 图像数据转换为Mat
    cv::Mat imgMat(image_info->img_height, image_info->img_width, CV_8UC1, image_info->img_data_pt);

    // 执行前处理，根据 prep_type 选择不同的逻辑
    cv::Mat preprocessedImg;
    switch (image_info->prep_type) {
    case PREP_TYPE_FULL_IMG_RESIZE: {
        // 全图resize
        cv::resize(imgMat, preprocessedImg, cv::Size(tensorWidth, tensorHeight));
        break;
    }
    case PREP_TYPE_ROI_DEP_ON_COORD: {
        // 不进行resize，直接使用原图
        preprocessedImg = imgMat;
        break;
    }
    default: {
        // 默认不进行任何处理
        preprocessedImg = imgMat;
        break;
    }
    }

    // 根据MNN张量的格式（例如NCHW或NHWC）创建临时张量
    MNN::Tensor::DimensionType dimType = inputTensor->getDimensionType();
    MNN::Tensor* tempTensor = nullptr;
    if (dimType == MNN::Tensor::CAFFE) { // NCHW格式
        tempTensor = new MNN::Tensor(inputTensor, MNN::Tensor::CAFFE);
    }
    else if (dimType == MNN::Tensor::TENSORFLOW) { // NHWC格式
        tempTensor = new MNN::Tensor(inputTensor, MNN::Tensor::TENSORFLOW);
    }
    else {
        return -1;  // 错误：不支持的张量格式
    }

    // 填充数据到临时张量并进行归一化
    auto data = tempTensor->host<float>();
    int dataSize = tensorHeight * tensorWidth;
    if (preprocessedImg.isContinuous()) {
        for (int i = 0; i < dataSize; ++i) {
            data[i] = static_cast<float>(preprocessedImg.data[i]) / 255.0f;
        }
    }
    else {
        for (int i = 0; i < tensorHeight; ++i) {
            for (int j = 0; j < tensorWidth; ++j) {
                data[i * tensorWidth + j] = static_cast<float>(preprocessedImg.at<uint8_t>(i, j)) / 255.0f;
            }
        }
    }

    // 将临时张量的数据拷贝到输入张量
    inputTensor->copyFromHostTensor(tempTensor);
    delete tempTensor;

    timestamps.preprocess_end = std::chrono::high_resolution_clock::now(); // 前处理结束时间戳

    return 0;  // 成功
}




//第一次推理
int Inference(MNN::Interpreter* barcodeNet, MNN::Session* barcodeSession, MNN::Tensor* barcodeInputTensor, MNN::Tensor* barcodeOutputTensor, cv::Mat& image, std::vector<DetectionBBoxInfo>& det_results, TimeStamps& timestamps)
{
    // 转换为灰度图像（1通道）
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // 创建ImageInfo结构体用于前处理
    ImageInfo imageInfo;
    imageInfo.img_data_pt = grayImage.data;
    imageInfo.img_height = grayImage.rows;
    imageInfo.img_width = grayImage.cols;
    imageInfo.prep_type = PREP_TYPE_FULL_IMG_RESIZE;  // 使用全图resize的预处理类型

    // 输入图像数据到模型
    if (Input_Image(&imageInfo, barcodeInputTensor) != 0)
    {
        std::cerr << "Failed to input image data!" << std::endl;
        return -1;
    }

    // 记录推理开始时间戳
    timestamps.inference_start = std::chrono::high_resolution_clock::now();

    // 运行推理
    barcodeNet->runSession(barcodeSession);

    // 记录推理结束时间戳
    timestamps.inference_end = std::chrono::high_resolution_clock::now();

    // 获取推理的输出
    auto nchwTensor = new MNN::Tensor(barcodeOutputTensor, MNN::Tensor::CAFFE);
    barcodeOutputTensor->copyToHostTensor(nchwTensor);

    // 处理推理结果
    if (Output_Detection(nchwTensor, &det_results, &imageInfo, barcodeInputTensor) != 0)
    {
        std::cerr << "Failed to output detection results!" << std::endl;
        delete nchwTensor;
        return -1;
    }

    delete nchwTensor;
    return 0;
}



//第二次推理
int SecondInference(MNN::Interpreter* barcodeNet, MNN::Session* barcodeSession, MNN::Tensor* barcodeInputTensor, MNN::Tensor* barcodeOutputTensor, const cv::Mat& image, const std::vector<DetectionBBoxInfo>& det_results, TimeStamps& timestamps)
{
    // 对NMS过滤后的每个检测框区域进行裁剪并再次推理
    for (const auto& bbox : det_results)
    {
        if (bbox.score <= 0.5)
        {
            continue;
        }

        // 裁剪NMS过滤后的区域
        cv::Rect roiRect(cv::Point(bbox.xmin, bbox.ymin), cv::Point(bbox.xmax, bbox.ymax));

        // 确保裁剪区域在原图范围内
        if (roiRect.x < 0) roiRect.x = 0;
        if (roiRect.y < 0) roiRect.y = 0;
        if (roiRect.x + roiRect.width > image.cols) roiRect.width = image.cols - roiRect.x;
        if (roiRect.y + roiRect.height > image.rows) roiRect.height = image.rows - roiRect.y;

        cv::Mat roiImage = image(roiRect);

        // 检查裁剪后的区域尺寸是否超过输入张量的大小
        if (roiImage.cols > barcodeInputTensor->width() || roiImage.rows > barcodeInputTensor->height())
        {
            std::cerr << "Skipping ROI inference due to size constraints." << std::endl;
            continue;
        }

        // 更新ImageInfo为ROI信息
        ImageInfo roiInfo;
        roiInfo.img_data_pt = roiImage.data;
        roiInfo.img_height = roiImage.rows;
        roiInfo.img_width = roiImage.cols;
        roiInfo.prep_type = PREP_TYPE_ROI_DEP_ON_COORD;

        // 进行ROI区域推理
        if (Input_Image(&roiInfo, barcodeInputTensor) != 0)
        {
            std::cerr << "Failed to input ROI image data!" << std::endl;
            return -1;
        }

        // 执行推理
        std::vector<DetectionBBoxInfo> roi_results;
        if (Inference(barcodeNet, barcodeSession, barcodeInputTensor, barcodeOutputTensor, roiImage, roi_results, timestamps) != 0)
        {
            std::cerr << "ROI Inference failed!" << std::endl;
            return -1;
        }

        // 绘制并展示ROI推理结果
        cv::Mat roiResultImage = image.clone();
        cv::rectangle(roiResultImage, roiRect, cv::Scalar(0, 255, 0), 2);

        if (!roi_results.empty())
        {
            DrawDetections(roiResultImage, roi_results);
            cv::imshow("Barcode Detection - ROI", roiResultImage);
            cv::waitKey(0);
        }
    }

    return 0;
}



//后处理
int Output_Detection(MNN::Tensor* outputTensor, std::vector<DetectionBBoxInfo>* det_results, const ImageInfo* image_info, MNN::Tensor* inputTensor)
{
    if (!outputTensor || !det_results) {
        return -1;  // 错误：输入参数为空
    }

    // 计算全图resize的比例因子
    float widthScale = static_cast<float>(image_info->img_width) / inputTensor->width();
    float heightScale = static_cast<float>(image_info->img_height) / inputTensor->height();

    // 从输出张量中解析检测框信息
    auto outputHost = outputTensor->host<float>();  // 输出是float类型
    int numDetections = outputTensor->shape()[1];   // 输出张量的第二个维度表示检测框数量


    for (int i = 0; i < numDetections; ++i)
    {

        DetectionBBoxInfo bbox;
        float xc = outputHost[i * 6 + 0];  // 假设输出为[xmin, ymin, xmax, ymax, score]
        float yc = outputHost[i * 6 + 1];
        float w = outputHost[i * 6 + 2];
        float h = outputHost[i * 6 + 3];
        bbox.score = outputHost[i * 6 + 4];


        // 转换为原始图像坐标系
        bbox.xmin = (xc - (w / 2)) * widthScale;
        bbox.ymin = (yc - (h / 2)) * heightScale;
        bbox.xmax = (xc + (w / 2)) * widthScale;
        bbox.ymax = (yc + (h / 2)) * heightScale;

        // 将转换后的检测框添加到结果集中
        det_results->emplace_back(bbox);

    }


    // 使用NMS进行过滤，设定IOU阈值
    // 非极大值抑制过滤
    float iouThreshold = 0.6;
    NonMaximumSuppression(*det_results, iouThreshold);

    return 0;  // 成功
}



