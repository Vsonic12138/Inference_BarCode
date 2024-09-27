#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"

#include "AIEngineCommon.h"  // 包含了PDF中描述的AIEngine相关定义




int main()
{
    // 在主函数中声明 timestamps 变量
    TimeStamps timestamps;

    // 加载条码检测模型
    std::shared_ptr<MNN::Interpreter> barcodeNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile("BarCode_Inference_model.mnn"));

    // 配置后端
    MNN::BackendConfig backendConfig;
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;  // 正常内存模式
    backendConfig.power = MNN::BackendConfig::Power_Normal;    // 正常功耗模式
    backendConfig.precision = MNN::BackendConfig::Precision_Normal; // 正常精度模式

    // 配置调度配置结构体，并设置后端类型和并发数
    MNN::ScheduleConfig barcodeConfig;
    //barcodeConfig.type = MNN_FORWARD_OPENCL; // 使用GPU-OPENCL进行推理
    barcodeConfig.type = MNN_FORWARD_CPU;   // 使用CPU进行推理
    barcodeConfig.numThread = 4; // 设置并发数和线程数
    barcodeConfig.backendConfig = &backendConfig; // 传递后端配置

    // 创建Session并传递配置
    MNN::Session* barcodeSession = barcodeNet->createSession(barcodeConfig);

    // 读取输入图像
    cv::Mat image = cv::imread("222.png");
    if (image.empty())
    {
        std::cerr << "Failed to read image!" << std::endl;
        return -1;
    }

    // 获取输入张量和输出张量
    MNN::Tensor* barcodeInputTensor = barcodeNet->getSessionInput(barcodeSession, nullptr);
    auto barcodeOutputTensor = barcodeNet->getSessionOutput(barcodeSession, "output");

    // 执行第一次推理
    std::vector<DetectionBBoxInfo> det_results;
    if (Inference(barcodeNet.get(), barcodeSession, barcodeInputTensor, barcodeOutputTensor, image, det_results, timestamps) != 0)
    {
        std::cerr << "Inference failed!" << std::endl;
        return -1;
    }

    // 绘制并展示第一次推理的检测框
    cv::Mat detectionImage = image.clone();
    DrawDetections(detectionImage, det_results);
    cv::imshow("Barcode Detection - Full Image", detectionImage);
    // cv::waitKey(0);

    // 执行二次推理
    if (SecondInference(barcodeNet.get(), barcodeSession, barcodeInputTensor, barcodeOutputTensor, image, det_results, timestamps) != 0)
    {
        std::cerr << "Second Inference failed!" << std::endl;
        return -1;
    }

    OutputElapsedTime(timestamps);  // 输出耗时信息

    // 释放资源
    barcodeNet->releaseSession(barcodeSession);

    return 0;
}