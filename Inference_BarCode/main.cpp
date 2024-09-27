#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"

#include "AIEngineCommon.h"  // ������PDF��������AIEngine��ض���




int main()
{
    // �������������� timestamps ����
    TimeStamps timestamps;

    // ����������ģ��
    std::shared_ptr<MNN::Interpreter> barcodeNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile("BarCode_Inference_model.mnn"));

    // ���ú��
    MNN::BackendConfig backendConfig;
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;  // �����ڴ�ģʽ
    backendConfig.power = MNN::BackendConfig::Power_Normal;    // ��������ģʽ
    backendConfig.precision = MNN::BackendConfig::Precision_Normal; // ��������ģʽ

    // ���õ������ýṹ�壬�����ú�����ͺͲ�����
    MNN::ScheduleConfig barcodeConfig;
    //barcodeConfig.type = MNN_FORWARD_OPENCL; // ʹ��GPU-OPENCL��������
    barcodeConfig.type = MNN_FORWARD_CPU;   // ʹ��CPU��������
    barcodeConfig.numThread = 4; // ���ò��������߳���
    barcodeConfig.backendConfig = &backendConfig; // ���ݺ������

    // ����Session����������
    MNN::Session* barcodeSession = barcodeNet->createSession(barcodeConfig);

    // ��ȡ����ͼ��
    cv::Mat image = cv::imread("222.png");
    if (image.empty())
    {
        std::cerr << "Failed to read image!" << std::endl;
        return -1;
    }

    // ��ȡ�����������������
    MNN::Tensor* barcodeInputTensor = barcodeNet->getSessionInput(barcodeSession, nullptr);
    auto barcodeOutputTensor = barcodeNet->getSessionOutput(barcodeSession, "output");

    // ִ�е�һ������
    std::vector<DetectionBBoxInfo> det_results;
    if (Inference(barcodeNet.get(), barcodeSession, barcodeInputTensor, barcodeOutputTensor, image, det_results, timestamps) != 0)
    {
        std::cerr << "Inference failed!" << std::endl;
        return -1;
    }

    // ���Ʋ�չʾ��һ������ļ���
    cv::Mat detectionImage = image.clone();
    DrawDetections(detectionImage, det_results);
    cv::imshow("Barcode Detection - Full Image", detectionImage);
    // cv::waitKey(0);

    // ִ�ж�������
    if (SecondInference(barcodeNet.get(), barcodeSession, barcodeInputTensor, barcodeOutputTensor, image, det_results, timestamps) != 0)
    {
        std::cerr << "Second Inference failed!" << std::endl;
        return -1;
    }

    OutputElapsedTime(timestamps);  // �����ʱ��Ϣ

    // �ͷ���Դ
    barcodeNet->releaseSession(barcodeSession);

    return 0;
}