#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"

#include "AIEngineCommon.h"  // ������PDF��������AIEngine��ض���

TimeStamps timestamps;  // ȫ�������������ں���������


// �������ʱ��ĺ���
// �������ʱ��ĺ���
void OutputElapsedTime(const TimeStamps& timestamps)
{
    // �����ʱ��ת��Ϊ���루����С�������λ��
    auto preprocess_time = std::chrono::duration<double, std::milli>(timestamps.preprocess_end - timestamps.preprocess_start).count();
    auto inference_time = std::chrono::duration<double, std::milli>(timestamps.inference_end - timestamps.inference_start).count();
    auto postprocess_time = std::chrono::duration<double, std::milli>(timestamps.postprocess_end - timestamps.postprocess_start).count();

    // �������С�������λ��ʱ��
    std::cout << std::fixed << std::setprecision(2); // ����С���㾫��Ϊ2λ
    std::cout << "Preprocessing Time: " << preprocess_time << " ms" << std::endl;
    std::cout << "Inference Time: " << inference_time << " ms" << std::endl;
    std::cout << "Postprocessing Time: " << postprocess_time << " ms" << std::endl;
}

//IOU�㷨
float CalculateIOU(const DetectionBBoxInfo& bbox1, const DetectionBBoxInfo& bbox2) 
{
    // ���㽻������
    int inter_xmin = std::max(bbox1.xmin, bbox2.xmin);
    int inter_ymin = std::max(bbox1.ymin, bbox2.ymin);
    int inter_xmax = std::min(bbox1.xmax, bbox2.xmax);
    int inter_ymax = std::min(bbox1.ymax, bbox2.ymax);

    // ���㽻�����
    int inter_width = std::max(0, inter_xmax - inter_xmin);
    int inter_height = std::max(0, inter_ymax - inter_ymin);
    int inter_area = inter_width * inter_height;

    // ����ÿ��������
    int bbox1_area = (bbox1.xmax - bbox1.xmin) * (bbox1.ymax - bbox1.ymin);
    int bbox2_area = (bbox2.xmax - bbox2.xmin) * (bbox2.ymax - bbox2.ymin);

    // ���㲢�����
    int union_area = bbox1_area + bbox2_area - inter_area;

    // ����IOU
    float iou = static_cast<float>(inter_area) / union_area;
    return iou;
}

void NonMaximumSuppression(std::vector<DetectionBBoxInfo>& boxes, float iouThreshold)
{
    // �����ŶȴӸߵ�������
    std::sort(boxes.begin(), boxes.end(), [](const DetectionBBoxInfo& a, const DetectionBBoxInfo& b)
        {
            return a.score > b.score;
        });

    // ���Ҫ���ƵĿ�
    std::vector<char> suppressed(boxes.size(), 0); // ʹ��char���ͱ���std::vector<bool>��������Ϊ

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (suppressed[i]) continue; // ����Ѿ������ƣ�����

        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (CalculateIOU(boxes[i], boxes[j]) > iouThreshold)
            {
                suppressed[j] = 1; // ���Ƶ�ǰ��
            }
        }
    }

    // ɾ�������ƵĿ�
    auto it = std::remove_if(boxes.begin(), boxes.end(), [&](const DetectionBBoxInfo& box)
        {
            size_t idx = &box - &boxes[0]; // ���㵱ǰԪ�ص�����
            return suppressed[idx]; // ����index���ж��Ƿ�ɾ��
        });

    boxes.erase(it, boxes.end());

    // �����������NMSɸѡ��Ŀ�����
    //std::cerr << "����NMS���˺�Ķ�ά������: " << boxes.size() << std::endl;
}



// ���Ƽ�������Ŷ�
void DrawDetections(cv::Mat& img, const std::vector<DetectionBBoxInfo>& detections)
{
    for (const auto& bbox : detections)
    {
        if (bbox.score > 0.6) // ֻ��ʾ���Ŷȴ���0.6�Ľ��
        {
            // ���Ƽ���
            cv::rectangle(img, cv::Point(bbox.xmin, bbox.ymin), cv::Point(bbox.xmax, bbox.ymax), cv::Scalar(0, 255, 0), 2);

            // �������Ŷ��ı�
            std::string label = cv::format("%.2f", bbox.score);
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.5;
            int thickness = 1;
            cv::Scalar color = cv::Scalar(0, 255, 0); // ������ɫ
            int baseline = 0;

            // �����ı���Ĵ�С
            cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);
            // �ı����ڿ�����½�
            cv::Point textOrg(bbox.xmin, bbox.ymax + textSize.height);

            // ȷ���ı����ᳬ��ͼ��߽�
            if (textOrg.y + textSize.height > img.rows)
            {
                textOrg.y = bbox.ymax - baseline;
            }

            // �����ı�
            cv::putText(img, label, textOrg, fontFace, fontScale, color, thickness, 8);
        }
    }
}



//ǰ����
int Input_Image(const ImageInfo* image_info, MNN::Tensor* inputTensor) 
{
    if (!image_info || !inputTensor) {
        return -1;  // ��������Ĳ���Ϊ��
    }

    timestamps.preprocess_start = std::chrono::high_resolution_clock::now(); // ǰ����ʼʱ���

    // ��ȡ����������ά��
    auto tensorDims = inputTensor->shape();
    int tensorHeight = tensorDims[2];
    int tensorWidth = tensorDims[3];

    // ͼ������ת��ΪMat
    cv::Mat imgMat(image_info->img_height, image_info->img_width, CV_8UC1, image_info->img_data_pt);

    // ִ��ǰ�������� prep_type ѡ��ͬ���߼�
    cv::Mat preprocessedImg;
    switch (image_info->prep_type) {
    case PREP_TYPE_FULL_IMG_RESIZE: {
        // ȫͼresize
        cv::resize(imgMat, preprocessedImg, cv::Size(tensorWidth, tensorHeight));
        break;
    }
    case PREP_TYPE_ROI_DEP_ON_COORD: {
        // ������resize��ֱ��ʹ��ԭͼ
        preprocessedImg = imgMat;
        break;
    }
    default: {
        // Ĭ�ϲ������κδ���
        preprocessedImg = imgMat;
        break;
    }
    }

    // ����MNN�����ĸ�ʽ������NCHW��NHWC��������ʱ����
    MNN::Tensor::DimensionType dimType = inputTensor->getDimensionType();
    MNN::Tensor* tempTensor = nullptr;
    if (dimType == MNN::Tensor::CAFFE) { // NCHW��ʽ
        tempTensor = new MNN::Tensor(inputTensor, MNN::Tensor::CAFFE);
    }
    else if (dimType == MNN::Tensor::TENSORFLOW) { // NHWC��ʽ
        tempTensor = new MNN::Tensor(inputTensor, MNN::Tensor::TENSORFLOW);
    }
    else {
        return -1;  // ���󣺲�֧�ֵ�������ʽ
    }

    // ������ݵ���ʱ���������й�һ��
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

    // ����ʱ���������ݿ�������������
    inputTensor->copyFromHostTensor(tempTensor);
    delete tempTensor;

    timestamps.preprocess_end = std::chrono::high_resolution_clock::now(); // ǰ�������ʱ���

    return 0;  // �ɹ�
}




//��һ������
int Inference(MNN::Interpreter* barcodeNet, MNN::Session* barcodeSession, MNN::Tensor* barcodeInputTensor, MNN::Tensor* barcodeOutputTensor, cv::Mat& image, std::vector<DetectionBBoxInfo>& det_results, TimeStamps& timestamps)
{
    // ת��Ϊ�Ҷ�ͼ��1ͨ����
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // ����ImageInfo�ṹ������ǰ����
    ImageInfo imageInfo;
    imageInfo.img_data_pt = grayImage.data;
    imageInfo.img_height = grayImage.rows;
    imageInfo.img_width = grayImage.cols;
    imageInfo.prep_type = PREP_TYPE_FULL_IMG_RESIZE;  // ʹ��ȫͼresize��Ԥ��������

    // ����ͼ�����ݵ�ģ��
    if (Input_Image(&imageInfo, barcodeInputTensor) != 0)
    {
        std::cerr << "Failed to input image data!" << std::endl;
        return -1;
    }

    // ��¼����ʼʱ���
    timestamps.inference_start = std::chrono::high_resolution_clock::now();

    // ��������
    barcodeNet->runSession(barcodeSession);

    // ��¼�������ʱ���
    timestamps.inference_end = std::chrono::high_resolution_clock::now();

    // ��ȡ��������
    auto nchwTensor = new MNN::Tensor(barcodeOutputTensor, MNN::Tensor::CAFFE);
    barcodeOutputTensor->copyToHostTensor(nchwTensor);

    // ����������
    if (Output_Detection(nchwTensor, &det_results, &imageInfo, barcodeInputTensor) != 0)
    {
        std::cerr << "Failed to output detection results!" << std::endl;
        delete nchwTensor;
        return -1;
    }

    delete nchwTensor;
    return 0;
}



//�ڶ�������
int SecondInference(MNN::Interpreter* barcodeNet, MNN::Session* barcodeSession, MNN::Tensor* barcodeInputTensor, MNN::Tensor* barcodeOutputTensor, const cv::Mat& image, const std::vector<DetectionBBoxInfo>& det_results, TimeStamps& timestamps)
{
    // ��NMS���˺��ÿ������������вü����ٴ�����
    for (const auto& bbox : det_results)
    {
        if (bbox.score <= 0.5)
        {
            continue;
        }

        // �ü�NMS���˺������
        cv::Rect roiRect(cv::Point(bbox.xmin, bbox.ymin), cv::Point(bbox.xmax, bbox.ymax));

        // ȷ���ü�������ԭͼ��Χ��
        if (roiRect.x < 0) roiRect.x = 0;
        if (roiRect.y < 0) roiRect.y = 0;
        if (roiRect.x + roiRect.width > image.cols) roiRect.width = image.cols - roiRect.x;
        if (roiRect.y + roiRect.height > image.rows) roiRect.height = image.rows - roiRect.y;

        cv::Mat roiImage = image(roiRect);

        // ���ü��������ߴ��Ƿ񳬹����������Ĵ�С
        if (roiImage.cols > barcodeInputTensor->width() || roiImage.rows > barcodeInputTensor->height())
        {
            std::cerr << "Skipping ROI inference due to size constraints." << std::endl;
            continue;
        }

        // ����ImageInfoΪROI��Ϣ
        ImageInfo roiInfo;
        roiInfo.img_data_pt = roiImage.data;
        roiInfo.img_height = roiImage.rows;
        roiInfo.img_width = roiImage.cols;
        roiInfo.prep_type = PREP_TYPE_ROI_DEP_ON_COORD;

        // ����ROI��������
        if (Input_Image(&roiInfo, barcodeInputTensor) != 0)
        {
            std::cerr << "Failed to input ROI image data!" << std::endl;
            return -1;
        }

        // ִ������
        std::vector<DetectionBBoxInfo> roi_results;
        if (Inference(barcodeNet, barcodeSession, barcodeInputTensor, barcodeOutputTensor, roiImage, roi_results, timestamps) != 0)
        {
            std::cerr << "ROI Inference failed!" << std::endl;
            return -1;
        }

        // ���Ʋ�չʾROI������
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



//����
int Output_Detection(MNN::Tensor* outputTensor, std::vector<DetectionBBoxInfo>* det_results, const ImageInfo* image_info, MNN::Tensor* inputTensor)
{
    if (!outputTensor || !det_results) {
        return -1;  // �����������Ϊ��
    }

    // ����ȫͼresize�ı�������
    float widthScale = static_cast<float>(image_info->img_width) / inputTensor->width();
    float heightScale = static_cast<float>(image_info->img_height) / inputTensor->height();

    // ����������н���������Ϣ
    auto outputHost = outputTensor->host<float>();  // �����float����
    int numDetections = outputTensor->shape()[1];   // ��������ĵڶ���ά�ȱ�ʾ��������


    for (int i = 0; i < numDetections; ++i)
    {

        DetectionBBoxInfo bbox;
        float xc = outputHost[i * 6 + 0];  // �������Ϊ[xmin, ymin, xmax, ymax, score]
        float yc = outputHost[i * 6 + 1];
        float w = outputHost[i * 6 + 2];
        float h = outputHost[i * 6 + 3];
        bbox.score = outputHost[i * 6 + 4];


        // ת��Ϊԭʼͼ������ϵ
        bbox.xmin = (xc - (w / 2)) * widthScale;
        bbox.ymin = (yc - (h / 2)) * heightScale;
        bbox.xmax = (xc + (w / 2)) * widthScale;
        bbox.ymax = (yc + (h / 2)) * heightScale;

        // ��ת����ļ�����ӵ��������
        det_results->emplace_back(bbox);

    }


    // ʹ��NMS���й��ˣ��趨IOU��ֵ
    // �Ǽ���ֵ���ƹ���
    float iouThreshold = 0.6;
    NonMaximumSuppression(*det_results, iouThreshold);

    return 0;  // �ɹ�
}



