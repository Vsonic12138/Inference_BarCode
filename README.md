# 一、基于MNN框架的图片二维码检测工程。实现功能如下：


1、完成MNN框架的推理单次推理

2、完成对MNN单次推理框选区域截取后的二次推理

3、实现对CPU、OPENCL、Vulkan推理方式的配置

4、实现对推理方式的后端配置

lib文件夹当中缺少MNN.lib，见百度网盘链接：  [MNN.lib](https://pan.baidu.com/s/1WHIhsV-2UmP2fv4eNY6M4Q?pwd=ed8r)


# 二、运行步骤

1、下载mnn.lib文件到 "lib->release_MD" 文件夹

2、调整工程配置，选择release模式

3、如果报错找不到 "opencl.lib",到 "工程属性->链接器->输入"，删掉opencl.lib即可。
