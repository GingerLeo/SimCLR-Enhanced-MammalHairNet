clc,clear; 

% 设置路径
imageFolder = '/home/lzh/hair/seg_pic_label/hair_202407_per15';
labelFile = '/home/lzh/hair/seg_pic_label/gTruth_20240919.mat';
pixelLabelFolder = '/home/lzh/hair/seg_pic_label/PixelLabelData_1';

% 读取标注文件
gTruth = load(labelFile).gTruth;

% 创建图像数据存储
imds = imageDatastore(imageFolder, 'FileExtensions', '.Tif', 'IncludeSubfolders', true);

classNames= ["hair","background","scale_bar","length"];
pxds=pixelLabelDatastore(gTruth);  %标签图像

% 分割数据集为训练和验证集
[imdsTrain, imdsVal, pxdsTrain, pxdsVal] = partitionData(imds, pxds);

% 定义网络架构 (U-Net)

imageSize = [size(readimage(imds,1),1) size(readimage(imds,1),2) 1];
numClasses = numel(pxds.ClassNames);
lgraph = unetLayers(imageSize, numClasses, 'EncoderDepth', 4);

% 将图像数据存储和像素标签数据存储组合在一起
pximds_train=pixelLabelImageDatastore(imdsTrain, pxdsTrain,'OutputSize',[1024 1280]);
pximds_valid=pixelLabelImageDatastore(imdsVal, pxdsVal,'OutputSize',[1024 1280]);


% 设置训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 8, ...
    'InitialLearnRate', 1e-3, ...
    'ValidationData', pximds_valid, ...
    'Plots', 'training-progress', ...
    'VerboseFrequency', 10);

% 训练网络
[net,info] = trainNetwork(pximds_train, lgraph, options);

% 保存模型
modelFolder = '/home/lzh/hair/trainedModel_unet_segnet_deeplab';
if ~exist(modelFolder, 'dir')
    mkdir(modelFolder);
end
save(fullfile(modelFolder, 'trainedUNet.mat'), 'net');
save('trainedInfo.mat','info');


% 数据分割函数
function [imdsTrain, imdsVal, pxdsTrain, pxdsVal] = partitionData(imds, pxds)
    % 获取文件数量
    numFiles = numel(imds.Files);
    shuffledIndices = randperm(numFiles);
    numTrain = round(0.9 * numFiles);
    
    trainingIdx = shuffledIndices(1:numTrain);
    valIdx = shuffledIndices(numTrain+1:end);
    
    imdsTrain = subset(imds, trainingIdx);
    imdsVal = subset(imds, valIdx);
    pxdsTrain = subset(pxds, trainingIdx);
    pxdsVal = subset(pxds, valIdx);
end