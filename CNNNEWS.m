clc
clear all
% Define the input size
inputSize = [224 224 3];  % 224x224 RGB images

% Load the training images
imdsTrain = imageDatastore('LaoPhoi', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Load the validation images
imdsValidation = imageDatastore('LaoPhoi', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Resize the training and validation images
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);

% Define the network layers
layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')

    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.5, 'Name', 'dropout')
    fullyConnectedLayer(numel(unique(imdsTrain.Labels)), 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Set the training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(augimdsTrain, layers, options);

% Evaluate the network
[YPred, scores] = classify(net, augimdsValidation);
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);
fprintf('Validation accuracy: %.2f%%\n', accuracy * 100);
