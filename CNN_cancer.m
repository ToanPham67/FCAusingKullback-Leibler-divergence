% clc
% clear all
% % Load the dataset (assuming you have a dataset in the form of image files)
% imds = imageDatastore('LaoPhoi', ...
%     'IncludeSubfolders', true, 'LabelSource','foldernames');
% % imds2 = imresize(imds1, [224 224]);
% 
% 
% augmentedImds = augmentedImageDatastore([224 224], imds);
% 
% [imdsTrain, imdsTest] = splitEachLabel(augmentedImds, 0.8, 'randomized');

clc
clear all

% Load the dataset (assuming you have a dataset in the form of image files)
imds = imageDatastore('LaoPhoi', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split the original datastore
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Apply augmentation to the training set
augmentedImdsTrain = augmentedImageDatastore([224 224], imdsTrain);

% No augmentation for the test set
imdsTestResized = augmentedImageDatastore([224 224], imdsTest);


% Define the layers of the CNN
layers = [
    imageInputLayer([224 224 1])    % Input layer with image size of 32x32 and 3 channels (RGB)
    convolution2dLayer(3, 16, 'Padding', 'same')   % Convolutional layer with 16 filters of size 3x3
    reluLayer()    % ReLU activation function
    maxPooling2dLayer(2, 'Stride', 2)   % Max pooling layer with a 2x2 window and stride of 2
    convolution2dLayer(3, 224, 'Padding', 'same')   % Convolutional layer with 32 filters of size 3x3
    reluLayer()    % ReLU activation function
    maxPooling2dLayer(2, 'Stride', 2)   % Max pooling layer
    fullyConnectedLayer(2)   % Fully connected layer with 10 output neurons
    softmaxLayer()    % Softmax activation function
    classificationLayer()   % Classification layer
];

% Set the training options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'ValidationData', imdsTest, ...
    'Plots', 'training-progress');
% Train the CNN
net = trainNetwork(augmentedImdsTrain, layers, options);

% Evaluate the performance on the testing set
YPred = classify(net, imdsTest);
YTest = imdsTest.Labels;
accuracy = sum(YPred == YTest) / numel(YTest);

% Display the accuracy
disp("Accuracy: " + accuracy);