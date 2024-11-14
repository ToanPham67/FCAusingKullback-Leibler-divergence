clc
clear all
% Load the pre-trained ResNet-50 network
net = resnet50();

% Display the network architecture
disp(net.Layers);

% Specify the path to the image directory
imageDir = 'LaoPhoi';

% Create an imageDatastore for the image directory
imds = imageDatastore(imageDir, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split the image datastore into training and testing sets
[trainSet, testSet] = splitEachLabel(imds, 0.8, 'randomized');

% Specify the layers to freeze during fine-tuning
layersToFreeze = 1:10;

% Modify the fully connected layer of the network for transfer learning
numClasses = numel(categories(trainSet.Labels));
newFullyConnectedLayer = fullyConnectedLayer(numClasses, ...
    'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);

% Replace the existing fully connected layer with the new layer
net.Layers(end-2) = newFullyConnectedLayer;
net.Layers(end) = classificationLayer();

% Set the training options for fine-tuning
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress');

% Fine-tune the network
net = trainNetwork(trainSet, net, options);

% Classify the testing set
predictedLabels = classify(net, testSet);

% Calculate the accuracy
accuracy = sum(predictedLabels == testSet.Labels) / numel(testSet.Labels);
disp("Accuracy: " + accuracy);