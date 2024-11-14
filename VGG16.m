% Load the pre-trained VGG-16 network
net = vgg16();

% Display the network architecture
disp(net.Layers);

% Specify the path to the image directory
imageDir = 'ungthuda';

% Create an imageDatastore for the image directory
imds = imageDatastore(imageDir, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');


% Split the image datastore into training and testing sets
[trainSet, testSet] = splitEachLabel(imds, 0.8, 'randomized');

% Resize images to the input size required by the VGG-16 network
inputSize = net.Layers(1).InputSize(1:2);
augmentedTrainSet = augmentedImageDatastore(inputSize, trainSet);
augmentedTestSet = augmentedImageDatastore(inputSize, testSet);

% Extract the features from the pre-trained layers of VGG-16
featuresTrain = activations(net, augmentedTrainSet, 'fc7', ...
    'MiniBatchSize', 224, 'OutputAs', 'columns');
featuresTest = activations(net, augmentedTestSet, 'fc7', ...
    'MiniBatchSize', 224, 'OutputAs', 'columns');

% Get the class labels for the training and testing sets
trainLabels = trainSet.Labels;
testLabels = testSet.Labels;

% Train a classifier on top of the extracted features
classifier = fitcecoc(featuresTrain, trainLabels);

% Predict the class labels for the testing set
predictedLabels = predict(classifier, featuresTest);

% Evaluate the performance
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
disp("Accuracy: " + accuracy);