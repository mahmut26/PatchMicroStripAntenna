
folder = '.TXT PATH MUST ENTERED';
files = dir(fullfile(folder, '*.txt'));

X = [];
Y = [];

for i = 1:length(files)
    filepath = fullfile(folder, files(i).name);
    data = readmatrix(filepath);

    if size(data,1) >= 901 && size(data,2) >= 9
        R = data(1, 1:5);
        frekans_log = log10(data(:,6));        
        s11_db = data(:,9);

        X = [X; [repmat(R,901,1), frekans_log]];
        Y = [Y; s11_db];
    else
        warning("Wrong file: %s", files(i).name);
    end
end


valid = all(~isnan(X), 2) & ~isnan(Y);
X = X(valid,:);
Y = Y(valid);


valid_range = Y > -60 & Y < 5;
X = X(valid_range, :);
Y = Y(valid_range);


X = normalize(X);


cv = cvpartition(size(X,1), 'HoldOut', 0.2);
idx = cv.test;

XTrain = X;
YTrain = Y;

layers = [
    featureInputLayer(6)

    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(32)
    reluLayer

    fullyConnectedLayer(16)
    
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(1)
    regressionLayer
];



options = trainingOptions('adam', ...
    'MaxEpochs', 250, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 5e-3, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');




net = trainNetwork(XTrain, YTrain, layers, options);

testFolder = 'TEST FOLDER PATH';
testFiles = dir(fullfile(testFolder, 'temp_sample*.txt'));
testFiles = testFiles(1:min(5, length(testFiles)));

XTest = [];
YTest = [];


for i = 1:length(testFiles)
    filepath = fullfile(testFolder, testFiles(i).name);
    data = readmatrix(filepath);

    if size(data,1) >= 901 && size(data,2) >= 9
        R = data(1, 1:5);
        frekans_log = log10(data(:,6));
        s11_db = data(:,9);

        XTest = [XTest; [repmat(R,901,1), frekans_log]];
        YTest = [YTest; s11_db];
    else
        warning("TEST FILE IS WRONG: %s", testFiles(i).name);
    end
end

valid = all(~isnan(XTest),2) & ~isnan(YTest);
XTest = XTest(valid,:);
YTest = YTest(valid,:);
valid_range = YTest > -60 & YTest < 5;
XTest = XTest(valid_range,:);
YTest = YTest(valid_range,:);

YPredTest = predict(net, XTest);
mseTest = mean((YPredTest - YTest).^2);
fprintf("Test MSE: %.4f dB^2\n", mseTest);

SS_res = sum((YTest - YPredTest).^2);
SS_tot = sum((YTest - mean(YTest)).^2);
R2 = 1 - SS_res / SS_tot;
fprintf("R²: %.4f\n", R2);

figure;
plot(YTest, YPredTest, 'bo')
xlabel('REAL S11 (dB)')
ylabel('PREDICT S11 (dB)')
title('REAL vs. PREDICTED S11 VALUES')
grid on;
hold on;
minVal = min([YTest; YPredTest]);
maxVal = max([YTest; YPredTest]);
plot([minVal maxVal], [minVal maxVal], 'r--')
legend('PREDICTS', 'y = x LINE');
