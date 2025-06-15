
folderSobol = 'SOBOL PATH';
folderLHS = 'LHS PATH';

filesSobol = dir(fullfile(folderSobol, '*.txt'));
filesLHS = dir(fullfile(folderLHS, '*.txt'));

X = [];
Y = [];

for i = 1:length(filesSobol)
    filepath = fullfile(folderSobol, filesSobol(i).name);
    data = readmatrix(filepath);
    if size(data,1) >= 901 && size(data,2) >= 9
        R = data(1,1:5);
        frekans_log = log10(data(:,6));
        s11_db = data(:,9);
        X = [X; [repmat(R, 901, 1), frekans_log]];
        Y = [Y; s11_db];
    else
        warning("Sobol FILE WRONG: %s", filesSobol(i).name);
    end
end

maxLHSfiles = min(110, length(filesLHS));
for i = 1:maxLHSfiles
    filepath = fullfile(folderLHS, filesLHS(i).name);
    data = readmatrix(filepath);
    if size(data,1) >= 901 && size(data,2) >= 9
        R = data(1,1:5);
        frekans_log = log10(data(:,6));
        s11_db = data(:,9);
        X = [X; [repmat(R, 901, 1), frekans_log]];
        Y = [Y; s11_db];
    else
        warning("LHS FILE WRONG: %s", filesLHS(i).name);
    end
end

valid = all(~isnan(X),2) & ~isnan(Y);
X = X(valid,:);
Y = Y(valid);


X_mu = mean(X);
X_sigma = std(X);
X_norm = (X - X_mu) ./ X_sigma;


Y_mu = mean(Y);
Y_sigma = std(Y);
Y_norm = (Y - Y_mu) ./ Y_sigma;



cv = cvpartition(size(X_norm,1), 'HoldOut', 0.2);
idxTest = cv.test;

XTrain = X_norm(~idxTest,:);
YTrain = Y_norm(~idxTest);

XTest = X_norm(idxTest,:);
YTest = Y_norm(idxTest);


layers = [
    featureInputLayer(6)

    fullyConnectedLayer(1024)  
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)        

    fullyConnectedLayer(512) 
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)

    fullyConnectedLayer(2048) 
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)

    fullyConnectedLayer(128)  
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)

    fullyConnectedLayer(64)   
    reluLayer

    fullyConnectedLayer(32)   
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(16)   
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(8)   
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(1)
    maeRegressionLayer()  

];



options = trainingOptions('adam', ...
    'MaxEpochs', 2000, ...
    'MiniBatchSize', 1024, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');


net = trainNetwork(XTrain, YTrain, layers, options);


YPred_norm = predict(net, XTest);


YPred = YPred_norm * Y_sigma + Y_mu;
YTest_orig = YTest * Y_sigma + Y_mu;


figure;
plot(YTest_orig, YPred, 'bo');
hold on;
minVal = min([YTest_orig; YPred]);
maxVal = max([YTest_orig; YPred]);
plot([minVal maxVal], [minVal maxVal], 'r--');
xlabel('REAL S11 (dB)');
ylabel('PREDICT S11 (dB)');
title('REAL vs. PREDICTED S11 VALUES');
grid on;
legend('PREDICTS', 'y = x LINE');


mseTest = mean((YPred - YTest_orig).^2);
fprintf("Test MSE: %.6f dB^2\n", mseTest);

SS_res = sum((YTest_orig - YPred).^2);
SS_tot = sum((YTest_orig - mean(YTest_orig)).^2);
R2 = 1 - SS_res / SS_tot;
fprintf("R²: %.4f\n", R2);

fprintf("YPred min: %.4f, max: %.4f\n", min(YPred), max(YPred));
fprintf("YTest min: %.4f, max: %.4f\n", min(YTest_orig), max(YTest_orig));

