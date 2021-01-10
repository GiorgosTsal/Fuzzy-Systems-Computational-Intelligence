clear; close all; clc;

dataSet=load('../Datasets/airfoil_self_noise.dat');

[trainingSet,validationSet,testSet]=split_scale(dataSet,1);


%% Evaluation function
Rsq = @(yPrediciton,y) 1-sum((yPrediciton-y).^2)/sum((y-mean(y)).^2);
Nmse = @(yPrediciton,y) sum((y-yPrediciton).^2)/sum((y-mean(y)).^2);

FIS=genfis1(trainingSet,3,'gbellmf','linear'); %2,3 and constant,linear
[trainFis,trainingError,~,validationFIS,validationError]=anfis(trainingSet,FIS,[500 0 0.01 0.9 1.1],[],validationSet);
plotMFs(FIS,size(trainingSet,2)-1);

%% Validation
figure;
plot([trainingError validationError],'LineWidth',2); grid on;
xlabel('# of Iterations'); ylabel('Error');
%legend('Training Error','Validation Error');
title('ANFIS Hybrid Training - Validation');
% prediction=evalfis(testSet(:,1:end-1),validationFIS);
prediction=evalfis(validationFIS,testSet(:,1:end-1));
R2=Rsq(prediction,testSet(:,end));
meanSquaredError = mse(prediction,testSet(:,end));
RMSE=sqrt(meanSquaredError);

figure;
plotMFs(validationFIS,size(trainingSet,2)-1);

%% Performance table initialization
metrics=zeros(4,1);

%% Metrics computation

NMSE = Nmse(prediction, testSet(:,end));
NDEI = sqrt(NMSE);
R2 = Rsq(prediction, testSet(:,end));

metrics(1:4,:) = [RMSE; NMSE; NDEI; R2];

