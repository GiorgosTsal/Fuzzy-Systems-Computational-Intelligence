format compact
clear
clc

%% Load data - Split data
data=readtable('../Datasets/epileptic_seizure_data.csv');
data=removevars(data,{'Var1'});
data= table2array(data);

preproc=1;
[trnData,chkData,tstData]=split_scale(data,preproc);
Perf=zeros(1,4);


NF = [10 15 19 20]; % number of features
% values for radii
RADII = [0.3 0.5 0.7 0.9];
     
     
%% GRID SEARCH & 5-fold cross validation
fprintf('\n *** Cross validation \n');

% Keep only the number of features we want and not all of them
% Specify their order and later use the ranks array
[ranks, weights] = relieff(data(:, 1:end - 1), data(:, end), 100);
    
% Check every case for every parameter possible

for f = 1 : length(NF)
 
    for r = 1 : length(RADII)
        fprintf('\n *** Number of features: %d', NF(f));
        fprintf('\n *** Radii value: %d \n', RADII(r));
     
        
        % error in each fold
        c = cvpartition(trnData(:, end), 'KFold', 5);
        error = zeros(c.NumTestSets, 1);
     
        % Generate the FIS
        fprintf('\n *** Generating the FIS\n');
     
        % As input data I give the train_id's that came up with the
        % partitiong and only the most important features

        init_fis = genfis2(trnData(:, ranks(1:NF(f))), trnData(:, end), RADII(r));
        rule_grid(f, r) = length(init_fis.rule);
        if (rule_grid(f, r) == 1 || rule_grid(f,r) > 100) % if there is only one rule we cannot create a fis, so continue to next values
            continue; % or more than 100, continue, for speed reason
        end
        % 5-fold cross validation
        for i = 1 : c.NumTestSets
            fprintf('\n *** Fold #%d\n', i);
         
            train_id = c.training(i);
            test_id = c.test(i);
         
            % Keep separate
            training_data_x = trnData(train_id, ranks(1:NF(f)));
            training_data_y = trnData(train_id, end);
         
            validation_data_x = trnData(test_id, ranks(1:NF(f)));
            validation_data_y = trnData(test_id, end);
         
            % Tune the fis
            fprintf('\n *** Tuning the FIS\n');
         
            % Set some options
            % The fis structure already exists
            % set the validation data to avoid overfitting
         
            anfis_opt = anfisOptions('InitialFIS', init_fis, 'EpochNumber', 50, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0, 'ValidationData', [validation_data_x validation_data_y]);
         
            [trn_fis, trainError, stepSize, init_fis, chkError] = anfis([training_data_x training_data_y], anfis_opt);
         
            % Evaluate the fis
            fprintf('\n *** Evaluating the FIS\n');
         
            % No need to specify specific options for this, keep the defaults
            Y = evalfis(chkData(:, ranks(1:NF(f))), init_fis);
            Y=round(Y);
            % Calculate the error
            error(i) = sum((Y - chkData(:, end)) .^ 2);
        end
     
        cvErr = sum(error) / c.NumTestSets;
        error_grid(f, r) = cvErr / length(Y)
    end
end

%% PLOT THE ERROR
fprintf('The error for diffent values of F and Radii is:');
error_grid
% save('error_grid', 'error_grid');

fprintf('The number of rules created for diffent values of F and Radii is:');
rule_grid
% save('rule_grid', 'rule_grid');


%% PLOT
figure;
suptitle('Error for different number of features and radii values');

subplot(2,2,1);
bar(error_grid(1,:))
xlabel('radii value');
ylabel('Mean Square Error');
xticklabels({'0.3','0.5','0.7','0.9'});
legend('10 features')

subplot(2,2,2);
bar(error_grid(2,:));
xlabel('radii value');
ylabel('Mean Square Error');
xticklabels({'0.3','0.5','0.7','0.9'});
legend('15 features')

subplot(2,2,3);
bar(error_grid(3,:));
xlabel('radii value');
ylabel('Mean Square Error');
xticklabels({'0.3','0.5','0.7','0.9'});
legend('19 features')

subplot(2,2,4);
bar(error_grid(4,:));
xlabel('radii value');
ylabel('Mean Square Error');
xticklabels({{'0.3','0.5','0.7','0.9'});
legend('20 features')
saveas(gcf, 'error_grid_wrg_f_r.png');

figure;
bar3(error_grid);
ylabel('Number of feature');
xticklabels({'0.2','0.6','0.5','0.7'});
xlabel('Radii values');
xticklabels({'1st','2nd','3rd','4th'});
zlabel('Mean square error');
title('Error for different number of features and radii');
saveas(gcf, 'error_wrt_f_r.png');

figure;
bar3(rule_grid);
ylabel('Number of features');
yticklabels({'10','15','19','20'});
xlabel('Radii values');
xticklabels({'1st','2nd','3rd','4th'});
zlabel('Number of rules created');
title('Rules created for different number of features and radii');
saveas(gcf, 'rules_wrt_f_r.png');