%% Remark: this program should be run on MATLAB2019a.

%% Prepare data
clear; clc; load('data/Evolutionry155.mat');
rng(1000)
%% Processing sequences

N_snapshot_max = 0;

for i = 1:length(Evolutionry155)
    
    N_snapshot = Evolutionry155(i).N_snapshot;
    
    if i == 1
        
        N_snapshot_max = N_snapshot;
    end
    N_snapshot_max = max(N_snapshot_max,N_snapshot);
end

% parameter initiali_errzation

% parameter initiali_errzation
[WW1, WW2, C1] = deal(cell(length(Evolutionry155),1)); % Initial predictor and target Sequences

[C2, A, spectralC] = deal(cell(length(Evolutionry155),N_snapshot_max)); % initial clustering results

[CC, CC3, AA3, spectralCC3] = deal(cell(length(Evolutionry155),1));

results = struct([]); % initial final results

num_algs = 1;

Tot = length(Evolutionry155);

training_loss = zeros(Tot, 50, 100);
training_rmse = zeros(Tot, 50, 100);

test_err = zeros(Tot,100);

lambda_all = logspace(-5, 0, 100);


for i = 1:Tot
    
    %% data initiali_errzation
    fprintf('Sequences: %i out of %i\n',i,length(Evolutionry155));
    % extract out the ith video sequence
    
    % s = Evolutionry155(i).s;
    N_snapshot = Evolutionry155(i).N_snapshot;
    ngroups = Evolutionry155(i).N_motion;
    % F = Evolutionry155(i);
    N = Evolutionry155(i).N;
    
    % xord = Evolutionry155(i).xord;
    % x1ord = Evolutionry155(i).x1ord;
    % yord = Evolutionry155(i).yord;
    
    snapshots_xord = Evolutionry155(i).snapshots_xord;
    % snapshots_x1ord = Evolutionry155(i).snapshots_x1ord;
    % snapshots_yord = Evolutionry155(i).snapshots_yord;
    
    % Reshape the ith sequence
    
    % temporary parameter initiali_errzation
    [WWt1,WWt2] = deal([]);
    
    errorss = zeros(num_algs,N_snapshot-1);
%     timess = zeros(num_algs,N_snapshot);
    %% data precessing
    nKeypoints = 0;
    for ii = 1:N_snapshot
        
        WW = snapshots_xord(ii).WW;
        kappa = 2e-7;
          
        
        % Dimension deduction
        [U,S,V] = svd(WW',0);
        
                
        % column normali_errzation
        WW = cnormalize(U(:,1:4*ngroups)');
        
        
        % predictors
        nKeypoints = size(WW,2); % # of key points selected
                
        WWt1(:,ii) = WW(:);
             
        
        % Target
        WWt =WW'*WW;
        WWt2(:,ii) = WWt(:);
    end
    
    WW1{i} = WWt1;
    WW2{i} = WWt2;
     
    
    %% Define LSTM network architecture
    featureDimension = size(WWt1,1);
    
    numHiddenUnits = ceil(nKeypoints/5); % tunable hyperparam
    
    numResponses = nKeypoints^2 - nKeypoints;
    
    paddingSize = nKeypoints;
    
%     lambda = 0.1; % tunable hyperparam
    
    for mmld = 1: 100
        lambda = lambda_all(mmld);
        layers = [ ...
            sequenceInputLayer(featureDimension)
            lstmLayer(numHiddenUnits,'OutputMode','sequence')
            fullyConnectedLayer(numResponses)
            myPaddingLayer(paddingSize)
            myRegressionLayer('Evolving', lambda)];
    
        maxEpochs = 50;  %60
        %     miniBatchSize = 20;
        
        
        options = trainingOptions('adam', ...
            'MaxEpochs',maxEpochs, ...
            'InitialLearnRate',0.001, ...
            'GradientThreshold',1, ...
            'Shuffle','never', ...
            'Verbose',0); %'Plots','training-progress',...
        %% Network training
        ts=cputime;
    
        %     num_training = 1;
        % Now for each video of these 155 videos, we take the last snapshot as
        % the test data and all the rest snapshots before as the training data.
        % For the videos with only two snapshots, we repeat the first snapshot
        % once. Then take this snapshot and the repeated snapshot together as
        % the training data. The original last snapshot is still the test data.
        if N_snapshot == 2
            WWt1_1snapshot = [WWt1(:,1:end-1) WWt1(:,1:end-1)];
            WWt2_1snapshot = [WWt2(:,1:end-1) WWt2(:,1:end-1)];
            [net, info] = trainNetwork(WWt1_1snapshot,WWt2_1snapshot,layers,options);
            C1{i} = double(predict(net,WWt1_1snapshot(:,1:end-1)));
        else
            [net, info] = trainNetwork(WWt1(:,1:end-1),WWt2(:,1:end-1),layers,options);
            C1{i} = double(predict(net,WWt1(:,1:end-1)));
        end
    % Here the training loss and training rmse are from the regression task
    % ||X - XC||^2 + lambda * ||C|| and ||X - XC||^2
        training_loss(i,:,mmld) = info.TrainingLoss;
        training_rmse(i,:,mmld) = info.TrainingRMSE;
    
    
    %% sepctral clustering
        for iii = 1:N_snapshot - 1

            % reshape Cl
            C2{i,iii} = reshape(C1{i}(:,iii),nKeypoints,nKeypoints);

            % create affinity matrix
            A{i,iii} = abs(C2{i,iii})+abs(C2{i,iii}');

            % Spectral Clustering
            [~,~,~,~,spectralC{i,iii},~]=spectralcluster(A{i,iii},ngroups,ngroups);

            % error calculation
            errorss(1,iii) = missclass(spectralC{i,iii},N,ngroups)/sum(N)*100;
        end

        results(i,mmld).time = (cputime-ts);
        results(i,mmld).error = errorss;       
        
        
        
        %% Test error
%         test_err = zeros(Tot,100);
        
        CC{i,1} = predict(net,WWt1(:, end));
%         errorss_test = [];
        CC3{i,1} = reshape(CC{i,1},nKeypoints,nKeypoints);
        AA3{i,1} = abs(CC3{i,1})+abs(CC3{i,1}');
        [~,~,~,~,spectralCC3{i,1},~]=spectralcluster(double(AA3{i,1}),ngroups,ngroups);
        test_err(i,mmld) = missclass(spectralCC3{i,1},N,ngroups)/sum(N)*100;
    end
end

mean_test_error = mean(test_err, 1);
