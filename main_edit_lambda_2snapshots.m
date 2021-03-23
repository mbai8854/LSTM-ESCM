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
[WW1, WW2, C1] = deal(cell(length(Evolutionry155),2)); % Initial predictor and target Sequences

[C2, A, spectralC] = deal(cell(length(Evolutionry155),N_snapshot_max)); % initial clustering results

[CC, CC3, AA3, spectralCC3] = deal(cell(length(Evolutionry155),2));

results = struct([]); % initial final results

num_algs = 1;

Tot = length(Evolutionry155);

training_loss = zeros(Tot, 50, 100);
training_rmse = zeros(Tot, 50, 100);

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
            WWt1_1snapshot = [WWt1(:,1:end-1) WWt1(:,1:end-1) WWt1(:,1:end-1)];
            WWt2_1snapshot = [WWt2(:,1:end-1) WWt2(:,1:end-1) WWt2(:,1:end-1)];
            [net, info] = trainNetwork(WWt1_1snapshot(:,1:2),WWt2_1snapshot(:,1:2),layers,options);
            C1{i} = double(predict(net, WWt1_1snapshot(:,1:2)));
        elseif N_snapshot == 3
            WWt1_1snapshot = [WWt1(:,1) WWt1(:,1)];
            WWt2_1snapshot = [WWt2(:,1) WWt2(:,1)];
            [net, info] = trainNetwork(WWt1_1snapshot,WWt2_1snapshot,layers,options);
            C1{i} = double(predict(net, WWt1_1snapshot));         
        else
            [net, info] = trainNetwork(WWt1(:,1:end-2),WWt2(:,1:end-2),layers,options);
            C1{i} = double(predict(net,WWt1(:,1:end-2)));
        end
    % Here the training loss and training rmse are from the regression task
    % ||X - XC||^2 + lambda * ||C|| and ||X - XC||^2
        training_loss(i,:,mmld) = info.TrainingLoss;
        training_rmse(i,:,mmld) = info.TrainingRMSE;
    
    
    %% sepctral clustering
        for iii = 1:N_snapshot - 2

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
        
        
        
        %% Mingyuan's test error
        test_err = zeros(Tot,100);
%         CC{i,:} = predict(net,WWt1(:, end-1:end));
        errorss_test = [];
        CC_pred = predict(net,WWt1);
        for mmk = 1:2
            CC{i,mmk} = CC_pred(:, end-2+mmk);
            CC3{i,mmk} = reshape(CC{i,mmk},nKeypoints,nKeypoints);
            AA3{i,mmk} = abs(CC3{i,mmk})+abs(CC3{i,mmk}');
            [~,~,~,~,spectralCC3{i,mmk},~]=spectralcluster(double(AA3{i,mmk}),ngroups,ngroups);
            errorss_test(i,mmld,mmk) = missclass(spectralCC3{i,mmk},N,ngroups)/sum(N)*100;
        end
        
%         CC{i,1} = predict(net,WWt1(:, end-1:end));
%         errorss_test = [];
%         CC3{i,1} = reshape(CC{i,1},nKeypoints,nKeypoints);
%         AA3{i,1} = abs(CC3{i,1})+abs(CC3{i,1}');
%         [~,~,~,~,spectralCC3{i,1},~]=spectralcluster(double(AA3{i,1}),ngroups,ngroups);
%         errorss_test(i,mmld) = missclass(spectralCC3{i,1},N,ngroups)/sum(N)*100;
    end
end

%% Save and average
All_avg_smooth_err = zeros(155,100);
All_avg_tim = zeros(155,100);
snapshot1_train_err = zeros(155,100);
snapshots2_train_err = zeros(155,100);

for lii = 1: 100
    for ikk = 1: 155
        % Smooth (all snapshots: training error)
        ali_err = results(ikk, lii).error;
        ali_tim = results(ikk, lii).time;
        ali_snp1_err = results(ikk, lii).error(:, 1);
        if size(results(ikk, lii).error, 2) > 1
            ali_snp2_err = results(ikk, lii).error(:, 1:2);
        else
            ali_snp2_err = NaN;
        end
        
        sz1 = size(ali_err, 2);
        sz2 = size(ali_snp2_err, 2);
        
        Avg_err = sum(ali_err,2)/sz1; %each sequence
        Avg_tim = ali_tim/sz1; % each sequence
        SN1_err = ali_snp1_err;
        SN2_err = sum(ali_snp2_err, 2)/sz2;
        
        
        All_avg_smooth_err(ikk, lii) = Avg_err;
        All_avg_tim(ikk, lii) = Avg_tim;
        snapshot1_train_err(ikk, lii) = SN1_err;
        snapshots2_train_err(ikk, lii) = SN2_err;
    end
end

All_smooth_err_155mean = mean(All_avg_smooth_err, 1);
All_tim_155mean = mean(All_avg_tim, 1);
All_snapshot1_train_err = mean(snapshot1_train_err, 1);
All_snapshots2_train_err = nanmean(snapshots2_train_err, 1);

% for lii = 1: 100
%     Avg_err = 0;
%     Avg_tim = 0;
%     RES = 0;
%     for i = 1:2 %Tot
% 
%         ali_err = results(i,lii).error;
%         ali_tim = results(i, lii).time;
% 
%         sz = size(ali_err,2);
% 
%         Avg_err = Avg_err + sum(ali_err,2)/sz;
%         Avg_tim = Avg_tim + ali_tim/sz;
%         Avg_err_per_tot = Avg_err/Tot;
%         
%         All_avg_err(i, lii) = Avg_err_per_tot;
%         ali_err = results(i, lii).error(:,2:end);
%         sz = size(ali_err,2);
%         RES = RES + sum(ali_err,2)/sz;
%     end
% 
% %     Avg_err = Avg_err/Tot;
% %     All_avg_err(i, lii) = Avg_err;
%     RES = RES/Tot;
%     All_RES(lii) = RES;
%     Avg_tim = Avg_tim/Tot;
%     All_avg_tim(lii) = Avg_tim;
% end
% 
% avg_test_err = mean(errorss_test,1);

% clearvars -except results
% save('Allresults.mat');

%% Test error
% for k = Tot + 1 : length(Evolutionry155)
%     fprintf('Sequences: %i out of %i\n',k,length(Evolutionry155));
%     N_snapshot_k = Evolutionry155(k).N_snapshot;
%     ngroups_k = Evolutionry155(k).N_motion;
%     % F = Evolutionry155(i);
%     N_k = Evolutionry155(k).N;
%     snapshots_xord_k = Evolutionry155(i).snapshots_xord;
%     [WWt1_k,WWt2_k] = deal([]);
%     errorss_k = zeros(num_algs,N_snapshot_k);
%     nKeypoints = 0;
%     for iik = 1:N_snapshot
%         
%         WW_k = snapshots_xord(iik).WW;
%         kappa = 2e-7;
%           
%         
%         % Dimension deduction
%         [U_k,S_k,V_k] = svd(WW_k',0);
%         
%                 
%         % column normali_errzation
%         WW_k = cnormalize(U_k(:,1:4*ngroups_k)');
%         
%         
%         % predictors
%         nKeypoints_k = size(WW_k,2); % # of key points selected
%                 
%         WWt1_k(:,ii) = WW_k(:);
%              
%         
%         % Target
%         WWt_k =WW_k'*WW_k;
%         WWt2_k(:,iik) = WWt_k(:);
%     end
%     
%     WW1{k} = WWt1_k;
%     WW2{k} = WWt2_k;
% end
