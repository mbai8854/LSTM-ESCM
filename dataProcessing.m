%% load data
clear; clc; load('data/Evolutionry155.mat');

%% process data

% step 1: filter out series with 300~350 keypoints
threshold1 = 300;
threshold2 = 350;

tt = 0;

for i = 1: size(Evolutionry155,2)
    t_xord = Evolutionry155(i).xord;
    Size = size(t_xord);
    Xsize = Size(2);
    Ysize = Size(3);
    Evolutionry155(i).xord_ext = reshape(t_xord(1,:,:),[Xsize, Ysize]);
    
    if (threshold1 <= Xsize) && (Xsize <= threshold2)
        
        tt = tt+1;
        FilteredEvolutionary155(tt) = Evolutionry155(i);
    end
end
%%
% step 2: adjust series into 300 keypoints
for i = 1: size(FilteredEvolutionary155,2)
    
    N_motion = FilteredEvolutionary155(i).N_motion;
    N_keypoint = size(FilteredEvolutionary155(i).xord_ext,1);
    s = FilteredEvolutionary155(i).s;
    xord_ext = FilteredEvolutionary155(i).xord_ext;
    
    
    if N_motion == 2
        % keypoints to remove in each motion class
        removeNum_all = N_keypoint - threshold1;
        removeNum_1 = round(removeNum_all/2);
        removeNum_2 = removeNum_all - removeNum_1;
        
        % find the index of each motion
        index_1 = find(s == 1);
        index_2 = find(s == 2);
        
        % random generate index to remove
        % motion 1
        keepNum1 = length(index_1) - removeNum_1;
        pos = randperm(length(index_1));
        pos = sort(pos(1:keepNum1),2,'ascend');
        card_1 = index_1(pos);
        
        % motion 2
        keepNum2 = length(index_2) - removeNum_2;
        pos = randperm(length(index_2));
        pos = sort(pos(1:keepNum2),2,'ascend');
        card_2 = index_2(pos);
        
        FilteredEvolutionary155(i).xord_ext_filtered = ...
                                [xord_ext(card_1,:); xord_ext(card_2,:)];
        FilteredEvolutionary155(i).s_filtered = ...
                                [s(card_1,:); s(card_2,:)];
        FilteredEvolutionary155(i).N_filtered = [keepNum1, keepNum2];
    else % 3 motions
        % keypoints to remove in each motion class
        removeNum_all = N_keypoint - threshold1;
        removeNum_1 = round(removeNum_all/3);
        removeNum_2 = removeNum_1;
        removeNum_3 = removeNum_all - 2*removeNum_1;
        
        % find the index of each motion
        index_1 = find(s == 1);
        index_2 = find(s == 2);
        index_3 = find(s == 3);
        
        % random generate index to remove
        % motion 1
        keepNum1 = length(index_1) - removeNum_1;
        pos = randperm(length(index_1));
        pos = sort(pos(1:keepNum1),2,'ascend');
        card_1 = index_1(pos);
        
        % motion 2
        keepNum2 = length(index_2) - removeNum_2;
        pos = randperm(length(index_2));
        pos = sort(pos(1:keepNum2),2,'ascend');
        card_2 = index_2(pos);
        
        % motion 3
        keepNum3 = length(index_3) - removeNum_3;
        pos = randperm(length(index_3));
        pos = sort(pos(1:keepNum3),2,'ascend');
        card_3 = index_3(pos);
        
        FilteredEvolutionary155(i).xord_ext_filtered = ...
                                [xord_ext(card_1,:); xord_ext(card_2,:); xord_ext(card_3,:)];
        FilteredEvolutionary155(i).s_filtered = ...
                                [s(card_1,:); s(card_2,:); s(card_3,:)];
        FilteredEvolutionary155(i).N_filtered = [keepNum1, keepNum2, keepNum3];
    end
end

%% save data
clearvars -except FilteredEvolutionary155
save('data/FilteredEvolutionary155.mat');


