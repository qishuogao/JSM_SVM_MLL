close all
clear all
clc

%%Probailistic JSM classification

demo_JSM_IP;
clear

load IP

%%Select the training samples 
indexes = train_test_random_new_i(trainall(2,:),50,[1 7 9 16]);
train1 = trainall(:,indexes);

% the remaining used for test
test1 = trainall;
test1(:,indexes) = [];

ksub = 100;


no_train = size(train1,2);



%% Probabilistic SVM Classification

disp (['SVM: ... '])
img = reshape(im',[no_lines,no_rows,size(im,1)]);

groundtruth = zeros(no_lines, no_rows);
groundtruth(trainall(1,:))=trainall(2,:);

training2D=zeros(no_lines, no_rows);
training2D(train1(1,:))=train1(2,:);

in_param.probability_estimates = 1;
in_param.nfold = 5;  % cross validation for parameters
in_param.cost = 125;
in_param.gamma = 2^(-6);

[map_class outdata] = classify_svm(img,training2D,in_param);

% output probabilities
[pSVM,order,ordervalue] = ...
    aux_ordenar_v4(groundtruth,outdata.prob_estimates,no_lines, no_rows);
pSVM = pSVM';


%% Decision Fusion

load prob_ip
lambda = 0.4;
p_all = (lambda*prob+(1-lambda)*pSVM);


%% MRF-Based Spectral-Spatial Classification

mu = 2;

% SVM-MLL
Dc = reshape((log(pSVM+eps))',[no_lines, no_rows, no_classes]);
Sc = ones(no_classes) - eye(no_classes);
gch = GraphCut('open', -Dc, mu*Sc);
[gch SVMmll_map] = GraphCut('expand',gch);
gch = GraphCut('close', gch);
SVMmll_map=double(SVMmll_map)+1;
[SVMmll_acc] = ComputeClassificationAccuracy(SVMmll_map,gt)
clear Dc


%JSM-MLL

Dc = reshape((log(prob+eps))',[no_lines, no_rows, no_classes]);
Sc = ones(no_classes) - eye(no_classes);
gch = GraphCut('open', -Dc, mu*Sc);
[gch, JSMmll_map] = GraphCut('expand',gch);
gch = GraphCut('close', gch);
JSMmll_map=double(JSMmll_map)+1;
[JSMmll_acc] = ComputeClassificationAccuracy(JSMmll_map,gt)
clear Dc


% SVM-JSM-MLL
Dc = reshape((log(p_all+eps))',[no_lines, no_rows, no_classes]);
Sc = ones(no_classes) - eye(no_classes);
gch = GraphCut('open', -Dc, mu*Sc);
[gch, SVM_JSM_map] = GraphCut('expand',gch);
gch = GraphCut('close', gch);
SVM_JSM_map=double(SVM_JSM_map)+1;
[SVM_JSM_acc] = ComputeClassificationAccuracy(SVM_JSM_map,gt)
clear Dc

