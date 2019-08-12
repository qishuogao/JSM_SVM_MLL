function [indexes]=train_test_random_new(y,nmax,Vmin)
% nmax is the maximum number of train samples,  Vmin is the lable of class
% which have less samples
K = max(y);
% generate the  training set
indexes = [];
V = 1:K;
for i=1:length(Vmin)
    V(Vmin(i))=0;
end
V = nonzeros(V)';
for i = V
    index1 = find(y == i);                        %index1 equal to each label except for vmin
    per_index1 = randperm(length(index1));           %per_index1 is equal to the indexs randomly sorted by randperm
     indexes = [indexes ;index1(per_index1(1:nmax))']; %choose nmax smaples
end
for i = Vmin
    index1 = find(y == i);                  
    nmin = floor(length(index1)/2);
    per_index1 = randperm(length(index1));
    indexes = [indexes ;index1(per_index1(1:nmin))'];
end
indexes = indexes(:);
