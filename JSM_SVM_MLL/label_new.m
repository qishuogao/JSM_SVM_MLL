function [tt_ID_temp , gap_whole] = label_new(NumClass, scale_num,sparsity_matrix , tr_dat, tt_dat_SC, classids, tr_lab)
%=================================================================================
%This function is used to calculate  residual between the test sample and the reconstruction from
%training samples for each class. And then,the label of centered pixel can be determined by minimizing residual.
%input arguments:  NumClass     : number of class
%                  scale_num    : number of scale
%                  sparsity_matrix : multiscale sparsity matrix
%                  tr_dat       : training data
%                  tt_dat_SC    : multiscale test data
%                  tr_lab       : labels of training samples
%output arguments: tt_ID_temp   : the labels of centered pixel
%=================================================================================
tt_ID_temp = [];
gap_whole = zeros(1,NumClass);     
for is = 1: scale_num   
   gap =  label_class (tr_dat,tt_dat_SC{is},sparsity_matrix{is},NumClass,classids,tr_lab);
   gap = gap./size(sparsity_matrix{is},2);
   gap_whole = gap_whole + gap;     
end
  gap_whole;  
index  =  find(gap_whole==min(gap_whole));
tt_ID_temp  =  [tt_ID_temp classids(index(1))];