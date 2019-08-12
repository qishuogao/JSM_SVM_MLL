function [list]=list_create(gt,row,col,window)

%%%This function is to generate the neighor indexes of the test sample in
%%%DPR procedure



list=[];

for i=1:col
    for j=1:row
        dim=floor(window/2);
        index_y=[i-dim:i+dim];
        index_x=[j-dim:j+dim];
        indexes=[];
        
        num=length(index_x);
         for m=1:num
             for n=1:num
                 index=[];
                    if index_x(n)<=0 || index_y(m)<=0 || index_x(n)>row || index_y(m)>col
                      index=0;
                    elseif index_x(n)==j && index_y(m)==i
                      index=[];
                    else
                      index=sub2ind(size(gt),index_x(n),index_y(m));
                    end
             
                  indexes=[indexes index];
                
             end
         end
         list=[list; indexes];
         list=sort(list,2);
    end
end