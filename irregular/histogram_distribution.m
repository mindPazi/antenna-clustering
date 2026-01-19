function [h_values h_axis]=histogram_distribution(y_vector)

%%% convert y_vector into column
if size(y_vector,1)==1 % input vector is a row
    y_vector=y_vector.';
elseif size(y_vector,1)>1 & size(y_vector,2)>1 % input vector is a matrix
    y_vector=y_vector(:);
end

h_axis=min(y_vector):(max(y_vector)-min(y_vector))/10:max(y_vector);

for ix=1:size(h_axis,2)-1
    h_values(ix)=size(find(y_vector>=h_axis(ix) & y_vector<h_axis(ix+1)),1);
end
h_values=h_values./length(y_vector);

% figure
bar(h_axis(1:end-1),h_values);grid
axis([min(h_axis) max(h_axis) 0 max(h_values)+0.05])
xlabel('PS error [°]')
ylabel('distribution PS error [%]')