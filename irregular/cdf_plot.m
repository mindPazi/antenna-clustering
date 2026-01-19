function [x_cdf y_cdf]=cdf_plot(y_vector)

%%% convert y_vector into column
if size(y_vector,1)==1 % input vector is a row
    y_vector=y_vector.';
elseif size(y_vector,1)>1 & size(y_vector,2)>1 % input vector is a matrix
    y_vector=y_vector(:);
end

x_cdf = max(y_vector):-1:-20;

for ix=1:size(x_cdf,2)
    y_cdf(ix)=size(find(y_vector<=x_cdf(ix)),1)/length(y_vector);
end
warn=size(find (y_vector>x_cdf(1)),2)/length(y_vector);

figure
plot(x_cdf,y_cdf)
axis([min(x_cdf) max(x_cdf) min(y_cdf) 1])
xlabel('RPE values R_i [dB]')
ylabel('P(r>R_i)')
title('CDF of RPE values')