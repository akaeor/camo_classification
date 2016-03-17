function [out] = getNeighborhood(im,winSize)

% Gets a winSize x winSize neighborhood around each pixel in an image. Used
% to generate data to be fed to texton classifier. Following:
% A Statistical Approach to Material Classification Using Image Patch Exemplars
% Varma and Zisserman, 2009.
% Inputs:
%   im = texture image
%   winSize = size of the neighborhood to be extracted around each pixel
% Outputs
%   out = [winSize^2 x numel(im)] data matrix

half = floor(winSize/2); % half-width of neighborhood
[m n] = size(im);
num = numel(im(1+half:m-half,1+half:n-half)); % number of elements
out = zeros(winSize^2,num);

flag = 1;

% iterate over x-pixels
for i = 1+half:m-half
    
    % iterate over y-pixels
    for j = 1+half:n-half

        % get neighborhood
        temp = im(i-half:i+half,j-half:j+half);
        
        % Contrast normalization implied by Weber's law 
        L2 = sqrt(sum(sum(abs(temp).^2)));
        coef = log10(1+(L2/.03))/L2;
        
        % output
        out(:,flag) = temp(:)*coef;
        flag = flag+1;
        
    end
    
end
end