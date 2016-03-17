function [out] = randomNeighborhood(im,winSize,numPts)

% Gets a winSize x winSize neighborhood around each pixel in a random set
% of pixel coordinates in the image image. Used to generate data to be fed 
% to texton classifier. Following:
% A Statistical Approach to Material Classification Using Image Patch Exemplars
% Varma and Zisserman, 2009.
% Inputs:
%   im = texture image
%   winSize = size of the neighborhood to be extracted around each pixel
%   numPts = number of randomly selected body pixels 
% Outputs
%   out = [winSize^2 x numel(im)] data matrix


half = floor(winSize/2);
out = zeros(winSize^2,numPts);
ind = zeros(1,numPts); % save indicies to check for repeats
flag = 1;
while flag <= numPts 
    x = randi(numel(im),1,1);
    [r c] = ind2sub(size(im),x);
    
    if im(x) == 0 % skip if pixel is black
        continue
    elseif isempty(find(x == ind,1)) == 0 % skip if pixel has already been used
        continue
    elseif r-half <= 0 || r+half >= size(im,1) % skip if on edge of image
        continue
    elseif c-half <= 0 || c+half >= size(im,2) % skip if on edge of image
        continue
    end
    
    blah = im(r-half:r+half,c-half:c+half); % select region
    
    if isempty(find(blah==0,1)) % check to make sure there aren't any black pixels
        
        % Contrast normalization implied by Weber's law
        L2 = sqrt(sum(sum(abs(blah).^2)));
        coef = log10(1+(L2/.03))/L2;
        
        % output
        out(:,flag) = blah(:)*coef;
        ind(flag) = x;
        flag = flag+1;
        
    else % skip if the neighborhood contains any black pixels
        continue
    end
    
end
end