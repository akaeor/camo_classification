function [out] = textureLabel(im,texton, parallel)

% Labels each pixel in an image according to texton number.
%
%       Author: Eric C. Orenstein
%               Jaffe Laboratory for Underwater Imaging
%               Scripps Institution of Oceanography
%       Date: 01/27/16
%
% Selects neighborhoods around each pixel and labels as the nearest texton
% according to the Euclidean distance.
% 
% For more information see:
%   Orenstein et al., Automated classification of camouflaging cuttlefish.
%   2016
%
%   Varma and Zisserman, A Statistical Approach to Material Classification 
%   Using Image Patch Exemplars. 2009. 

% Inputs:
%   im = texture image
%   texton = matrix of textons [numTextons x pix]
%   parallel = flag switching parfor processing
% Outputs
%   map = matrix of labeled pixels in the texture image

winSize = sqrt(size(texton,2));
half = floor(winSize/2); % half-width of neighborhood
map = [];
ind = find(im ~= 0);

if parallel == 0
    
    % no parallel processing
    for i = 1:length(ind)
        idx = ind(i);
        [r c] = ind2sub(size(im),idx);

        if r-half <= 0 || r+half >= size(im,1) % skip if on edge of image
            continue
        elseif c-half <= 0 || c+half >= size(im,2) % skip if on edge of image
            continue
        end

        % get neighborhood
        temp = im(r-half:r+half,c-half:c+half);

        if isempty(find(temp==0,1))
            temp = repmat(temp(:),1,size(texton,1));
            [~,ix] = max(-1/2*sum((temp'-texton).^2,2));
            map(r,c) = ix;
        else
            continue
        end
    end
    
elseif parallel == 1
    
    % use parallel processing
    
    parfor i = 1:length(ind)
        idx = ind(i);
        [r c] = ind2sub(size(im),idx);

        if r-half <= 0 || r+half >= size(im,1) % skip if on edge of image
            continue
        elseif c-half <= 0 || c+half >= size(im,2) % skip if on edge of image
            continue
        end

        % get neighborhood
        temp = im(r-half:r+half,c-half:c+half);

        if isempty(find(temp==0,1))
            temp = repmat(temp(:),1,size(texton,1));
            [~,ix] = max(-1/2*sum((temp'-texton).^2,2));
            map = [map; ix];
        else
            continue
        end
    end
end
out = hist(map(:),size(texton,1));
end