function [out] = normalizeImages(imageList, pathToFiles, outBasePath, fmt)

% Normalize and contrast enhance a directory of images and saves them
%
%       Author: Eric C. Orenstein
%               Jaffe Laboratory for Underwater Imaging
%               Scripps Institution of Oceanography
%       Date: 01/27/16
% Inputs:
%       imageList == structure of image paths
%       pathToFiles == base path for files in imageList 
%       outBasePath == file path were normalized images will be saved
%       fmt == file format for normalized images
% Outputs:
%       out == is a place holder. No functional output

for ii = 1:length(imageList)
        im = imread([pathToFiles,'/',imageList(ii).name]);
        im = double(im);
        im = im./255; % normalize
        vec = im(im~=0);
        vec2 = (vec - mean(vec))./std(vec);
        temp = zeros(size(im));
        temp(im~=0) = vec2;
        temp = uint8(temp);
        outName = [outBasePath,'/',imageList(ii).name];
        imwrite(temp, outName, fmt);
        
        if rem(ii,100) == 0 
            sprintf('Doen with %d of %d',ii, length(imageList))
        end
end

out = 1;