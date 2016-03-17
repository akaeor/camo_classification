function [out] = makeDataMatrix(pathToFiles,files,texton,idx,parallel)

% Make the training matrix.
%
%       Author: Eric C. Orenstein
%               Jaffe Laboratory for Underwater Imaging
%               Scripps Institution of Oceanography
%       Date: 01/27/16
%
% Generates the training matrix. Labels every pixel in the subregions and 
% bins into historgram labels.
% 
% For more information see:
%   Orenstein et al., Automated classification of camouflaging cuttlefish.
%   2016
%
%   Varma and Zisserman, A Statistical Approach to Material Classification 
%   Using Image Patch Exemplars. 2009. 
%
% Inputs:
%   pathToFiles = string indicating where images are
%   files = cell array with specific image names
%   texton = matrix of textons
%   idx = indicies of the images to be used for training
%   parallel = flag that switches parallel processing on
% Outputs:
%   out = [numImage x numBins] training matrix for a single class

out = zeros(length(idx),size(texton,1));


for ii = 1:length(idx)
    jj = idx(ii);
    % Read
    temp = imread([pathToFiles,'/',files(jj).name]);
    temp = im2double(temp);
    
    % Label
    % [map] = textureLabel2(temp,texton);
    [map] = textureLabel(temp,texton,parallel);
    
    out(ii,:) = map;
    
    if rem(ii,100) == 0 
        fprintf('Done with %d of %d \n',ii, length(idx))
    end
end

end