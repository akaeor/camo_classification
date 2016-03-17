function [out] = textonGenerator(pathToFiles,files,numImage,neigh,...
    numClass,numPts,maxIter, method)

% Generates textons for a given class.
%
%       Author: Eric C. Orenstein
%               Jaffe Laboratory for Underwater Imaging
%               Scripps Institution of Oceanography
%       Date: 01/27/16
%
% Neigborhoods are chosen around randomly selected pixels from randomly 
% selected images to create a data matrix. kMeans is then used to generate
% textons. Textons defined as the cluster center.
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
%   numImage = number of example images to create textons
%   neigh = size of neighborhood to use
%   numClass = number of textons to create
%   numPts = number of points for texton creation
%   maxIter = maximum iterations for k-means to perform
%   method = which method to use 
% Outputs:
%   out = [numClass x neigh^2] matrix of textons

data = []; % initalize data matrix
bOut = randperm(length(files),numImage);
opt = statset('maxIter',maxIter);

if strcmpi(method,'randomBodyPix') == 1
    for i = 1:numImage

        % Get index
        b = bOut(i);

        % Get image and size parameters
        temp = imread([pathToFiles,'/',files(b).name]);
        temp = im2double(temp);

        % Create matrix using [neigh X neigh] sized patches from around numPts 
        % randomly selected pixels 
        [out] = randomNeighborhood(temp,neigh,numPts); % temp is name of file loaded above
        data = [data,out];

    end
elseif strcmpi(method,'centralPath')
    for i = 1:numImage

        % Get index
        b = bOut(i);

        % Get image and size parameters
        temp = imread([pathToFiles,'/',files(b).name]);

        % Create matrix using [neigh X neigh] sized patches from around 
        % all non-edge pixels in the square subregion
        [out] = getNeighborhood(temp,neigh); 
        data = [data,out];

    end
end
[~,out] = kmeans(data',numClass,'options',opt);