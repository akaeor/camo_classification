function [out] = classifyCamoCritters(pathToFiles, varargin)
% 
% classifyCamoCritters - trains, tests, runs texton classifier for animals
% camouflage
%
%       Author: Eric C. Orenstein
%               Jaffe Laboratory for Underwater Imaging
%               Scripps Institution of Oceanography
%       Date: 01/20/16
%
% This function classifies images of animal camoflauge using a texton based
% image classifier. It assumes that the images are already cropped from
% their background. For details see:
%
%   Orenstein et al., Automated classification of camouflaging cuttlefish.
%   2016
% 
% Note this implimentation requires the follwoing of MATLAB library
% Inputs: 
%   pathToFiles == full path to file containing files of images associated
%                  with classes of interest. Images should be cropped and
%                  in .png format
%
% Outputs:
%   out == a structure containing outputs of the process. This includes:
%
%               'textonMat'     - matrix of vectorized textons 
%                                  (textonDim^2, numTexton*numClasses)
%               'trainIdx'      - index of randomly selected images used for training
%               'trainData'     - matrix of texton histograms for all training 
%                                 images (numTrainIm*numClasses, numTexton*numClasses)
%               'trainLabel'    - vector of labels for training images derived 
%                                 from file name (numTrainIm*numClasses,1) 
%               'classifier'    - matlab svm structure
%               'testData'      - matrix of texton histograms for all test
%                                 data (numTestIm, numTexton*numClasses)
%               'testLabel'     - vector of labels for test images derived 
%                                 from file name (numTestIm, 1)
%               'machineLabels' - vector of labels from the machine classifier
%                                 (numTestIm, 1)
%               'acc'           - accuracy of the automated labels as
%                                 compared to human labels 
%   
% [out] = classifyCamoCritters(..., Name, Value) sets parameters for
% functionality of the system. Name and Value of parameters are:
%
%   'Parallel'          One of the numbers: 0 or 1. Specifies whether or
%                       not to use parallel computing. Parallel computing
%                       takes advantage of all processors on a computer and
%                       speeds the code execution. If toggled, other
%                       computer function maybe slower.
%
%                           0: Do not use parallel processing
%                           1: Use parallel processing
%
%                       Default: 0
%
%   'Method'            One of the strings: 'randomBodyPix' or
%                       'centralPatch'. Selects method for use in both
%                       training, testing, and deployment of the
%                       classifier.
%
%                       Default: 'randomBodyPix'
%
%   'trainOnly'         One of the numbers: 0 or 1. Flag to set to just
%                       train the classifier. This will turn off the
%                       testing phase and only output a trained classifier
%                       and the textons. Use in a deployment scenario.
%                       
%                           0: Train and test
%                           1: Train only
%
%                       Default: 0
%
%   'numTexton'         Any positive integer. Sets the number of textons to
%                       generate per class (ie the number of cluster 
%                       centers used in k-means). 
%
%                       Default: 10
%
%   'numTextonImages'   Any positive integer. Sets the number of images to 
%                       use per class for texton generation.
%
%                       Default: 20
%
%   'numPoints'         Any positive integer. Sets the number of points to 
%                       randomly select in the body subregion for texton 
%                       generation. This is not necessary to set if using 
%                       the central body patch approach.
%
%                       Default: 200
%                       
%   'textonDim'         Any odd integer. Sets the dimensions of the
%                       resulting textons. Must be an odd number to ensure
%                       that there is a central pixel. 
%
%                       Default: 9
%
%   'numTrainImages'    Any integer. Sets the number of images to use as
%                       training examples.
%
%                       Default: 40
%
%   'maxIters'          Any positive integer. Sets the number of iterations
%                       that k-means uses when generating the textons.
%
%                       Default: 200
%
%   'imFormat'          One of the strings 'png', 'jpg', 'tiff'. Format of
%                       the images to be classifier.
%
%                       Default: 'png'
%
%   'evalMethod'        One of the strings 'svm', 'chi'. Determines how to
%                       evaluate. Either applies the SVM using 'svm' or
%                       does a nearest neighbor analysis with a chi^2
%                       distance metric.
%   
%                       Default: 'svm'

%====================================================
% Parse the inputs
[basePath, parallel, method, trainOnly, numTex, ...
    numTexIm, numPts, textonDim, numTrainIm, maxIters, imFormat, evalMethod]...
    = parseInputs(pathToFiles, varargin{:});

%====================================================
% Generate the textons
% Open the local parallel cluster. This will only work if you have multiple
% cpus at your disposal for these computations
if parallel == 1
    parpool % for now, just uses default number of availble cores.
end

files = dir([basePath,'/*']);
files = files(arrayfun(@(x) x.name(1), files) ~='.');

textonMat = [];

fprintf('Generating textons for %d classes\n',length(files))

% Cycle through the classes and create the textons
for ii = 1:length(files)
    ptf = [basePath,'/',files(ii).name];
    imList = dir([ptf,['/*.',imFormat]]);
    
    [texPart] = textonGenerator(ptf, imList, numTexIm, ...
        textonDim, numTex, numPts, maxIters, method);
    
    % Matrix containing all the vectorized textons
    textonMat = [textonMat; texPart];
  
end

% out = textonMat; % this is for testing purposes ECO 020216

%====================================================
% Make the training matrix

fprintf('Done generating textons. Starting to make training matrix\n')

trainIdx = zeros(numTrainIm, length(files)); % numTrainImages x numClasses

trainData = zeros(numTrainIm*length(files), numTex*length(files));
trainLabel = zeros(numTrainIm*length(files),1);
flag = 1;

for ii = 1:length(files)
    fprintf('Making training data for %s\n', files(ii).name)
    % Loop through each class and get file names
    ptf = [basePath,'/',files(ii).name];
    imList = dir([ptf,['/*.',imFormat]]);
    
    % randomly select the training images
    in = randperm(length(imList),numTrainIm);
    trainIdx(:,ii) = in; % store for later use
    
    % make the training matrix
    [part] = makeDataMatrix(ptf, imList, textonMat, in, parallel);

    trainData(flag:(flag+numTrainIm-1), :) = part;
    trainLabel(flag:(flag+numTrainIm-1)) = repmat(ii, numTrainIm, 1);
    flag = flag + numTrainIm;

end

% out = trainData; % this is for testing purposes ECO 020216

%====================================================
% Train the classifier
% Use Mathworks built-in SVM functionality to scale data and train a
% classifier. Default kernal is linear

fprintf('Done making training matrix. Training classifier\n')

svmClassifier = fitcecoc(trainData,trainLabel);

%out = svmClassifier;
%====================================================
% Test the classifier

if trainOnly == 0
    % Start by making the test matrix of texton frequencies. This step may take
    % awhile depending on the amount of data.
    fprintf('Making test matrix\n')

    testData = [];
    testLabel = [];

    for ii = 1:length(files)
        fprintf('Making test data for %s\n', files(ii).name)

        % for each class, get the file names
        ptf = [basePath,'/',files(ii).name];
        imList = dir([ptf,['/*.',imFormat]]);

        % get the indicies of all the images and exclude those used for training
        in = 1:length(imList);
        in(ismember(in,trainIdx(:,ii))) = [];

        % make the matrix
        [part] = makeDataMatrix(ptf, imList, textonMat, in, parallel);
        testData = [testData; part];
        testLabel = [testLabel; repmat(ii,length(in),1)];

    end
    
    % Evaluate
    % predict the class labels using the SVM trained above
    if strcmpi(evalMethod,'svm') == 1
        fprintf('Test matrix complete. Applying classifier\n')
        machineLabels = predict(svmClassifier, testData);
        blah = machineLabels - testLabel;
        test = find(blah);
        acc = (length(testLabel)-length(test))/length(testLabel);
        
    % predict class labels using nearest neighbor with chi^2 distance
    % metric
    elseif strcmpi(evalMethod,'chi') == 1
        chi = [];

        for ii = 1:length(trainData)

            in = trainData(ii,:); 
            rep = repmat(in,size(testData,1),1); % replicate

            % do chi-squared test
            num = (testData - rep).^2;
            den = testData+rep;
            out = .5*sum(num./den,2);
            chi = [chi, out];

        end

        [~,ind] = min(chi,[],2);
        
        % Translate the index of the training sample to a class
        flag = 1;
        machineLabels = zeros(size(ind));
        numClass = length(trainLabel)/numTrainIm;
        
        while flag < length(trainLabel)
            % Loop through each class
            for ii = 1:numClass
                % loop through each sample
                for jj = 1:length(ind)
                    % Check condition
                    if ind(jj) >= flag && ind(jj) < flag+numTrainIm
                        machineLabels(jj) = ii;
                    else
                        continue
                    end
                end
                % increase the flag
                flag = flag + numTrainIm;
            end
        end

        blah = machineLabels - testLabel;
        test = find(blah);
        acc = (length(testLabel)-length(test))/length(testLabel);
    end
end

%====================================================
% Compile everything into a nice output structure

if trainOnly == 0
    out = struct('textonMat', textonMat, 'trainIdx', trainIdx, 'trainData', ...
        trainData, 'trainLabel', trainLabel, 'classifier', svmClassifier, ...
        'testData', testData, 'testLabel', testLabel, 'machineLabels', ...
        machineLabels, 'acc', acc);
else
    out = struct('textonMat', textonMat, 'trainIdx', trainIdx, 'trainData', ...
        trainData, 'trainLabel', trainLabel, 'classifier', svmClassifier);
end

% Shut down the cluster if using parallel computing
if parallel == 1
    fprintf('Closing cluster\n')
    poolobj = gcp('nocreate');
    delete(poolobj);
end

fprintf('Done\n')

%====================================================
% Parse and check inputs
function [basePath, parallel, method, trainOnly, numTex, ...
    numTexIm, numPts, textonDim, numTrainIm, maxIters, ...
    imFormat, evalMethod] = parseInputs(pathToFiles, varargin)

% Define valid values for string calls
methodOptions = {'randomBodyPix','centralPatch'};
formatOptions = {'png', 'jpg', 'tiff'};
evalOptions = {'svm','chi'};

% Create input parser
parser = inputParser;
parser.FunctionName = 'classifyCamoCritters';

% Specifiy the optional parameters (set defaults)
parser.addParameter('Parallel', 0);
parser.addParameter('Method', 'randomBodyPix');
parser.addParameter('trainOnly', 0);
parser.addParameter('numTexton', 10);
parser.addParameter('numTextonImages', 20);
parser.addParameter('numPoints', 200);
parser.addParameter('textonDim', 9);
parser.addParameter('numTrainImages',40);
parser.addParameter('maxIters',200);
parser.addParameter('imFormat','png');
parser.addParameter('evalMethod','svm');

parser.parse(varargin{:});
r = parser.Results;

parallel = r.Parallel;
method = r.Method;
trainOnly = r.trainOnly;
numTex = r.numTexton;
numTexIm = r.numTextonImages;
numPts = r.numPoints;
textonDim = r.textonDim;
numTrainIm = r.numTrainImages;
maxIters = r.maxIters;
imFormat = r.imFormat;
evalMethod = r.evalMethod;
basePath = pathToFiles;

% Check the optional parameters
checkStrings(methodOptions, method, 'method');
checkBinary(parallel, 'Parallel');
checkBinary(trainOnly, 'trainOnly');
checkInt(numTex, 'numTexton');
checkInt(numTexIm, 'numTextonTmages');
checkOdd(textonDim, 'textonDim');
checkInt(numTrainIm, 'numTrainImages');
checkInt(numPts, 'numPoints');
checkInt(maxIters, 'maxIters');
checkStrings(formatOptions, imFormat, 'imFormat');
checkStrings(evalOptions, evalMethod, 'evalMethod');
checkFile(basePath);

%=====================================================
% Check inputs for errors
function r = checkStrings(list, value, string)
potentialMatch = strcmpi(value, list);

if sum(potentialMatch) == 0
    error(...
        'Incorrect %s call. \nCheck if %s in accepted inputs', string, value)
end
r = 1;

function r = checkBinary(value, string)
if ~(value == 0 || value == 1)
    error('%s must be either 0 or 1',string)
end
r = 1;

function r = checkInt(value, string)
if ~(value > 0 && rem(value,1)==0 )
    error('%s must be a positive integer', string)
end
r = 1;

function r = checkOdd(value, string)
if ~(value > 0 && rem(value,1)==0 && mod(value,2)~=0)
    error('%s must be an odd positive integer', string)
end
r = 1;

function r = checkFile(value)
if exist(value,'file') == 0
    error('%s not found. \nCheck if path is correct', value)
end
r = 1;
%=====================================================







