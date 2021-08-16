path = '*';
addpath(genpath(path));

dataNameSet = {'MSRA-6view' 'still-2-mtv' 'Caltech101-7' 'proteinFold'  'nonpl' 'flower17' 'flower102' 'Reuters' };
dataName = 'Reuters';
load([path,'datasets\',dataName,'_Kmatrix'],'KH','Y');
numclass = length(unique(Y));
numker = size(KH,3);
num = size(KH,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
KH = kcenter(KH);
KH = knorm(KH);
options.seuildiffsigma=1e-5;        % stopping criterion for weight variation
%------------------------------------------------------
% Setting some numerical parameters
%------------------------------------------------------
options.goldensearch_deltmax=1e-1; % initial precision of golden section search
options.numericalprecision=1e-16;   % numerical precision weights below this value
% are set to zero
%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
options.firstbasevariable='first'; % tie breaking method for choosing the base
% variable in the reduced gradient method
options.nbitermax=500;             % maximal number of iteration
options.seuil=0;                   % forcing to zero weights lower than this
options.seuilitermax=10;           % value, for iterations lower than this one
options.miniter=0;                 % minimal number of iterations
options.threshold = 1e-4;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;
Sigma = ones(numker,1)/numker;
avgKer  = mycombFun(KH,Sigma);
tauset = [0.05:0.05:0.95];
res_mean = zeros(4,length(tauset));
res_std = zeros(4,length(tauset));
Sigma_ = zeros(numker,length(tauset));
for it =1:length(tauset)
	numSel = round(tauset(it)*num);
	NS = genarateNeighborhood(avgKer,numSel);
	%%--Calculate Neighborhood--%%%%%%
	A = zeros(num);
	for i =1:num
		A(NS(:,i),NS(:,i)) = A(NS(:,i),NS(:,i))+1;
	end
	[H_normalized,Sigma(:,it),obj] = localSimpleMKKM(KH,numclass,A,options);
	[res_mean(:,it),res_std(:,it)] = myNMIACCV2(H_normalized,Y,numclass);
end
timecost = toc;

save(*,'res_mean','res_std','Sigma','timecost'); 
