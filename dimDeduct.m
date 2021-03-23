%--------------------------------------------------------------------------
% Calculate the initial array for cell state in LSTM
% C: NxN data matrix
% hiddenNum: scalar number
% result: hiddenNum x 1 data matrix
%--------------------------------------------------------------------------
% Copyright @ Flute Xu, 2019
%--------------------------------------------------------------------------

function result = dimDeduct(hiddenNum,C)

% parameter intialization
Ct = C(:);
CtSize = size(Ct);
rng(1);
weight = randn(hiddenNum,CtSize(1));

result = weight * Ct;