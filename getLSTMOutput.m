function output = getLSTMOutput(varargin)
% This function can be used to get the output of a trained lstm layer
% that has been created using the builtin function 'lstmLayer'
%
% USAGE:
% output = getLSTMOutput(varargin)
% output = getLSTMOutput('lstm_layer', lstm_layer, 'input', test_input);
%
% INPUTS:
% 'lstm_layer': Long Short-Term Memory (LSTM) layer created using the
%               function lstmLayer
% 'input': input to be tested (size should match that of the lstm layer
%          input [check: lstm_layer.InputSize]
%
% OUTPUT:
% 'output': output of the lstm layer
%
% $KK 10/20/2017 
% Feel free to the use the function for any purpose :)-
% Also see 'lstmLayer' 'layers'


%%
isLSTM = @(net) isa(net, 'nnet.cnn.layer.LSTMLayer');
p = inputParser;
p.addParameter('lstm_layer', [],isLSTM);
p.addParameter('input', [], @isnumeric);
p.parse(varargin{:});

lstm_layer=p.Results.lstm_layer;
input=p.Results.input;

%% Assign the indices for each layer
totalSize = size(lstm_layer.InputWeights,1);
inputlayer_indices = 1:totalSize/4;
forgetlayer_indices = length(inputlayer_indices)+1:2*(totalSize/4);
layerinput_indices = length(inputlayer_indices)+length(forgetlayer_indices)+1:3*(totalSize/4);
outputlayer_indices = length(inputlayer_indices)+length(forgetlayer_indices)+length(layerinput_indices)+1:4*(totalSize/4);

%% Assign the value for each of the parameters
    W_i = lstm_layer.InputWeights(inputlayer_indices,:); % input gate weights
    W_f = lstm_layer.InputWeights(forgetlayer_indices,:); % forget gate weights
    W_c = lstm_layer.InputWeights(layerinput_indices,:);  % layer input weights
    W_o = lstm_layer.InputWeights(outputlayer_indices,:); % output gate weights
    U_o = lstm_layer.RecurrentWeights(inputlayer_indices,:); % recurrent weighst to input gate
    U_i = lstm_layer.RecurrentWeights(forgetlayer_indices,:); % recurrent weighst to forget gate
    U_f = lstm_layer.RecurrentWeights(layerinput_indices,:);  % recurrent weighst to layer input
    U_c = lstm_layer.RecurrentWeights(outputlayer_indices,:); % recurrent weighst to output gate
    b_i = lstm_layer.Bias(inputlayer_indices); % input gate bias
    b_f = lstm_layer.Bias(forgetlayer_indices); % forget gate bias
    b_c = lstm_layer.Bias(layerinput_indices); % layer input bias
    b_o = lstm_layer.Bias(outputlayer_indices);% output gate bias

    
%% Equations from https://arxiv.org/abs/1603.03827
    c_t0 = lstm_layer.CellState;
    h_t0 = lstm_layer.OutputState;
    i_t=logsig(W_i*input + U_i*h_t0 + b_i);
    f_t=logsig(W_f*input + U_f*h_t0 + b_f);
    c_t1 = tanh(W_c*input + U_c*h_t0 + b_c);
    c_t = f_t.*c_t0 + i_t.*c_t1;
    o_t = logsig(W_o*input + U_o*h_t0 + b_o);
    h_t = o_t.*tanh(c_t);
    
    
    output = h_t;
end