function [activations] = myActivations(net,data,layer_no)
if ~isnumeric(layer_no)
    warning('layer_no (3rd argument) should be an integer, representing index of layer activating')
elseif or(layer_no>(size(net.Layers,1)-1),layer_no<2)
    warning(strcat('layer_no exceeds network size, select a number between 2 and ',num2str((size(net.Layers,1)-1))))
end
if string(class(net.Layers((size(net.Layers,1)))))=="nnet.cnn.layer.RegressionOutputLayer"
    net_new=net.Layers([ 1:layer_no (size(net.Layers,1)) ]);
    % pretty straightforward when a regression network
    net_new=SeriesNetwork(net_new);
elseif layer_no==(size(net.Layers,1))-1
    warning(strcat('layer_no exceeds network size, select a number between 2 and ',num2str((size(net.Layers,1)-2)),'. For Softmax, use multiple output arguments with =classify()'))
else
    % We're going to have to cut off classificationOutput and replace with regression to convert layers back to a 'valid system' for predict command
    net_cut=net.Layers(1:layer_no);
    layers = [ ...
        net_cut
        regressionLayer]; % has to be a regression layer in order to be a 'valid system'
    net_new=SeriesNetwork(layers);
end
activations=predict(net_new,data);