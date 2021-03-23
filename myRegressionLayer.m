classdef myRegressionLayer < nnet.layer.RegressionLayer
    properties
    % (Optional) Layer properties.
    % Layer properties go here.
    
       l1_reg = 0.0;
    end
    methods
        function layer = myRegressionLayer(name, l1_reg)
        % (Optional) Create a myRegressionLayer.
        % Layer constructor function goes here.
            layer.Name = name;
            layer.l1_reg = l1_reg;
            layer.Description = 'Self-Representative error';
        end
        function loss = forwardLoss(layer, Y, T)
        % Return the loss between the predictions Y and the
        % training targets T.
        %
        % Inputs:
        % layer - Output layer
        % Y ? Predictions made by network
        % T ? Training targets
        %
        % Output:
        % loss - Loss between Y and T
        % Layer forward loss function goes here.
        
            Ysize = size(Y);
            squaredSize = sqrt(Ysize(1));
            Y = reshape(Y, [squaredSize, squaredSize, Ysize(2:end)]);
            T = reshape(T, [squaredSize, squaredSize, Ysize(2:end)]); 
            
            % This could be slow.  To do list
            loss = 0.0;
            for i = 1:Ysize(2)
                for j = 1:Ysize(3)
                    loss = loss + 0.5*trace(T(:,:,i,j)*((eye(squaredSize) - Y(:,:,i,j))*(eye(squaredSize) - Y(:,:,i,j))')) ...
                         + layer.l1_reg * sum(sum(sum(abs(Y(:,:,i,j)))));  
                end
            end
            loss  = loss / (Ysize(1)*Ysize(3));
        end
        function dLdY = backwardLoss(layer, Y, T)
        % Backward propagate the derivative of the loss function.
        %
        % Inputs:
        % layer - Output layer
        % Y ? Predictions made by network
        % T ? Training targets
        %
        % Output:
        % dLdY - Derivative of the loss with respect to the predictions Y
        % Layer backward loss function goes here. 
            Ysize = size(Y);
%             disp('test')
            squaredSize = sqrt(Ysize(1));
            dLdY = zeros([squaredSize, squaredSize, Ysize(2:end)], 'like', Y);
            Y = reshape(Y, [squaredSize, squaredSize, Ysize(2:end)]);
            T = reshape(T, [squaredSize, squaredSize, Ysize(2:end)]);
            for i = 1:Ysize(2)
                for j = 1:Ysize(3)
                    dLdY(:,:,i,j) = 0.5 * (T(:,:,i,j) + T(:,:,i,j)')*(Y(:,:,i,j) - eye(squaredSize)) + layer.l1_reg * sign(Y(:,:,i,j));
                end
            end
            dLdY = dLdY / (Ysize(1)*Ysize(3));
            dLdY = reshape(dLdY, Ysize);
        end
    end
end