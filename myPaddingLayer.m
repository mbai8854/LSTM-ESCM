classdef myPaddingLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties.
        % Layer properties go here.
        SizeEdge; 
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Layer learnable parameters go here.
    end
    
    methods
        function layer = myPaddingLayer(SquaredEdgeSize)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Layer constructor function goes here.
            layer.SizeEdge = SquaredEdgeSize;
        end
        
        function [Z] = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X           - Input data
            % Outputs:
            %         Z           - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
            
            Xsize = size(X);
            ii = logical(ones(layer.SizeEdge) - eye(layer.SizeEdge));
            Zsize = [layer.SizeEdge^2, Xsize(2:end)];
            Z = zeros(Zsize, 'like', X);
            Z(ii(:), :,:) = X;   % to do
        end

        %function [Z1, ?, Zm, memory] = forward(layer, X1, ?, Xn)
            % (Optional) Forward input data through the layer at training
            % time and output the result and a memory value.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            %         memory      - Memory value for backward propagation

            % Layer forward function for training goes here.
        %end

        function [dLdX] = backward(layer, ~, ~, dLdZ, ~)
            % Backward propagate the derivative of the loss function through 
            % the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X                 - Input data
            %         Z                 - Outputs of layer forward function            
            %         dLdZ              - Gradients propagated from the next layers
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX              - Derivatives of the loss with respect to the
            %                             inputs
            %         dLdW              - Derivatives of the loss with respect to each
            %                             learnable parameter
            
            % Layer backward function goes here.
            ii = [0: layer.SizeEdge: layer.SizeEdge^2 - layer.SizeEdge] + [1:layer.SizeEdge];
            dLdX = dLdZ;
            dLdX(ii, :, :) = [];
        end
    end
end