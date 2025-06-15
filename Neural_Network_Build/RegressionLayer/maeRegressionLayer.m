classdef maeRegressionLayer < nnet.layer.RegressionLayer
    methods
        function layer = maeRegressionLayer(name)

            if nargin == 1
                layer.Name = name;
            end
            layer.Description = 'Mean Absolute Error Regression Layer';
        end
        
        function loss = forwardLoss(layer, Y, T)

            loss = mean(abs(Y - T), 'all');
        end
    end
end
