


classdef convCuDNN2DSession < handle
    % classdef convCuDNN2DSession < handle
    %
    % Holds the session for cudnn, in order to save allocating and
    % deallocating the session of cuDNN in every convolution.
    % convCuDNN2DSession() allocates memory on the GPU, and initializes CuDNN.
    % This memory is released upon deletion of this class by MATLAB or if
    % an operation fails.
    
    properties
        sessionArray
    end
    
    methods
        function this = convCuDNN2DSession()
%             disp('Session create');
            this.sessionArray = convCuDNN2DSessionCreate_mex();
        end   
        function delete(this)
%             disp('Session destroy');
            convCuDNN2DSessionDestroy_mex(this.sessionArray);
        end
    end
end

