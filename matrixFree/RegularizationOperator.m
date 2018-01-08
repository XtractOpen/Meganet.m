classdef RegularizationOperator < LinearOperator
    % classdef RegularizationOperator < LinearOperator
    %
    % superclass for regularization operators used in Tikhonov norms
    %
    % R(x) = 0.5 *  | L*x|^2
    
    
    
    properties
        PC        % preconditioner for L'*L
        useGPU    % flag for GPU computing
        precision % flag for precision of computation 
    end
    
    methods
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.useGPU = value;
                this = gpuVar(this,this.useGPU,this.precision);
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.precision = value;
                this = gpuVar(this,this.useGPU,this.precision);
            end
        end    
    end
       
end

