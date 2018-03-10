classdef opEye < RegularizationOperator
    % identity operator
    
    properties
    end
    
    methods
        function this = opEye(n)
            this.m = n;
            this.n = n;
            this.Amv = @(x) x;
            this.ATmv = @(x) x;
        end
        
        function PCop = getPCop(this,~)
            PCop = this;
        end
        
        function y = PCmv(A,x,alpha,gamma)
            % x = argmin_x alpha/2*|A*x|^2+gamma/2*|x-y|^2
            if not(exist('alpha','var')) || isempty(alpha)
                alpha = 1;
            end
            if not(exist('gamma','var')) || isempty(gamma)
                gamma = 0;
            end
            
            y = x/(alpha+gamma);
        end
        function this = convertGPUorPrecision(this,useGPU,precision)
            % do nothing
        end

    end
end

