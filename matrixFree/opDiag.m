classdef opDiag < RegularizationOperator
    % identity operator
    
    properties
        D
    end
    
    methods
        function this = opDiag(D)
            n = numel(D);
            this.m = n;
            this.n = n;
            this.D = D;
            this.Amv = @(x) this.D(:).*x;
            this.ATmv = @(x) this.D(:).*x;
        end
        
        function PCop = getPCop(this,~)
            PCop = this;
        end
        
        function y = PCmv(this,x,alpha,gamma)
            % x = argmin_x alpha/2*|D*x|^2+gamma/2*|x-y|^2
            % minimum norm solution when rank-deficient
            if not(exist('alpha','var')) || isempty(alpha)
                alpha = 1;
            end
            if not(exist('gamma','var')) || isempty(gamma)
                gamma = 0;
            end
            
            s = 1./(alpha*this.D.^2 + gamma);
            s(isnan(s))=0;
            s(isinf(s))=0;
            
            y = x.*s;
        end
        function this = convertGPUorPrecision(this,useGPU,precision)
            % do nothing
        end

    end
end

