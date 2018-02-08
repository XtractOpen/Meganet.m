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
        
        function y = PCmv(this,x)
            y = x;
        end
        function this = convertGPUorPrecision(this,useGPU,precision)
            % do nothing
        end

    end
end

