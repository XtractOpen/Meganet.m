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
        
        function y = PCmv(this,x)
            y = x;
        end
        
        function this = gpuVar(this,useGPU,precision)
        end

    end
end

