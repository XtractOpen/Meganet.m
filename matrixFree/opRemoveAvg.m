classdef opRemoveAvg < LinearOperator
    % 
    
    properties
       sK
    end
    
    methods
        function this = opRemoveAvg(sK)
            this.sK = sK;
            this.m = prod(sK);
            this.n = prod(sK);
            this.Amv = @(x) removeMean(this,x);
            this.ATmv = @(x) removeMean(this,x);
        end
        
        function xm = removeMean(this,x)
            szx = size(x);
            x = reshape(x,prod(this.sK(1:2)),[]);
            xm = reshape(x-mean(x,1),szx);
        end
      
        function this = convertGPUorPrecision(this,useGPU,precision)
            % do nothing
        end

    end
end

