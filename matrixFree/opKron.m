classdef opKron < RegularizationOperator
    % linear operator for computing kronecker products
    %
    % kron(I,B)*vec(x) = vec(B*mat(x)*I)
    
    properties
    end
    
    methods
        function this = opKron(nI,B)
            this.m = nI*size(B,1);
            this.n = nI*size(B,2);
            this.Amv = @(x) vec(B*reshape(x,size(B,2),[]));
            this.ATmv = @(x)vec(B'*reshape(x,size(B,1),[]));
        end
        
        function getPCop(this,~)
            error('nyi');
        end
        
        function PCmv(A,x,alpha,gamma)
            error('nyi');
        end
        function this = convertGPUorPrecision(this,useGPU,precision)
            % do nothing
        end

    end
end

