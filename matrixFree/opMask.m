classdef opMask < RegularizationOperator
    properties
        mask
    end
    
    methods
        function this = opMask(sK,type)
            if sK(4)~=sK(3)
                error('sK(4)~=sK(3)');
            end
            this.m = prod(sK);
            
            channels = sK(3);
            mask = zeros(sK(1),sK(2),channels,channels);
            
            for k=1:channels
                mask(:,:,k,k) = 1.0; 
            end
            if type == 1
                mask(2,2,:,:) = 1.0;
%                 this.n = (sK(1)*sK(2)-1)*sK(3) + sK(3)*sK(4);
            end
            this.mask = find(mask~=0.0);
            this.n = length(this.mask);
            this.Amv =  @(x) mul(this,x);
            this.ATmv = @(x) x(this.mask);
        end
        
        function Ax = mul(this,x)
            Ax = zeros(size(x),'like',x);
            Ax(this.mask) = x(:);
        end

        function this = convertGPUorPrecision(this,useGPU,precision)
            % do nothing
        end

    end
end

