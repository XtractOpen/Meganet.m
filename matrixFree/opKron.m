classdef opKron < RegularizationOperator
    % linear operator for computing kronecker products
    %
    % kron(A,B)*vec(x) = vec(B*mat(x)*A')
    
    properties
        A
        B 
    end
    
    methods
        function this = opKron(A,B)
            this.m = size(A,1)*size(B,1);
            this.n = size(A,2)*size(B,2);
            this.B = B;
            this.A = A;
            this.Amv = @(x) this.matvec(x);
            this.ATmv = @(x) this.matvecT(x);
        end

        function x = matvec(this,x)
            nrhs = round(numel(x)/this.n);
            if nrhs == 1
                x = vec((this.A*((this.B*reshape(x,size(this.B,2),[]))'))');
            else
                Bx = this.B*reshape(x,size(this.B,2),[]);
                Bx = permute(reshape(Bx,size(this.B,1),[],nrhs),[2,1,3]);
                Bx = reshape(Bx,size(this.A,2),[]);
                BxAt = this.A*Bx;
                BxAt = permute(reshape(BxAt,size(this.A,1),[],nrhs),[2,1,3]);
                x = reshape(BxAt,[],nrhs);
            end
        end

        function x = matvecT(this,x)
            nrhs = round(numel(x)/this.m);
            if nrhs == 1
                x = vec((this.A'*((this.B'*reshape(x,size(this.B,1),[]))'))');
            else
                Btx = this.B'*reshape(x,size(this.B,1),[]);
                Btx = permute(reshape(Btx, size(this.B,2),[],nrhs),[2,1,3]);
                Btx = reshape(Btx,size(this.A,1),[]);
                BtxA = this.A'*Btx;
                BtxA = permute(reshape(BtxA,size(this.A,2),[],nrhs),[2,1,3]);
                x = reshape(BtxA,[],nrhs);
            end
        end
      
        function PC = getPCop(this,~)
            PC = opKron(getPCop(this.A), getPCop(this.B));
        end
        
        function PCmv(A,x,alpha,gamma)            
            error('nyi');
        end
        function this = convertGPUorPrecision(this,useGPU,precision)
            % do nothing
        end

    end
end

