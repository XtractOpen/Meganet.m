classdef optimizer
    % classdef optimizer
    %
    % master class for optimizers. Used for dispatching and to collect
    % methods used by all optimizers.
    
    properties
    end
    
    methods
          function [fctn,objFctn,objNames,objFrmt,objHis] = parseObjFctn(this,fctn)
            % if fctn is of type objFctn, return function handles for
            % evaluation and printing
            if exist('fctn','var') && not(isempty(fctn)) && isa(fctn,'objFctn')
                objFctn  = fctn;
                [objNames,objFrmt] = objFctn.hisNames();
                objHis   = @(para) objFctn.hisVals(para);
                fctn = @(x) eval(fctn,x);
            else
                objFctn  = [];
                objNames = {};
                objFrmt  = {};
                objHis   = @(x) [];
            end
        end
    end
    
end

