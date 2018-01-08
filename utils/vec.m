function v = vec(v)

if iscell(v)
    v = vec(cell2mat(cellfun(@vec,v,'UniformOutput',false)));
else
    v = v(:);
end