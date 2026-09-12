function descriptor = Permutation(size)
%PERMUTATION Declare an ordering of 1:size, where size is at least two.
if ~isnumeric(size) || ~isreal(size) || ~isscalar(size) || ~isfinite(size) || size<2 || size~=fix(size)
    error('gahybrid:InvalidInput','Permutation size must be an integer-valued numeric scalar of at least two.');
end
descriptor=struct('kind','permutation','low',1,'high',double(size),'size',double(size));
end
