function descriptor = Binary(size)
%BINARY Declare a logical row vector; its default length is one.
if nargin==0, size=1; end
if ~isnumeric(size) || ~isreal(size) || ~isscalar(size) || ~isfinite(size) || size<1 || size~=fix(size)
    error('gahybrid:InvalidInput','Binary size must be a positive integer-valued numeric scalar.');
end
descriptor=struct('kind','binary','low',0,'high',1,'size',double(size));
end
