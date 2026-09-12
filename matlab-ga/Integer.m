function descriptor = Integer(low, high)
%INTEGER Declare one ordinal integer in the inclusive interval [low, high].
if ~isnumeric(low) || ~isreal(low) || ~isscalar(low) || ~isfinite(low) ...
        || ~isnumeric(high) || ~isreal(high) || ~isscalar(high) || ~isfinite(high) ...
        || low~=fix(low) || high~=fix(high) || double(low)>=double(high) ...
        || ~isfinite(double(high)-double(low))
    error('gahybrid:InvalidInput','Integer bounds must be finite integer-valued numeric scalars with low < high.');
end
if abs(double(low))>flintmax-1 || abs(double(high))>flintmax-1 || double(high)-double(low)>2^52-1
    error('gahybrid:InvalidInput','Integer bound magnitudes must not exceed 2^53-1, and span must not exceed 2^52-1.');
end
descriptor=struct('kind','integer','low',double(low),'high',double(high),'size',1);
end
