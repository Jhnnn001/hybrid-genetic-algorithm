function descriptor = Real(low, high)
%REAL Declare one continuous scalar in the inclusive interval [low, high].
if ~isnumeric(low) || ~isreal(low) || ~isscalar(low) || ~isfinite(low) ...
        || ~isnumeric(high) || ~isreal(high) || ~isscalar(high) || ~isfinite(high) ...
        || double(low)>=double(high) || ~isfinite(double(high)-double(low))
    error('gahybrid:InvalidInput','Real bounds must be finite numeric scalars with low < high and finite span.');
end
descriptor=struct('kind','real','low',double(low),'high',double(high),'size',1);
end
