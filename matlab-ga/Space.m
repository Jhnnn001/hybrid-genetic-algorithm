classdef Space
    %SPACE Named mixed-variable encoding in unit coordinates.
    % Space(name, descriptor, ...) preserves the declared gene order.
    properties (SetAccess=private)
        names = {}
        n_genes = 0
        real_idx = []
        int_idx = []
        int_lo = []
        int_hi = []
        perm_slices = {}
        all_binary = true
    end
    properties (Access=private)
        variables = {}
        blocks = {}
        binary_positions = []
    end
    methods
        function obj=Space(varargin)
            if isempty(varargin) || mod(numel(varargin),2)
                error('gahybrid:InvalidInput','Space needs at least one complete name/descriptor pair.');
            end
            for i=1:2:numel(varargin)
                name=varargin{i}; d=varargin{i+1};
                if isstring(name) && isscalar(name), name=char(name); end
                if ~ischar(name) || ~isrow(name) || ~isvarname(name) || any(strcmp(obj.names,name))
                    error('gahybrid:InvalidInput','Space variable names must be unique valid MATLAB field names.');
                end
                if ~isstruct(d) || ~isscalar(d) || ~all(isfield(d,{'kind','low','high','size'}))
                    error('gahybrid:InvalidInput','Space variable %s requires a valid descriptor.',name);
                end
                if ~(ischar(d.kind) && isrow(d.kind)) && ~(isstring(d.kind) && isscalar(d.kind))
                    error('gahybrid:InvalidInput','Space variable %s has an invalid kind.',name);
                end
                switch char(d.kind)
                    case 'real', canonical=Real(d.low,d.high);
                    case 'integer', canonical=Integer(d.low,d.high);
                    case 'binary', canonical=Binary(d.size);
                    case 'permutation', canonical=Permutation(d.size);
                    otherwise, error('gahybrid:InvalidInput','Space variable %s has an unknown descriptor kind.',name);
                end
                fields={'low','high','size'};
                for j=1:3
                    value=d.(fields{j});
                    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isequal(double(value),canonical.(fields{j}))
                        error('gahybrid:InvalidInput','Space variable %s has an invalid %s.',name,fields{j});
                    end
                end
                idx=obj.n_genes+(1:canonical.size);
                obj.names{end+1}=name; obj.variables{end+1}=canonical;
                obj.blocks{end+1}=idx; obj.n_genes=obj.n_genes+canonical.size;
                switch canonical.kind
                    case 'real', obj.real_idx=[obj.real_idx idx];
                    case {'integer','binary'}
                        if strcmp(canonical.kind,'binary')
                            obj.binary_positions=[obj.binary_positions numel(obj.int_idx)+(1:canonical.size)];
                        end
                        obj.int_idx=[obj.int_idx idx];
                        obj.int_lo=[obj.int_lo repmat(canonical.low,1,canonical.size)];
                        obj.int_hi=[obj.int_hi repmat(canonical.high,1,canonical.size)];
                        obj.all_binary=obj.all_binary && strcmp(canonical.kind,'binary');
                    case 'permutation', obj.perm_slices{end+1}=idx;
                end
            end
        end

        function params=decode(obj,u)
            %DECODE Return a fresh struct of physical values for one chromosome.
            u=obj.row(u); u=min(1,max(0,u)); params=struct();
            for i=1:numel(obj.names)
                d=obj.variables{i}; g=u(obj.blocks{i});
                switch d.kind
                    case 'real'
                        if g==0, value=d.low;
                        elseif g==1, value=d.high;
                        else, value=min(d.high,max(d.low,d.low+g*(d.high-d.low))); end
                    case 'integer', value=min(d.high,max(d.low,d.low+floor(g*(d.high-d.low)+.5)));
                    case 'binary', value=g>=.5;
                    case 'permutation', [~,value]=sort(g,'ascend');
                end
                params.(obj.names{i})=value;
            end
        end

        function v=get_ints(obj,u)
            %GET_INTS Decode the Integer and Binary genes in genome order.
            u=obj.row(u); g=min(1,max(0,u(obj.int_idx)));
            v=min(obj.int_hi,max(obj.int_lo,obj.int_lo+floor(g.*(obj.int_hi-obj.int_lo)+.5)));
            v(obj.binary_positions)=g(obj.binary_positions)>=.5;
        end

        function u=set_ints(obj,u,v)
            %SET_INTS Return a chromosome with encoded integer values.
            u=obj.row(u);
            if ~isnumeric(v) || ~isreal(v) || numel(v)~=numel(obj.int_idx) ...
                    || any(~isfinite(v(:))) || any(v(:)~=fix(v(:)))
                error('gahybrid:InvalidInput','set_ints needs one finite integer per integer gene.');
            end
            v=double(v(:).');
            if any(v<obj.int_lo | v>obj.int_hi)
                error('gahybrid:InvalidInput','set_ints values must lie within integer bounds.');
            end
            u(obj.int_idx)=(v-obj.int_lo)./(obj.int_hi-obj.int_lo);
        end

        function p=get_perm(obj,u,block)
            %GET_PERM Decode one permutation block, breaking key ties stably.
            u=obj.row(u); obj.checkBlock(block);
            [~,p]=sort(u(block),'ascend');
        end

        function u=set_perm(obj,u,block,p)
            %SET_PERM Return a chromosome with inverse-rank permutation keys.
            u=obj.row(u); obj.checkBlock(block); count=numel(block);
            if ~isnumeric(p) || ~isreal(p) || ~isequal(sort(double(p(:).')),1:count)
                error('gahybrid:InvalidInput','set_perm needs a permutation of 1:block_size.');
            end
            inverse=zeros(1,count); inverse(p)=1:count;
            u(block)=(inverse-1)/(count-1);
        end

        function u=snap(obj,u)
            %SNAP Clip real genes and put every discrete gene on its grid.
            u=min(1,max(0,obj.row(u)));
            u=obj.set_ints(u,obj.get_ints(u));
            for i=1:numel(obj.perm_slices)
                block=obj.perm_slices{i}; u=obj.set_perm(u,block,obj.get_perm(u,block));
            end
        end

        function U=random(obj,stream,n,grid)
            %RANDOM Draw n candidates, uniformly on each discrete grid if grid.
            if ~isa(stream,'RandStream') || ~isnumeric(n) || ~isscalar(n) || ~isreal(n) || ~isfinite(n) || n<1 || n~=fix(n)
                error('gahybrid:InvalidInput','random needs a RandStream and a positive integer n.');
            end
            if ~(islogical(grid) || isnumeric(grid)) || ~isscalar(grid) || ~any(grid==[0 1])
                error('gahybrid:InvalidInput','random grid must be a scalar boolean.');
            end
            U=rand(stream,n,obj.n_genes);
            if grid
                for j=1:numel(obj.int_idx)
                    lo=obj.int_lo(j); hi=obj.int_hi(j);
                    U(:,obj.int_idx(j))=(randi(stream,[lo hi],n,1)-lo)/(hi-lo);
                end
                for i=1:n
                    for j=1:numel(obj.perm_slices)
                        block=obj.perm_slices{j};
                        U(i,:)=obj.set_perm(U(i,:),block,randperm(stream,numel(block)));
                    end
                end
            end
        end
    end
    methods (Access=private)
        function u=row(obj,u)
            if ~isnumeric(u) || ~isreal(u) || ~isvector(u) || numel(u)~=obj.n_genes || any(~isfinite(u(:)))
                error('gahybrid:InvalidInput','Chromosome must be a finite numeric vector of n_genes elements.');
            end
            u=double(u(:).');
        end
        function checkBlock(obj,block)
            if ~any(cellfun(@(b)isequal(b,block),obj.perm_slices))
                error('gahybrid:InvalidInput','Permutation block must belong to this Space.');
            end
        end
    end
end
