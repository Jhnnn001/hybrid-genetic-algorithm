classdef GA < handle
    %GA Native mixed-variable genetic minimization: split, flat, or relax.
    % GA(objective,space,'option',value,...) accepts the full names in README.
    % The objective receives named physical values in a fresh scalar struct.
    % Split crosses the real, integer/binary, and permutation parts separately;
    % flat crosses the whole chromosome; relax uses continuous genes with
    % discrete decoding, including random keys.
    properties (SetAccess=private)
        space
        method
        rng
        pop_size
        max_gen
        gen = 0
        population = []
        fitness = []
        best_u = []
        best_x = []
        best_f = Inf
        n_evals = 0
        diversity = NaN
    end
    properties (Access=private)
        objective
        options
        operators
    end
    methods
        function obj=GA(objective,space,varargin)
            if ~isa(objective,'function_handle') || ~isscalar(objective)
                error('gahybrid:InvalidInput','objective must be a function handle.');
            end
            if ~isa(space,'Space') || ~isscalar(space)
                error('gahybrid:InvalidInput','space must be a scalar Space.');
            end
            k=struct('method','split','pop_size',30,'max_gen',200,'seed',[], ...
                'callback',[],'init','random','selection',[], ...
                'crossover_real',[],'crossover_int',[],'crossover_perm',[], ...
                'mutation_real',[],'mutation_int',[],'mutation_perm',[], ...
                'crossover_prob',.9,'mutation_prob',.1,'elitism',1, ...
                'tournament_size',3,'k_points',2,'blx_alpha',.5,'sbx_eta',2, ...
                'sigma',.1,'boltzmann_t',10,'boltzmann_alpha',.95, ...
                'hop_steps',[-2 -1 1 2],'heuristic_factor',3);
            if mod(numel(varargin),2)
                error('gahybrid:InvalidInput','GA options require complete name/value pairs.');
            end
            for i=1:2:numel(varargin)
                name=varargin{i};
                if isstring(name) && isscalar(name), name=char(name); end
                if ~ischar(name) || ~isrow(name)
                    error('gahybrid:InvalidInput','GA option names must be character vectors or string scalars.');
                end
                if ~isfield(k,name)
                    error('gahybrid:InvalidInput','Unknown GA option %s; use complete case-sensitive names.',name);
                end
                k.(name)=varargin{i+1};
            end
            k.method=lower(optionText(k.method,'method',false));
            if ~any(strcmp(k.method,{'split','flat','relax'}))
                error('gahybrid:InvalidInput','method must be split, flat, or relax; scipy is Python-only.');
            end
            for name={'pop_size','max_gen','elitism','tournament_size','k_points','heuristic_factor'}
                value=number(k.(name{1}),name{1});
                if value~=fix(value), error('gahybrid:InvalidInput','%s must be integer-valued.',name{1}); end
                k.(name{1})=value;
            end
            if k.pop_size<2, error('gahybrid:InvalidInput','pop_size must be at least two.'); end
            if k.max_gen<1, error('gahybrid:InvalidInput','max_gen must be at least one.'); end
            if k.elitism<0 || k.elitism>=k.pop_size
                error('gahybrid:InvalidInput','elitism must be between zero and pop_size-1.');
            end
            if k.tournament_size<1 || k.tournament_size>k.pop_size
                error('gahybrid:InvalidInput','tournament_size must be between one and pop_size.');
            end
            if k.k_points<1, error('gahybrid:InvalidInput','k_points must be at least one.'); end
            if k.heuristic_factor<1, error('gahybrid:InvalidInput','heuristic_factor must be at least one.'); end
            for name={'crossover_prob','mutation_prob','blx_alpha','sbx_eta','sigma','boltzmann_t','boltzmann_alpha'}
                k.(name{1})=number(k.(name{1}),name{1});
            end
            for name={'crossover_prob','mutation_prob'}
                if k.(name{1})<0 || k.(name{1})>1
                    error('gahybrid:InvalidInput','%s must lie in [0,1].',name{1});
                end
            end
            if k.blx_alpha<0, error('gahybrid:InvalidInput','blx_alpha must be nonnegative.'); end
            for name={'sbx_eta','sigma','boltzmann_t','boltzmann_alpha'}
                if k.(name{1})<=0, error('gahybrid:InvalidInput','%s must be positive.',name{1}); end
            end
            if k.boltzmann_alpha>1, error('gahybrid:InvalidInput','boltzmann_alpha must be at most one.'); end
            if ~isnumeric(k.hop_steps) || ~isreal(k.hop_steps) || ~isvector(k.hop_steps) ...
                    || isempty(k.hop_steps) || any(~isfinite(k.hop_steps(:))) ...
                    || any(k.hop_steps(:)==0 | k.hop_steps(:)~=fix(k.hop_steps(:)))
                error('gahybrid:InvalidInput','hop_steps must be a nonempty vector of nonzero integers.');
            end
            k.hop_steps=double(k.hop_steps(:).');
            if ~isempty(k.callback) && (~isa(k.callback,'function_handle') || ~isscalar(k.callback))
                error('gahybrid:InvalidInput','callback must be empty or a function handle.');
            end
            op=ga_operators(); k.init=optionText(k.init,'init',false);
            k.selection=optionText(k.selection,'selection',true);
            if isempty(k.selection), k.selection='tournament'; end
            if ~isfield(op.INIT,k.init), error('gahybrid:InvalidInput','Unknown init operator %s.',k.init); end
            if ~isfield(op.SELECTION,k.selection), error('gahybrid:InvalidInput','Unknown selection operator %s.',k.selection); end
            names={'crossover_real','crossover_int','crossover_perm','mutation_real','mutation_int','mutation_perm'};
            defaults={'sbx','uniform','ox','gaussian','hop','swap'};
            used=true(1,6);
            if strcmp(k.method,'flat'), used([2 3])=false; defaults{1}='single_point'; end
            if strcmp(k.method,'relax'), used([2 3 5 6])=false; end
            present=[~isempty(space.real_idx) ~isempty(space.int_idx) ~isempty(space.perm_slices)];
            exists=[present present];
            if ~strcmp(k.method,'split'), exists(1)=true; end
            if strcmp(k.method,'relax'), exists(4)=true; end
            for i=1:6
                name=names{i}; value=optionText(k.(name),name,true);
                if ~isempty(value) && ~used(i)
                    error('gahybrid:InvalidInput','%s operator %s is unused by method %s.',name,value,k.method);
                end
                if ~isempty(value) && ~exists(i)
                    error('gahybrid:InvalidInput','%s operator %s needs its variable part in Space.',name,value);
                end
                if isempty(value) && used(i) && exists(i), value=defaults{i}; end
                if ~isempty(value) && ~isfield(op.(upper(name)),value)
                    error('gahybrid:InvalidInput','Unknown %s operator %s.',name,value);
                end
                k.(name)=value;
            end
            if strcmp(k.mutation_int,'bit_flip') && ~space.all_binary
                error('gahybrid:InvalidInput','bit_flip requires every integer-part gene to be Binary.');
            end
            if any(strcmp(k.mutation_int,{'swap','inversion','scramble','displacement'})) ...
                    && (numel(space.int_idx)<2 || any(space.int_lo~=space.int_lo(1)) || any(space.int_hi~=space.int_hi(1)))
                error('gahybrid:InvalidInput','%s requires at least two integer genes with identical bounds.',k.mutation_int);
            end
            if isempty(k.seed)
                stream=RandStream('mt19937ar','Seed','shuffle');
            elseif isa(k.seed,'RandStream') && isscalar(k.seed)
                stream=k.seed;
            else
                value=number(k.seed,'seed');
                if value<0 || value>2^32-1 || value~=fix(value)
                    error('gahybrid:InvalidInput','seed must be empty, a RandStream, or an integer in [0,2^32-1].');
                end
                stream=RandStream('mt19937ar','Seed',value);
            end
            obj.objective=objective; obj.space=space; obj.method=string(k.method);
            obj.rng=stream; obj.pop_size=k.pop_size; obj.max_gen=k.max_gen;
            obj.options=k; obj.operators=op;
        end

        function result=run(obj)
            %RUN Start fresh while continuing this instance's random stream.
            k=obj.options; n=obj.pop_size; grid=obj.method~="relax";
            obj.n_evals=0; obj.gen=0; obj.best_f=Inf; obj.best_u=[]; obj.best_x=[];
            obj.population=[]; obj.fitness=[]; obj.diversity=NaN;
            U=obj.operators.INIT.(k.init)(obj.rng,obj.space,n,grid,k);
            f=obj.evaluate(U);
            if strcmp(k.init,'heuristic')
                keep=zeros(n,1);
                for i=1:n
                    group=(i-1)*k.heuristic_factor+(1:k.heuristic_factor);
                    [~,j]=min(f(group)); keep(i)=group(j);
                end
                U=U(keep,:); f=f(keep);
            end
            obj.population=U; obj.fitness=f;
            obj.diversity=2*mean(std(U,1,1));
            history=zeros(obj.max_gen+1,1); spreads=history;
            history(1)=obj.best_f; spreads(1)=obj.diversity;
            for generation=1:obj.max_gen
                parents=obj.operators.SELECTION.(k.selection)(obj.rng,f,n,generation,k);
                parents=parents(randperm(obj.rng,n));
                [~,order]=sort(f,'ascend'); elite=order(1:k.elitism);
                count=n-k.elitism; children=zeros(count,obj.space.n_genes);
                cursor=1;
                while cursor<=count
                    a=U(parents(mod(cursor-1,n)+1),:); b=U(parents(mod(cursor,n)+1),:);
                    if rand(obj.rng)<k.crossover_prob, [a,b]=obj.cross(a,b); end
                    a=obj.mutate(a,generation); b=obj.mutate(b,generation);
                    children(cursor,:)=a;
                    if cursor<count, children(cursor+1,:)=b; end
                    cursor=cursor+2;
                end
                if grid
                    for i=1:count, children(i,:)=obj.space.snap(children(i,:)); end
                else
                    children=min(1,max(0,children));
                end
                fc=obj.evaluate(children);
                U=[U(elite,:);children]; f=[f(elite);fc];
                obj.population=U; obj.fitness=f; obj.gen=generation;
                obj.diversity=2*mean(std(U,1,1));
                history(generation+1)=obj.best_f; spreads(generation+1)=obj.diversity;
                if ~isempty(k.callback), k.callback(obj); end
            end
            result=struct('x',obj.best_x,'fun',obj.best_f,'history',history, ...
                'diversity',spreads,'n_gen',obj.gen,'n_evals',obj.n_evals,'method',obj.method);
        end
    end
    methods (Access=private)
        function f=evaluate(obj,U)
            f=zeros(size(U,1),1);
            for i=1:size(U,1)
                params=obj.space.decode(U(i,:)); obj.n_evals=obj.n_evals+1;
                value=obj.objective(params);
                if ~isnumeric(value) || ~isscalar(value) || ~isreal(value)
                    error('gahybrid:ObjectiveValue','objective must return one real numeric scalar.');
                end
                value=double(value); if isnan(value), value=Inf; end
                f(i)=value;
                if isempty(obj.best_u) || value<obj.best_f
                    obj.best_f=value; obj.best_u=U(i,:); obj.best_x=obj.space.decode(obj.best_u);
                end
            end
        end

        function [a,b]=cross(obj,a,b)
            k=obj.options; op=obj.operators; s=obj.space;
            if obj.method~="split"
                [a,b]=op.CROSSOVER_REAL.(k.crossover_real)(obj.rng,a,b,k);
                if obj.method=="flat", a=s.snap(a); b=s.snap(b); end
                return;
            end
            if ~isempty(s.real_idx)
                [a(s.real_idx),b(s.real_idx)]=op.CROSSOVER_REAL.(k.crossover_real)(obj.rng,a(s.real_idx),b(s.real_idx),k);
            end
            if ~isempty(s.int_idx)
                [va,vb]=op.CROSSOVER_INT.(k.crossover_int)(obj.rng,s.get_ints(a),s.get_ints(b),k);
                a=s.set_ints(a,va); b=s.set_ints(b,vb);
            end
            for i=1:numel(s.perm_slices)
                block=s.perm_slices{i};
                [pa,pb]=op.CROSSOVER_PERM.(k.crossover_perm)(obj.rng,s.get_perm(a,block),s.get_perm(b,block),k);
                a=s.set_perm(a,block,pa); b=s.set_perm(b,block,pb);
            end
        end

        function c=mutate(obj,c,generation)
            k=obj.options; op=obj.operators; s=obj.space;
            if obj.method=="relax"
                if rand(obj.rng)<k.mutation_prob
                    c=op.MUTATION_REAL.(k.mutation_real)(obj.rng,c,generation,obj.max_gen,k);
                end
                return;
            end
            if ~isempty(s.real_idx) && rand(obj.rng)<k.mutation_prob
                c(s.real_idx)=op.MUTATION_REAL.(k.mutation_real)(obj.rng,c(s.real_idx),generation,obj.max_gen,k);
            end
            if ~isempty(s.int_idx) && rand(obj.rng)<k.mutation_prob
                values=op.MUTATION_INT.(k.mutation_int)(obj.rng,s.get_ints(c),s.int_lo,s.int_hi,k);
                c=s.set_ints(c,values);
            end
            for i=1:numel(s.perm_slices)
                if rand(obj.rng)<k.mutation_prob
                    block=s.perm_slices{i}; p=op.MUTATION_PERM.(k.mutation_perm)(obj.rng,s.get_perm(c,block));
                    c=s.set_perm(c,block,p);
                end
            end
        end
    end
end

function value=number(value,name)
if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value)
    error('gahybrid:InvalidInput','%s must be a finite real numeric scalar.',name);
end
value=double(value);
end

function value=optionText(value,name,allowEmpty)
if isstring(value) && isscalar(value), value=char(value); end
if allowEmpty && isempty(value), value=''; return; end
if ~ischar(value) || ~isrow(value) || isempty(value)
    error('gahybrid:InvalidInput','%s must be a character vector or string scalar.',name);
end
end
