function op=ga_operators()
%GA_OPERATORS Internal pure operator tables; every draw uses the given stream.
% Crossover: (stream,a,b,knobs). Selection: (stream,f,n,gen,knobs).
op.INIT=struct('random',@randomInit,'chaotic',@chaoticInit,'heuristic',@heuristicInit);
op.SELECTION=struct('roulette',@(r,f,n,g,k)weighted(r,f,n,g,k,'roulette'), ...
    'sus',@(r,f,n,g,k)weighted(r,f,n,g,k,'sus'), ...
    'rank',@(r,f,n,g,k)weighted(r,f,n,g,k,'rank'), ...
    'tournament',@tournament,'boltzmann',@(r,f,n,g,k)weighted(r,f,n,g,k,'boltzmann'));
op.CROSSOVER_REAL=struct('single_point',@singlePoint,'k_point',@kPoint, ...
    'uniform',@uniformCross,'arithmetic',@arithmetic,'blx',@blx, ...
    'linear',@linear,'sbx',@sbx);
op.CROSSOVER_INT=struct('single_point',@singlePoint,'k_point',@kPoint, ...
    'uniform',@uniformCross,'shuffle',@shuffleCross);
op.CROSSOVER_PERM=struct('pmx',@pmx,'ox',@ox);
op.MUTATION_REAL=struct('uniform',@uniformReal,'gaussian',@gaussian,'non_uniform',@nonUniform);
op.MUTATION_INT=struct('hop',@hop,'uniform',@uniformInt,'bit_flip',@bitFlip);
op.MUTATION_PERM=struct();
for name={'swap','inversion','scramble','displacement'}
    mode=name{1};
    op.MUTATION_INT.(mode)=@(r,v,lo,hi,k)rearrange(r,v,mode);
    op.MUTATION_PERM.(mode)=@(r,p)rearrange(r,p,mode);
end
end

function U=randomInit(r,space,n,grid,~)
U=space.random(r,n,grid);
end

function U=heuristicInit(r,space,n,grid,k)
U=space.random(r,n*k.heuristic_factor,grid);
end

function U=chaoticInit(r,space,n,grid,~)
s=2*rand(r,1,space.n_genes)-1;
bad=abs(s)<=1e-8 | abs(abs(s)-1)<=1e-8+1e-5;
while any(bad)
    s(bad)=2*rand(r,1,sum(bad))-1;
    bad=abs(s)<=1e-8 | abs(abs(s)-1)<=1e-8+1e-5;
end
U=zeros(n,space.n_genes);
for i=1:n
    U(i,:)=(s+1)/2;
    if grid, U(i,:)=space.snap(U(i,:)); end
    s=min(1,max(-1,4*s.^3-3*s));
end
end

function indices=tournament(r,f,n,~,k)
indices=zeros(1,n);
for i=1:n
    draw=randperm(r,numel(f),k.tournament_size);
    [~,j]=min(f(draw)); indices(i)=draw(j);
end
end

function indices=weighted(r,f,n,gen,k,method)
f=f(:).'; w=zeros(size(f)); finite=isfinite(f);
if any(f==-Inf)
    w(f==-Inf)=1;
elseif any(finite)
    values=f(finite);
    switch method
        case {'roulette','sus'}
            values=values/max(1,max(abs(values)));
            w(finite)=max(values)-values;
        case 'rank'
            [sorted,order]=sort(values); m=numel(values); ranked=zeros(1,m);
            i=1;
            while i<=m
                j=i;
                while j<m && sorted(j+1)==sorted(i), j=j+1; end
                ranked(order(i:j))=m+1-(i+j)/2; i=j+1;
            end
            w(finite)=ranked;
        case 'boltzmann'
            temperature=k.boltzmann_t*k.boltzmann_alpha^(gen-1);
            if temperature==0
                w(finite)=values==min(values);
            else
                w(finite)=exp(-(values-min(values))/temperature);
            end
    end
    if ~any(w), w(finite)=1; end
else
    w(:)=1;
end
cumulative=cumsum(w/sum(w)); cumulative(end)=1;
if strcmp(method,'sus'), draws=(rand(r)+(0:n-1))/n;
else, draws=rand(r,1,n); end
indices=zeros(1,n);
for i=1:n, indices(i)=find(draws(i)<cumulative,1); end
end

function [c,d]=singlePoint(r,a,b,~)
c=a; d=b; L=numel(a);
if L<2, return; end
cut=randi(r,L-1); c(cut+1:end)=b(cut+1:end); d(cut+1:end)=a(cut+1:end);
end

function [c,d]=kPoint(r,a,b,k)
c=a; d=b; L=numel(a);
if L<2, return; end
cuts=[0 sort(randperm(r,L-1,min(k.k_points,L-1))) L];
for i=2:2:numel(cuts)-1
    segment=cuts(i)+1:cuts(i+1); c(segment)=b(segment); d(segment)=a(segment);
end
end

function [c,d]=uniformCross(r,a,b,~)
mask=rand(r,size(a))<.5; c=b; d=a; c(mask)=a(mask); d(mask)=b(mask);
end

function [c,d]=shuffleCross(r,a,b,k)
permutation=randperm(r,numel(a));
[first,second]=singlePoint(r,a(permutation),b(permutation),k);
c=a; d=b; c(permutation)=first; d(permutation)=second;
end

function [c,d]=arithmetic(r,a,b,~)
alpha=rand(r); c=alpha*a+(1-alpha)*b; d=(1-alpha)*a+alpha*b;
end

function [c,d]=blx(r,a,b,k)
delta=abs(a-b); lo=min(a,b)-k.blx_alpha*delta; hi=max(a,b)+k.blx_alpha*delta;
u=rand(r,size(a)); c=clip((1-u).*lo+u.*hi);
u=rand(r,size(a)); d=clip((1-u).*lo+u.*hi);
end

function [c,d]=linear(r,a,b,~)
c=.5*(a+b);
if rand(r)<.5, d=1.5*a-.5*b; else, d=-.5*a+1.5*b; end
c=clip(c); d=clip(d);
end

function [c,d]=sbx(r,a,b,k)
u=rand(r,size(a)); beta=(2*u).^(1/(k.sbx_eta+1));
upper=u>.5; beta(upper)=(1./(2*(1-u(upper)))).^(1/(k.sbx_eta+1));
c=clip(.5*((1+beta).*a+(1-beta).*b));
d=clip(.5*((1-beta).*a+(1+beta).*b));
end

function [c,d]=pmx(r,a,b,~)
cuts=sort(randperm(r,numel(a)+1,2))-1;
c=pmxChild(a,b,cuts); d=pmxChild(b,a,cuts);
end

function child=pmxChild(a,b,cuts)
segment=cuts(1)+1:cuts(2); child=zeros(size(a)); child(segment)=a(segment);
for i=segment
    if ~any(a(segment)==b(i))
        pos=i;
        while pos>cuts(1) && pos<=cuts(2), pos=find(b==a(pos),1); end
        child(pos)=b(i);
    end
end
empty=child==0; child(empty)=b(empty);
end

function [c,d]=ox(r,a,b,~)
cuts=sort(randperm(r,numel(a)+1,2))-1;
c=oxChild(a,b,cuts); d=oxChild(b,a,cuts);
end

function child=oxChild(a,b,cuts)
L=numel(a); segment=cuts(1)+1:cuts(2); child=zeros(size(a)); child(segment)=a(segment);
positions=[cuts(2)+1:L 1:cuts(1)]; donor=b([cuts(2)+1:L 1:cuts(2)]);
child(positions)=donor(~ismember(donor,a(segment)));
end

function x=uniformReal(r,x,~,~,~)
j=randi(r,numel(x)); x(j)=rand(r);
end

function x=gaussian(r,x,~,~,k)
j=randi(r,numel(x)); x(j)=clip(x(j)+k.sigma*randn(r));
end

function x=nonUniform(r,x,gen,max_gen,~)
j=randi(r,numel(x)); sign=2*(rand(r)<.5)-1;
x(j)=clip(x(j)+sign*rand(r)*(1-gen/max_gen)^2);
end

function v=hop(r,v,lo,hi,k)
j=randi(r,numel(v)); v(j)=min(hi(j),max(lo(j),v(j)+k.hop_steps(randi(r,numel(k.hop_steps)))));
end

function v=uniformInt(r,v,lo,hi,~)
j=randi(r,numel(v)); v(j)=randi(r,[lo(j) hi(j)]);
end

function v=bitFlip(r,v,~,~,~)
j=randi(r,numel(v)); v(j)=1-v(j);
end

function v=rearrange(r,v,method)
pair=sort(randperm(r,numel(v),2)); segment=pair(1):pair(2);
switch method
    case 'swap', v(pair)=v(fliplr(pair));
    case 'inversion', v(segment)=v(fliplr(segment));
    case 'scramble', v(segment)=v(segment(randperm(r,numel(segment))));
    case 'displacement'
        block=v(segment); remainder=v; remainder(segment)=[];
        cut=randi(r,numel(remainder)+1)-1;
        v=[remainder(1:cut) block remainder(cut+1:end)];
end
end

function x=clip(x)
x=min(1,max(0,x));
end
