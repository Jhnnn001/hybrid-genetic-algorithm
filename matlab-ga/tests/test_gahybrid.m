function tests = test_gahybrid
% Native MATLAB contracts and private GA acceptance objectives.
tests = functiontests(localfunctions);
end

function testSpaceDecode(t)
s = Space('n', Integer(0,40));
for pair = [0 1 .5 .0124 .0126; 0 40 20 0 1]
    p = s.decode(pair(1)); verifyEqual(t,p.n,pair(2));
end
s = Space('real',Real(-5,5),'bits',Binary(2),'order',Permutation(3));
p = s.decode([.75 .49 .5 .3 .1 .2]);
verifyEqual(t,p.real,2.5); verifyClass(t,p.real,'double');
verifyEqual(t,p.bits,logical([0 1]));
verifyEqual(t,p.order,[2 3 1]); verifyClass(t,p.order,'double');
verifyEqual(t,s.get_perm([.75 .49 .5 .2 .2 .1],4:6),[3 1 2]);
verifyEqual(t,s.set_perm(zeros(1,6),4:6,[3 1 2]),[0 0 0 .5 1 0]);
end

function testSpaceRoundtrip(t)
[~,s] = toyMixed(); r = RandStream('mt19937ar','Seed',0);
U = s.random(r,50,true);
for i=1:50
    u=U(i,:); verifyEqual(t,s.snap(u),u,'AbsTol',1e-15);
    verifyEqual(t,s.set_ints(u,s.get_ints(u)),u);
    for j=1:numel(s.perm_slices)
        b=s.perm_slices{j}; verifyEqual(t,s.set_perm(u,b,s.get_perm(u,b)),u);
    end
end
U=s.random(r,50,false);
for i=1:50
    verifyEqual(t,s.decode(s.snap(U(i,:))),s.decode(U(i,:)));
    verifyEqual(t,s.snap(s.snap(U(i,:))),s.snap(U(i,:)));
end
% Uniform grid draws give endpoints the same probability as middle values.
s=Space('n',Integer(0,2)); U=s.random(r,12000,true);
counts=arrayfun(@(v)sum(U==v),[0 .5 1]);
verifyLessThan(t,max(abs(counts-4000)),250);
end

function testExactIntegerLimits(t)
verifyError(t,@()Integer(0,flintmax),'gahybrid:InvalidInput');
verifyError(t,@()Integer(-flintmax,0),'gahybrid:InvalidInput');
verifyError(t,@()Integer(-flintmax+1,1),'gahybrid:InvalidInput');
verifyError(t,@()Integer(0,2^52),'gahybrid:InvalidInput');
s=Space('n',Integer(0,2^52-1));
for v=[0 1 2^51 2^52-2 2^52-1]
    verifyEqual(t,s.get_ints(s.set_ints(0,v)),v);
end
end

function testEmptyStringOperator(t)
s=Space('x',Real(0,1));
g=GA(@(p)p.x,s,'seed',0,'selection',"",'crossover_real',"",'max_gen',1);
result=g.run(); verifyEqual(t,result.n_evals,59);
end

function testRealCancellationBounds(t)
s=Space('x',Real(-1e16,3)); p=s.decode(1); verifyEqual(t,p.x,3);
s=Space('x',Real(-1e308,1)); p=s.decode(1); verifyEqual(t,p.x,1);
end

function testBinaryThresholdBeforeSnap(t)
s=Space('mask',Binary()); u=.5-eps(.5)/2;
p=s.decode(u); verifyFalse(t,p.mask);
verifyEqual(t,s.get_ints(u),0); verifyEqual(t,s.snap(u),0);
verifyEqual(t,s.decode(s.snap(u)),p);
end

function testBlxFiniteLargeAlpha(t)
op=ga_operators(); k=knobs(); k.blx_alpha=1e308;
r=RandStream('mt19937ar','Seed',1); copy=RandStream('mt19937ar','Seed',1);
expected1=double(rand(copy,1,8)>.5); expected2=double(rand(copy,1,8)>.5);
[a,b]=op.CROSSOVER_REAL.blx(r,zeros(1,8),ones(1,8),k);
verifyEqual(t,a,expected1); verifyEqual(t,b,expected2);
end

function testValidation(t)
[f,s]=toyMixed();
bad={@()Real(1,1),@()Real(false,1),@()Real(-realmax,realmax), ...
    @()Real(0,Inf),@()Integer(3,2),@()Integer(0,.5), ...
    @()Binary(0),@()Binary(true),@()Permutation(1),@()Space(), ...
    @()Space('a',3),@()Space('not valid',Real(0,1)), ...
    @()Space('a',Real(0,1),'a',Binary()), ...
    @()Space('a',struct('kind','real','low',0,'high',NaN,'size',1)), ...
    @()GA(f,s,'method','ga'),@()GA(f,s,'method','scipy'), ...
    @()GA(f,s,'pop_size',1),@()GA(f,s,'max_gen',0), ...
    @()GA(f,s,'elitism',30),@()GA(f,s,'tournament_size',31), ...
    @()GA(f,s,'crossover_prob',1.5),@()GA(f,s,'hop_steps',[]), ...
    @()GA(f,s,'hop_steps',[0 1]),@()GA(f,s,'seed',true), ...
    @()GA(f,s,'seed',2^32),@()GA(f,s,'sigma',NaN), ...
    @()GA(f,s,'pop_size',true),@()GA(f,s,'callback',3), ...
    @()GA(f,s,'method','relax','mutation_int','hop'), ...
    @()GA(f,s,'method','flat','crossover_int','uniform'), ...
    @()GA(f,s,'mutation_int','bit_flip'),@()GA(f,s,'mutation_int','swap'), ...
    @()GA(@(p)p.x,Space('x',Real(0,1)),'crossover_perm','ox'), ...
    @()GA(f,s,'nope',1),@()GA(f,s,'max',1)};
for i=1:numel(bad)
    verifyError(t,bad{i},'gahybrid:InvalidInput');
end
try
    GA(f,s,'nope',1);
catch exception
    verifyTrue(t,contains(exception.message,'nope'));
end
end

function testMethodsContracts(t)
[f,s]=toyMixed();
for method=["split","flat","relax"]
    calls=0; callbackGen=[]; callbackBest=[];
    g=GA(@checked,s,'method',method,'pop_size',10,'max_gen',20, ...
        'init','heuristic','seed',0,'callback',@callback);
    verifyEmpty(t,g.population); verifyEmpty(t,g.best_x);
    verifyEqual(t,g.n_evals,0); verifyTrue(t,isnan(g.diversity));
    res=g.run();
    verifyEqual(t,res.n_evals,calls); verifyEqual(t,calls,30+20*9);
    verifySize(t,res.history,[21 1]); verifySize(t,res.diversity,[21 1]);
    verifyTrue(t,all(res.history(2:end)<=res.history(1:end-1)));
    verifyEqual(t,res.history(end),res.fun); verifyEqual(t,res.fun,f(res.x));
    verifyTrue(t,all(res.diversity>=0 & res.diversity<=1));
    verifyEqual(t,res.n_gen,20); verifyEqual(t,res.method,method);
    verifyEqual(t,callbackGen,1:20);
    verifyTrue(t,all(callbackBest(2:end)<=callbackBest(1:end-1)));
    verifySize(t,g.fitness,[10 1]); verifySize(t,g.best_u,[1 s.n_genes]);
end
    function value=checked(p)
        calls=calls+1; checkMixed(t,p); value=f(p);
    end
    function callback(ga)
        callbackGen(end+1)=ga.gen; callbackBest(end+1)=ga.best_f;
        verifySize(t,ga.population,[10 s.n_genes]);
        verifyEqual(t,ga.n_evals,30+ga.gen*9);
    end
end

function testCountsAndReproducibility(t)
[f,s]=toyMixed();
for method=["split","flat","relax"]
    for init=["random","chaotic","heuristic"]
        calls=0; g=GA(@counted,s,'method',method,'pop_size',7, ...
            'max_gen',11,'elitism',2,'seed',0,'init',init);
        result=g.run(); expected=7+11*5+14*(init=="heuristic");
        verifyEqual(t,result.n_evals,expected); verifyEqual(t,calls,expected);
    end
    a=ga_minimize(f,s,'method',method,'seed',123,'max_gen',15);
    b=ga_minimize(f,s,'method',method,'seed',123,'max_gen',15);
    c=ga_minimize(f,s,'method',method,'seed',124,'max_gen',15);
    verifyEqual(t,a.history,b.history); verifyEqual(t,a.x,b.x);
    verifyNotEqual(t,a.history,c.history);
end
    function value=counted(p)
        calls=calls+1; value=f(p);
    end
end

function testAllOperators(t)
[f,s]=toyMixed(); op=ga_operators();
families={'INIT','SELECTION','CROSSOVER_REAL','CROSSOVER_INT','CROSSOVER_PERM', ...
    'MUTATION_REAL','MUTATION_INT','MUTATION_PERM'};
options={'init','selection','crossover_real','crossover_int','crossover_perm', ...
    'mutation_real','mutation_int','mutation_perm'};
for i=1:numel(families)
    names=fieldnames(op.(families{i}));
    for j=1:numel(names)
        ff=f; ss=s; methods="split";
        if any(i==[3 6]), methods=["split","flat","relax"]; end
        if i==7 && any(strcmp(names{j},{'bit_flip','swap','inversion','scramble','displacement'}))
            ss=Space('mask',Binary(12)); ff=@(p)sum(p.mask);
        end
        for method=methods
            g=GA(ff,ss,'method',method,'pop_size',8,'max_gen',5,'seed',0,options{i},names{j});
            g.run(); verifyTrue(t,all(g.population>=0 & g.population<=1,'all'));
            for row=1:8
                p=ss.decode(g.population(row,:));
                if isequal(ss.names,s.names), checkMixed(t,p); end
                if method~="relax", verifyEqual(t,ss.snap(g.population(row,:)),g.population(row,:)); end
            end
        end
    end
end
end

function testToyMixedConverges(t)
[f,s]=toyMixed();
for method=["split","flat","relax"]
    r=ga_minimize(f,s,'method',method,'pop_size',40,'max_gen',150,'elitism',2,'seed',0);
    if method=="relax"
        verifyLessThan(t,r.fun,1);
    else
        verifyEqual(t,r.x.n,7); verifyEqual(t,r.x.mask,logical([1 0 1 1 0 0]));
        verifyEqual(t,r.x.order,1:4); verifyLessThan(t,r.fun,.05);
    end
end
end

function testNotebookToy(t)
s=Space('r0',Real(0,1),'r1',Real(0,1),'r2',Real(0,1),'r3',Real(0,1),'mask',Binary(4));
r=ga_minimize(@notebookToy,s,'pop_size',20,'max_gen',500,'elitism',1, ...
    'seed',0,'init','chaotic','selection','tournament','tournament_size',3, ...
    'crossover_int','shuffle','crossover_real','blx','blx_alpha',.3, ...
    'crossover_prob',.8,'mutation_int','swap','mutation_real','gaussian','mutation_prob',.1);
verifyLessThanOrEqual(t,r.fun,-.59);
end

function testOperatorContracts(t)
op=ga_operators(); k=knobs(); a=[.1 .3 .6 .8]; b=[.8 .6 .3 .1];
for name={'single_point','k_point','uniform'}
    r=RandStream('mt19937ar','Seed',1); [c,d]=op.CROSSOVER_REAL.(name{1})(r,a,b,k);
    verifyTrue(t,all((c==a & d==b)|(c==b & d==a)));
    [c,d]=op.CROSSOVER_REAL.(name{1})(r,.2,.8,k);
    verifyEqual(t,sort([c d]),[.2 .8]);
end
r=RandStream('mt19937ar','Seed',4); r2=RandStream('mt19937ar','Seed',4);
alpha=rand(r2); [c,d]=op.CROSSOVER_REAL.arithmetic(r,a,b,k);
verifyEqual(t,c,alpha*a+(1-alpha)*b,'AbsTol',1e-15);
verifyEqual(t,c+d,a+b,'AbsTol',1e-15);
r=RandStream('mt19937ar','Seed',4); r2=RandStream('mt19937ar','Seed',4);
lo=min(a,b)-k.blx_alpha*abs(a-b); hi=max(a,b)+k.blx_alpha*abs(a-b);
expected=min(1,max(0,lo+rand(r2,1,4).*(hi-lo)));
[c,~]=op.CROSSOVER_REAL.blx(r,a,b,k); verifyEqual(t,c,expected,'AbsTol',1e-15);
r=RandStream('mt19937ar','Seed',4); r2=RandStream('mt19937ar','Seed',4);
u=rand(r2,1,4); beta=(2*u).^(1/(k.sbx_eta+1));
beta(u>.5)=(1./(2*(1-u(u>.5)))).^(1/(k.sbx_eta+1));
expected=min(1,max(0,.5*((1+beta).*a+(1-beta).*b)));
[c,~]=op.CROSSOVER_REAL.sbx(r,a,b,k); verifyEqual(t,c,expected,'AbsTol',1e-15);
verifyEqual(t,a,[.1 .3 .6 .8]); verifyEqual(t,b,[.8 .6 .3 .1]);
% Shared Python/MATLAB fixture: zero-based segment [2,5), converted to MATLAB.
seed=0;
while true
    r=RandStream('mt19937ar','Seed',seed);
    if isequal(sort(randperm(r,9,2))-1,[2 5]), break; end
    seed=seed+1;
end
a=1:8; b=[3 7 5 2 8 6 1 4];
[c,d]=op.CROSSOVER_PERM.pmx(RandStream('mt19937ar','Seed',seed),a,b,k);
verifyEqual(t,c,[8 7 3 4 5 6 1 2]); verifyEqual(t,d,[1 4 5 2 8 6 7 3]);
[c,d]=op.CROSSOVER_PERM.ox(RandStream('mt19937ar','Seed',seed),a,b,k);
verifyEqual(t,c,[2 8 3 4 5 6 1 7]); verifyEqual(t,d,[3 4 5 2 8 6 7 1]);
[c,d]=op.CROSSOVER_PERM.ox(RandStream('mt19937ar','Seed',seed),b,b,k);
verifyEqual(t,c,b); verifyEqual(t,d,b);
for name={'swap','inversion','scramble','displacement'}
    r=RandStream('mt19937ar','Seed',4);
    p=op.MUTATION_PERM.(name{1})(r,b); verifyEqual(t,sort(p),1:8);
    bits=[0 1 0 1 1 0];
    p=op.MUTATION_INT.(name{1})(r,bits,zeros(1,6),ones(1,6),k);
    verifyEqual(t,sort(p),sort(bits));
end
p=op.MUTATION_INT.bit_flip(r,bits,zeros(1,6),ones(1,6),k);
verifyEqual(t,sum(p~=bits),1);
verifyEqual(t,op.MUTATION_REAL.non_uniform(r,[.2 .4],5,5,k),[.2 .4]);
[c,d]=op.CROSSOVER_REAL.single_point(r,zeros(1,8),ones(1,8),k);
verifyTrue(t,any(c==0)&&any(c==1)); verifyEqual(t,c+d,ones(1,8));
end

function testExceptionalFitness(t)
op=ga_operators(); k=knobs();
for name={'roulette','sus','rank','boltzmann'}
    select=op.SELECTION.(name{1}); r=RandStream('mt19937ar','Seed',2);
    idx=select(r,[-Inf 1 -Inf Inf],100,1,k); verifyTrue(t,all(ismember(idx,[1 3])));
    idx=select(r,[2 Inf 2],100,1,k); verifyTrue(t,all(ismember(idx,[1 3])));
    idx=select(r,[-realmax realmax Inf],100,1,k); verifyTrue(t,all(idx<=2));
    idx=select(r,[Inf Inf],100,1,k); verifyTrue(t,all(ismember(idx,[1 2])));
end

k.boltzmann_alpha=.001;
idx=op.SELECTION.boltzmann(r,[2 1 1 Inf],100,10000,k);
verifyTrue(t,all(ismember(idx,[2 3])));
[f,s]=toyMixed();
for method=["split","flat","relax"]
    for value=[NaN Inf -Inf]
        g=GA(@(~)value,s,'method',method,'pop_size',8,'max_gen',3,'seed',0);
        result=g.run(); checkMixed(t,result.x);
        verifyEqual(t,result.fun,replaceNaN(value));
        verifyTrue(t,all(result.history(2:end)<=result.history(1:end-1)));
    end
    result=ga_minimize(@partial,s,'method',method,'pop_size',10,'max_gen',20,'seed',0);
    verifyTrue(t,isfinite(result.fun)); verifyLessThanOrEqual(t,result.x.x0,0);
end
    function value=partial(p)
        if p.x0>0, value=NaN; else, value=f(p); end
    end
end

function testSelectionWeights(t)
op=ga_operators(); k=knobs(); r=RandStream('mt19937ar','Seed',7);
indices=op.SELECTION.rank(r,[0 0 1 Inf],6000,1,k);
counts=arrayfun(@(v)sum(indices==v),1:4);
verifyLessThan(t,max(abs(counts-[2500 2500 1000 0])),150);
indices=op.SELECTION.sus(r,[0 1 3 Inf],6000,1,k);
verifyEqual(t,arrayfun(@(v)sum(indices==v),1:4),[3600 2400 0 0]);
k.tournament_size=3;
indices=op.SELECTION.tournament(r,[5 -1 3],30,1,k);
verifyEqual(t,indices,2*ones(1,30));
end

function testPermutationOperatorInvariants(t)
op=ga_operators(); k=knobs(); r=RandStream('mt19937ar','Seed',9);
for i=1:80
    a=randperm(r,8); b=randperm(r,8);
    for name={'pmx','ox'}
        [c,d]=op.CROSSOVER_PERM.(name{1})(r,a,b,k);
        verifyEqual(t,sort(c),1:8); verifyEqual(t,sort(d),1:8);
    end
end
end

function testStateIsolationAndErrors(t)
[f,s]=toyMixed(); before=rng;
stream=RandStream('mt19937ar','Seed',5);
g=GA(@mutating,s,'seed',stream,'pop_size',7,'max_gen',5,'elitism',0);
first=g.run(); state1=stream.State; second=g.run();
verifyEqual(t,rng,before); verifyNotEqual(t,stream.State,state1);
for method=["split","flat","relax"]
    ga_minimize(f,s,'method',method,'max_gen',1);
end
verifyEqual(t,rng,before);
verifyEqual(t,first.n_evals,42); verifyEqual(t,second.n_evals,42);
verifyEqual(t,second.fun,f(second.x)); verifyNotEqual(t,first.history,second.history);
checkMixed(t,second.x);
verifyError(t,@()ga_minimize(@(~)[1 2],s,'max_gen',1),'gahybrid:ObjectiveValue');
verifyError(t,@()ga_minimize(@(~)1i,s,'max_gen',1),'gahybrid:ObjectiveValue');
verifyError(t,@()ga_minimize(@(~)'text',s,'max_gen',1),'gahybrid:ObjectiveValue');
% With no elites, a lost first candidate must remain in the archive.
calls=0; g=GA(@once,s,'seed',0,'pop_size',7,'max_gen',3,'elitism',0);
result=g.run(); verifyEqual(t,result.fun,-1); verifyTrue(t,all(g.fitness==1));
again=g.run(); verifyEqual(t,again.fun,1); verifyEqual(t,again.n_evals,28);
    function value=mutating(p)
        value=f(p); p.mask(:)=false; p.order(:)=1; p.x0=Inf; %#ok<NASGU>
    end
    function value=once(~)
        calls=calls+1; value=1; if calls==1, value=-1; end
    end
end

function testNonidentityPermutationTarget(t)
s=Space('order',Permutation(4)); target=[3 1 4 2];
for method=["split","flat","relax"]
    r=ga_minimize(@(p)sum(abs(p.order-target)),s,'method',method, ...
        'seed',0,'pop_size',20,'max_gen',30,'mutation_prob',1);
    verifyEqual(t,r.x.order,target); verifyEqual(t,r.fun,0);
end
end

function checkMixed(t,p)
verifyEqual(t,fieldnames(p),{'x0';'x1';'n';'mask';'order'});
verifyClass(t,p.x0,'double'); verifySize(t,p.x0,[1 1]);
verifyClass(t,p.x1,'double'); verifySize(t,p.x1,[1 1]);
verifyTrue(t,p.x0>=-5&&p.x0<=5&&p.x1>=-5&&p.x1<=5);
verifyClass(t,p.n,'double'); verifyEqual(t,p.n,floor(p.n));
verifyTrue(t,p.n>=0&&p.n<=20); verifyClass(t,p.mask,'logical');
verifySize(t,p.mask,[1 6]); verifyClass(t,p.order,'double');
verifyEqual(t,sort(p.order),1:4);
end

function [f,s]=toyMixed()
s=Space('x0',Real(-5,5),'x1',Real(-5,5),'n',Integer(0,20), ...
    'mask',Binary(6),'order',Permutation(4));
f=@(p)(p.x0-1.5)^2+(p.x1-1.5)^2+(p.n-7)^2 ...
    +sum(abs(double(p.mask)-[1 0 1 1 0 0]))+sum(abs(p.order-(1:4)));
end

function value=notebookToy(p)
v=[p.r0 p.r1 p.r2 p.r3]; selected=v(p.mask);
if isempty(selected), value=0; return; end
m=mean(selected); q=numel(selected)/4;
value=-m*(1-4*(m-.5)^2)*(1-4*(q-.5)^2);
end

function value=replaceNaN(value)
if isnan(value), value=Inf; end
end

function k=knobs()
k=struct('k_points',2,'blx_alpha',.5,'sbx_eta',2,'sigma',.1, ...
    'hop_steps',[-2 -1 1 2],'tournament_size',3,'boltzmann_t',10, ...
    'boltzmann_alpha',.95,'heuristic_factor',3);
end
