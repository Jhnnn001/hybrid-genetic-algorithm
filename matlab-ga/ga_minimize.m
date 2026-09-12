function result=ga_minimize(objective,space,varargin)
%GA_MINIMIZE Construct a GA and return its run result; minimizes objective.
ga=GA(objective,space,varargin{:});
result=ga.run();
end
