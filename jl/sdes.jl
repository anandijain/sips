module DataGen
using DifferentialEquations, Distributions, StatsBase, Plots, StatsPlots, DataFrames, CSV

function init_conditions(n::Int=20)
	normal = Normal(0, 200)
	u₀_arr = rand(normal, n)
	return u₀_arr
end	

function gen_data(u₀_arr::Array)
	p = plot()
	for u₀ in u₀_arr
		println(u₀)
	end
end

function basic_sde(u₀)
	α, β = randn(2)
	f(u, p, t) = α * u
	g(u, p, t) = β * u
	dt = 1//2 ^ (4)
	tspan = (0.0, 1.0)
	prob = SDEProblem(f, g, u₀, (0.0, 1.0))
	return prob
end

# function ensemble_sde(u₀s)


end
