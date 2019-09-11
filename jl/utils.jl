module Utils

using CSV, DataFrames, Plots, DiffEqFlux, DiffEqMonteCarlo, StatsBase

function odds_df(fn::String="../../sips/gym_sip/data/static/nba2.csv")
	CSV.read(fn)
end

function games_from_odds(df::DataFrame, cols::Array{Symbol, 1}=[:game_id])
	groupby(df, cols)
end

function plot_all(dfs; all::Bool=false, num::Int=5, scatter::Bool=false)
	p = plot()
	
	if all
		plot_dfs = dfs
	else
		plot_dfs = sample(dfs, num, replace=false)
	end

	for df in plot_dfs 
		if scatter
			scatter!(p, df.cur_time, df.a_odds_ml, markersize=2)
		else
			plot!(p, df.cur_time, df.a_odds_ml)
		end
	end
	return p
end

function plot_game(df)
	p = plot()
	plot!(p, df.cur_time, [df.a_odds_ml, df.h_odds_ml])
	return p
end

function adjust_times(dfs, time_col::Symbol=:cur_time)
	adjusted_dfs = []
	for df in dfs
		game = copy(df)
		ts = game.cur_time
		t0 = ts[1]
		ts .-= t0
		game[!, time_col] = ts
		push!(adjusted_dfs, game)
	end
	return adjusted_dfs
end 

function adj_all_games(;fn::String="../../sips/gym_sip/data/static/nba2.csv")
	df = odds_df(fn)
	games = games_from_odds(df)
	adj_dfs = adjust_times(games)
	return adj_dfs
end

function eq(odd::Number)
	if odd == 0
		return 0
	elseif odd > 0
		return odd/100
	else
		return abs(100/odd)
	end
end

# function train_nsde(games)
end
