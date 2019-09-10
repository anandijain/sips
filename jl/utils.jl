module Utils

using CSV, DataFrames, Plots, DiffEqFlux, DiffEqMonteCarlo

function odds_df(fn::String="../sips/gym_sip/data/static/nba2.csv")
	CSV.read(fn)
end

function games_from_odds(df::DataFrame, cols::Array{Symbol, 1}=[:a_team, :game_id])
	groupby(df, cols)
end

function plot_all(dfs)
	for df in dfs
		plot!(df.cur_time, df.a_odds_ml)
	end
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

function adj_all_games()
	df = odds_df()
	games = games_from_odds(df)
	adj_dfs = adjust_times(games)
	return adj_dfs
end

# function train_nsde(games)
end
