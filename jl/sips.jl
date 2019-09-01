module Sips
	using TimeSeries, CSV, Plots

	function get_dfs
		dfs = []
		files = readdir("/home/sippycups/Downloads/datasets/price-volume-data-for-all-us-stocks-etfs/Data/Stocks")
		for file in files
			df = CSV.read(file)
			push!(dfs, df)
		end
		return dfs
	end
