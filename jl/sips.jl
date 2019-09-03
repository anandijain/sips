module Sips
	using TimeSeries, CSV, Plots

	function get_dfs()
		dfs = []
		files = readdir("/home/sippycups/Downloads/datasets/price-volume-data-for-all-us-stocks-etfs/Data/Stocks")
		for file in files
			df = CSV.read(file)
			push!(dfs, df)
		end
		return dfs

	function dfs_to_ts(dfs)
		time_series = []
		for df in dfs:
			date_times = df.Dates
			ohlc_values = df[:, 2:5]
			column_names = names(df)[2:5]
			metadata = df[:, 6]
			time_array = TimeArray(date_times, ohlc_values, column_names, metadata)
			push!(time_series, time_array)
		end
		return time_series
	end
end
