module Sips
	using TimeSeries, CSV, Plots

	function get_dfs(directory::String)
		dfs = []
		files = readdir(directory)
		for file in files
			fp = string(directory, file)
			df = CSV.read(fp)
			push!(dfs, df)
		end
		return dfs
	end

	function dfs_to_ts(dfs)
		time_series = []
		for df in dfs
			if length(names(df)) == 0
				continue
			end

			date_times = convert(Array, df.Date)
			ohlc_values = convert(Matrix, df[:, 2:5])
			column_names = names(df)[2:5]
			metadata = df[:, 6]
			time_array = TimeArray(date_times, ohlc_values, column_names, metadata)
			push!(time_series, time_array)
		end
		return time_series
	end

	function dates_to_int(dates)
		# epoch time (slow)		
		for d in dates:

	


	function lstm_tuples(df; window_len=10)
		tups = []
		for i in range(1, stop=size(df, 1) - window_len - 1)
			idx = i + window_len
			x = convert(Matrix, df[i:idx, :])
			y = convert(Array, df[idx + 1, :])
			tup = (x, y)
			push!(tups, tup)
		end
		return tups
	end
end
