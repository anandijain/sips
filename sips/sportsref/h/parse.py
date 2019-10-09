import bs4

def comments(page):
	comments = page.findAll(text=lambda text:isinstance(text, bs4.Comment))
	return comments


def get_table(page, table_id):  # given bs4 page and table id, finds table using bs4. returns tbody
	table = page.find('table', {'id': table_id})
	return table


def parse_table(table, split='th'):
	tbody = table.tbody
	rows = tbody.find_all('tr')
	data_rows = [[] for i in range(len(r))]
	for row in rows:
		row_class = row.get('class')
		if row_class == 'spacer':
			continue
		print(row.text)
		row_data = []
		things = row.find_all(split)
		for thing in things:
			row_data.append(thing)

			print(thing.text)
		data_rows.append(row_data)
	return data_rows


def write_table(table, fn, split='th'):
	try:
		tbody = table.tbody
	except AttributeError:
		return
	try:
		file = open('.' + m.data + fn + '.csv', 'w')
	except FileExistsError:
		print('skip')
		return

	thead = table.thead
	columns_row = thead.tr
	col_items = columns_row.find_all('th')
	for i, col in enumerate(col_items):

		file.write(col.text)

		if i == len(col_items) - 1:
			file.write('\n')
		else:
			file.write(',')

	rows = tbody.find_all('tr')
	for row in rows:
		row_class = row.get('class')
		if row_class is None:  # when the row class is none it is a data row
			row_data = row.find_all(split)
			for i, data_pt in enumerate(row_data):
				file.write(data_pt.text)

				if i == len(row_data) - 1:
					file.write('\n')
				else:
					file.write(',')

	print('{} written to {}'.format(fn, m.data))
	file.close()
