import csv

class CSVParser():
	def parse(self, location):

		data_list = []
		target_list = []

		with open(location, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter = ';', quotechar = '"')
			next(reader, None)
			for row in reader:
				failures = int(row[14])
				sex = 1
				if row[1] == 'F':
					sex = 0

				#higher = 1
				#if row[20] == 'no':
				#	higher = 0

				freetime = int(row[24])
				go_out = int(row[25])
				absences = int(row[29])

				dalc = int(row[26])
				walc = int(row[27])
				g1 = int(row[30])
				g2 = int(row[31])
				f = int(row[32])

				data_list.append((failures,sex,dalc,walc,freetime,go_out,absences,g2))
				target_list.append(f)

		return (data_list, target_list)
