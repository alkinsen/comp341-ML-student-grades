import csv

class CSVParser():

	def normalize(self, data, max):
		return int(round(float(data) / float(max) * 100))

	def parse(self, location):

		data_list = []
		target_list = []

		with open(location, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter = ';', quotechar = '"')
			next(reader, None)
			for row in reader:
				school = 0 # GP
				if row[0] == 'MS':
					school = 1
				sex = 1
				if row[1] == 'F':
					sex = 0

				age = self.normalize(int(row[2]) - 15, 7)

				address = 0 #Urban
				if row[3] == 'R':
					address = 1

				famsize = 0 #LE3
				if row[4] == 'GT3':
					famsize = 1

				pstatus = 0 #T - living together
				if row[5] == 'A':
					pstatus = 1

				medu = self.normalize(int(row[6]),4)
				fedu = self.normalize(int(row[7]),4)
				#mjob: ommitted
				#fjob: ommitted
				#reason: ommitted
				#guardian: ommitted
				traveltime = self.normalize(int(row[12])-1, 3)
				studytime = self.normalize(int(row[13])-1, 3)
				failures = self.normalize(int(row[14]),4)

				schoolsup = 0 #no
				if row[15] == 'yes':
					schoolsup = 1

				famsup = 0 #no
				if row[16] == 'yes':
					famsup = 1

				paid = 0 #no
				if row[17] == 'yes':
					paid = 1

				activities = 0 #no
				if row[18] == 'yes':
					activities = 1

				nursery = 0 #no
				if row[19] == 'yes':
					nursery = 1

				higher = 0 #no
				if row[20] == 'yes':
					higher = 1

				internet = 0 #no
				if row[21] == 'yes':
					internet = 1

				romantic = 0 #no
				if row[22] == 'yes':
					romantic = 1

				famrel = self.normalize(int(row[23])-1, 4)
				freetime = self.normalize(int(row[24])-1, 4)
				goout = self.normalize(int(row[25])-1, 4)
				dalc = self.normalize(int(row[26])-1, 4)
				walc = self.normalize(int(row[27])-1, 4)
				health = self.normalize(int(row[28])-1, 4)
				absences = self.normalize(int(row[29]), 93)

				g1 = self.normalize(int(row[30]), 20)
				g2 = self.normalize(int(row[31]), 20)
				f = self.normalize(int(row[32]), 20)
				avrg = int(round((g1 * 0.3 + g2 * 0.3 + f * 0.4)))

				data_list.append((school, sex, age, address, famsize, pstatus, medu, fedu, traveltime, studytime,failures*2,
				schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime*3, goout, dalc, walc*5,
				health, absences))

				if avrg >= 50:
					target_list.append(1)
				else:
					target_list.append(0)

		return (data_list, target_list)
