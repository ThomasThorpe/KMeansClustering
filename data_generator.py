import csv
import random

class DataGenerator:
    def __init__(self,dimensions,num_rows,lower_bound,upper_bound,output_file):
        self.dimensions = dimensions
        self.num_rows = num_rows
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.output_file = output_file

    def write_data(self):
        with open(self.output_file,'w',newline='\n') as csvfile:
            writer = csv.writer(csvfile,delimiter=',')
            for row in range(self.num_rows):
                data_point = []
                for dimension in range(self.dimensions):
                    data_point.append(random.uniform(self.lower_bound,self.upper_bound))
                writer.writerow(data_point)

if __name__ == "__main__":
    dimensions = random.randint(2,5)
    num_rows = random.randint(10000,100000)
    lower_bound = -10
    upper_bound = 10
    output_file = "data.csv"
    g = DataGenerator(dimensions,num_rows,lower_bound,upper_bound,output_file)
    g.write_data()