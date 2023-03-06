import sys
import json
import abc

class SampleGenerator(abc.ABC):
    def map_digits_to_int_list(self, string):
        return [int(char) if char.isnumeric() else char for char in string]

    def jsonfiy(self, src_filename):
        with open(f'{self.src_folder}/{src_filename}', 'r') as file:
            contents = file.read()
            rows = contents.split("\n")
            board = [self.map_digits_to_int_list(row) for row in rows]
            return json.dumps(board)   

    def generate(self, filename):
        jsonified_content = self.jsonfiy(filename)
        with open(f'{self.destination_folder}/{filename}', 'w') as file:
            file.write(jsonified_content)

    def generate_samples(self, filenames):
        confirmed = input("Are you running this .py program in the directory of \"utils/generate_samples\"? (Y/N) ")
        if confirmed.strip().upper() == "Y":
            for filename in filenames:
                self.generate(filename)
        else:
            print("You need to be in the directory of \"utils/generate_samples\" to run this .py program.")