from sample_generator import SampleGenerator
import sys

class InvalidSampleGenerator(SampleGenerator):
    def __init__(self):
        self.src_folder = "./valid_text_files"
        self.destination_folder = "../../assets/sample_files_for_upload"

def main():
    generator = InvalidSampleGenerator()
    filenames = sys.argv[1:]
    generator.generate_samples(filenames)

if __name__ == "__main__":
    main()