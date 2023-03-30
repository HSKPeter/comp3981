class BoardSampleManager:

    _default_raw_samples_path = "assets/raw_samples"

    @classmethod
    def save_board(cls, file_path, filename, board):
        result = ""

        for row in board:
            row_str = ",".join(str(num) for num in row)
            result += row_str + "\n"

        cls.write_file(file_path, filename, str.strip(result))

    @classmethod
    def read_samples_file(cls, filename):
        with open(f"{cls._default_raw_samples_path}/{filename}", "r") as file:
            return file.read()

    @classmethod
    def write_file(cls, file_path, filename, content):
        with open(f"{file_path}/{filename}", "w") as file:
            file.write(content)

    @classmethod
    def save_easy_samples(cls):
        file_path = "assets/standard_samples/9x9/easy"
        easy_samples_text = cls.read_samples_file("list-easy-samples-9x9.txt")
        easy_samples = easy_samples_text.split("========")
        for i in range(len(easy_samples)):
            easy_sample = easy_samples[i]
            rows = [line for line in easy_sample.split("\n") if line != '']

            board_in_chars = [list(row) for row in rows]
            board = [[int(num) for num in row] for row in board_in_chars]

            cls.save_board(file_path, f"easy_sample_{str(i + 1).zfill(2)}.txt", board)

    @classmethod
    def save_hard_samples(cls):
        file_path = "assets/standard_samples/9x9/hard"
        hard_samples_text = cls.read_samples_file("list-hard-samples-9x9.txt")
        hard_samples = [line for line in hard_samples_text.split("\n") if line != '']
        for i in range(len(hard_samples)):
            hard_sample = hard_samples[i]
            hard_sample_with_empty_cell_as_zero = hard_sample.replace(".", "0")
            int_list = [int(num) for num in hard_sample_with_empty_cell_as_zero]
            board = [int_list[j:j + 9] for j in range(0, len(int_list), 9)]
            cls.save_board(file_path, f"hard_sample_{str(i + 1).zfill(2)}.txt", board)


def main():
    print("Please uncomment src code to save samples into local file storage")
    # BoardSampleSaver.save_easy_samples()
    # BoardSampleSaver.save_hard_samples()


if __name__ == "__main__":
    main()