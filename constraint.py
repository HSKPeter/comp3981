from assignment import Assignments
import copy


class Constraints:
    def __init__(self, assignments: Assignments):
        # key = (row_index, col_index, sub_square_index)
        # Note: sub_square_index starting from 0, counting from top-left to bottom-right of the board
        # value = a set that representing the domain
        self.domains = dict()
        for cell_key, cell_value in assignments.values.items():
            if cell_value == 0:
                self.domains[cell_key] = set(range(1, 10))
            else:
                self.domains[cell_key] = {cell_value}

        self.domains_copy = copy.deepcopy(self.domains)

    def add_inferences(self, inferences_to_add):
        """
        Args:
            inferences_to_add: a dict where the key is a tuple of (row_index, col_index, sub_square_index),
            and the value is a set of numbers that represents the domain values
        """
        self.domains.update(inferences_to_add)

    def copy(self):
        return copy.deepcopy(self)

    def remove_inferences(self):
        self.domains = {key: value for (
            key, value) in self.domains_copy.items()}
