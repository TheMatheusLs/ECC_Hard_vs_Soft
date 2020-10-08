from modules.aux_functions import *
import numpy as np
from tabulate import tabulate 
from math import floor

class LinearCode:
    def __init__(self, G: list, H: list = None) -> None:
        self.generate2parity(G)
        self.codewords = []
        self.find_all_codewords()
        self.find_diagram()
        self.find_dmin()
        
        # Validating the diagram
        # All syndromes must be unique
        if len(set(self.get_liders_syndromes().keys())) == 2 ** self.G.shape[0]:
            # Each line must have the same syndrome
            for num, coset in enumerate(self.diagram):
                unique_syndrome = set()
                for codeword in coset:
                    unique_syndrome.add(self.get_syndromes(codeword))
                if len(unique_syndrome) != 1:
                    print(f"The line {num} does not have all the same syndromes")
        else:
            print("Syndromes are not unique")
    
    
    def generate2parity(self, G: list) -> None:
        """Generates the parity matrix using the code generating matrix

        Args:
            G (list): Code generating matrix
        """
        self.G = np.array(G)

        # Separates the information bits from the identity matrix
        parity_bits = np.array([row[:-self.G.shape[0]] for row in self.G])
        parity_bits.T
        
        self.H = np.array([list(row_eye) + list(row_parity_T) for row_eye, row_parity_T in zip(np.eye(parity_bits.shape[1]), parity_bits.T)], dtype=int)
        
        
    def find_all_codewords(self) -> None:
        """Generates all words in the code.
        """
        for binary in gen_all_binary(self.H.shape[0], False):
            self.codewords.append(dot_mod(binary, self.G))
            

    def find_diagram(self) -> np.array:
        """Finds the standard array for the code

        Returns:
            np.array: Standard array 
        """
        # Find all possible leaders for cosets
        posible_liders = sorted(gen_all_binary(len(self.codewords[0]), False), key=sum)[1:]

        diagram =  []

        # Adds the code words on the first line of the diagram
        diagram.append([bit2str(vec) for vec in self.codewords])

        # Go through all possible leaders
        for leaders in posible_liders:
            row_aux = []

            for codeword in self.codewords:
                # Sums the leader with the code word
                new_word = bit2str(sum_mod(leaders, codeword))

                # Checks whether the result is already in the diagram
                if new_word in np.array(diagram).reshape(-1):
                    break

                row_aux.append(new_word)

            if row_aux != []:
                diagram.append(row_aux)

        self.diagram = np.array(diagram)
    

    def print_diagram(self) -> None:
        """Tabulate the diagram using the tabulate module
        """
        print(tabulate(self.diagram, tablefmt="fancy_grid"))
        

    def get_leaders(self) -> np.array:
        """Returns a list of cosets leaders

        Returns:
            np.array: Cosets leaders
        """
        return np.array([coset[0] for coset in self.diagram])
    
    def get_liders_syndromes(self) -> dict:
        """Creates a dictionary associating syndromes with side class leaders
 
        Returns:
            dict: Dictionary in which syndromes are the key
        """
        return_data = {}
        for leader in self.get_leaders():
            return_data[bit2str(dot_mod(str2bit(leader), self.H.T))] = leader 
        return return_data
    
    def get_syndromes(self, vec: str) -> str:
        """Calculates the syndrome for the 'vec' vector

        Args:
            vec (str): Vector to be calculated the syndrome

        Returns:
            str: Vector syndrome
        """
        return bit2str(dot_mod(str2bit(vec), self.H.T))
    

    def find_dmin(self) -> int:
        """Finds the minimum distance for the code

        Returns:
            int: Minimum distance
        """
        dmin = float('inf')
        for codeword_a in self.codewords[1:]:
            for codeword_b in self.codewords[1:]:
                if bit2str(codeword_a) != bit2str(codeword_b):
                    if dmin > hamming_distance(codeword_a, codeword_b):
                        dmin = hamming_distance(codeword_a, codeword_b)
        self.dmin = dmin


    def decoderBSC(self, received: str) -> str:
        """Decodes the received vector, if possible.

        Args:
            received (str): Channel received vector

        Returns:
            str: Vector after decoding
        """

        t = floor((self.dmin - 1)/2)

        received_syndrome = self.get_syndromes(received)

        error_pattern = self.get_liders_syndromes()[received_syndrome]

        hw = hamming_weight(error_pattern)

        # The code is able to correct all error patterns less than or equal to t, but it is not able to correct all error patterns of weight greater than or equal to t + 1
        if hw >= t + 1:
            return f"Unable to correct! Weight of the error pattern is {hw} and it is only possible to correct the smaller weight equal to {t}" 
        
        p_codeword = sum_mod(str2bit(received), str2bit(error_pattern))

        p_codeword_sindrome = (dot_mod(p_codeword, self.H.T))

        if p_codeword_sindrome.all() == 0:
            return bit2str(p_codeword)
        else:
            return f"The word is not part of the code dictionary"