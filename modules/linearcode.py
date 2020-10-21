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
        # if len(set(self.get_liders_syndromes().keys())) == 2 ** self.G.shape[0]:
        #     # Each line must have the same syndrome
        #     for num, coset in enumerate(self.diagram):
        #         unique_syndrome = set()
        #         for codeword in coset:
        #             unique_syndrome.add(self.get_syndromes(codeword))
        #         if len(unique_syndrome) != 1:
        #             print(f"The line {num} does not have all the same syndromes")
        # else:
        #     print("Syndromes are not unique")
    
    
    def generate2parity(self, G: list) -> None:
        """Generates the parity matrix using the code generating matrix

        Args:
            G (list): Code generating matrix
        """
        self.G = np.array(G, dtype=int)

        # Separates the information bits from the identity matrix
        parity_bits = np.array([row[:-self.G.shape[0]] for row in self.G])
        parity_bits.T
        
        self.H = np.array([list(row_eye) + list(row_parity_T) for row_eye, row_parity_T in zip(np.eye(parity_bits.shape[1]), parity_bits.T)], dtype=int)
        
        
    def find_all_codewords(self) -> None:
        """Generates all words in the code.
        """
        for binary in gen_all_binary(self.G.shape[0], False):
            self.codewords.append(dot_mod(binary, self.G))
        self.codewords = np.array(self.codewords, dtype=int) 

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
            return_data[bit2str(dot_mod(str2bit(leader), self.H.T))] = str2bit(leader) 
        return return_data
    
    def get_syndromes(self, vec: list) -> list:
        """Calculates the syndrome for the 'vec' vector

        Args:
            vec (list): Vector to be calculated the syndrome

        Returns:
            list: Vector syndrome
        """
        return dot_mod(vec, self.H.T)
    

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


    def decoderBSC(self, vector_received: list) -> np.array:
        """Decodes the received vector.

        Args:
            received (str): Channel received vector

        Returns:
            str: Vector after decoding
        """
        #Convert polar to binary
        vector_received_binary = np.real(vector_received) > 0
        vector_received_binary = vector_received_binary.astype(int)

        # Received vector length
        n_symbols_sequence = len(vector_received_binary)

        # Length of information blocks
        n_symbols_block = self.G.shape[1]

        # Vector received separated in list of 'n' bits. 'n' is the information bits along with the parity bits
        vector_received_wordblock = np.reshape(np.real(vector_received_binary),(int(n_symbols_sequence/n_symbols_block), n_symbols_block))

        # Vector syndrome
        vector_received_syndromes = dot_mod(vector_received_wordblock, self.H.T) 

        # Vector error pattern 
        vector_error_pattern = np.array([self.get_liders_syndromes()[bit2str(syndrome)] for syndrome in vector_received_syndromes]) 

        # Decodes for codeword
        #vector_received_wordblock = np.array([sum_mod(codeword, str2bit(vector_error)) for codeword, vector_error  in zip(vector_received_wordblock, vector_error_pattern)])
        vector_received_word = np.array(sum_mod(vector_received_wordblock.reshape(-1), vector_error_pattern.reshape(-1)))
        vector_received_wordblock = np.reshape(vector_received_word,(int(n_symbols_sequence/n_symbols_block), n_symbols_block))

        # remove parity bits
        vector_received_harddecode_block = np.array([codeword_parity_bits[self.G.shape[1] - self.G.shape[0]:] for codeword_parity_bits in vector_received_wordblock])

        return vector_received_harddecode_block.reshape(-1)


    def decoderAWGN(self, vector_received: list) -> np.array:
        """Soft decodes the vector received by the channel

        Args:
            vector_received (list): Channel received vector

        Returns:
            np.array: Decoded Vector
        """
        # Received vector length
        n_symbols_sequence = len(vector_received)

        # Length of information blocks
        n_symbols_block = self.G.shape[1]

        # Vector received separated in list of 'n' bits. 'n' is the information bits along with the parity bits
        vector_received_wordblock = np.reshape(np.real(vector_received),(int(n_symbols_sequence/n_symbols_block), n_symbols_block))

        # Creates matrix A from code words
        A_matrix = 2 * self.codewords.T - 1

        # Creates matrix B a by multiplying the signal received by matrix B (Multiplication is in the reals field)
        B_matrix = vector_received_wordblock @ A_matrix

        # Create a list to store the column values in each row that has the highest value and also its value
        max_col_index_and_value = list()

        # Scrolls the lines and selects the highest value and its position on the line
        for B_row in B_matrix:
            max_col_index_and_value.append(max(enumerate(B_row), key=lambda index: index[1]))

        # Isolates information related to tuple indexes 'max_col_index_and value'
        col_index = np.array(max_col_index_and_value, dtype=int).T[0]

        # Function to convert a decimal number to binary. Obs: The binary value is spread to follow the way the code was created previously
        dec2bit = lambda dec: [int(bit) for bit in '{:0{}b}'.format(dec, self.G.shape[0])][::-1]

        # Concatenates the bits to generate the output signal
        vector_received_softdecode = np.concatenate(([dec2bit(index) for index in col_index]))
        
        return vector_received_softdecode


    def enconder(self, information_vector: list) -> np.array:
        """Encodes the information bits next to the parity bits

        Args:
            information_vector (list): Information bits

        Returns:
            np.array: information bits with parity bits
        """
        n_symbols_sequence = len(information_vector)

        # Length of information blocks
        n_symbols_block = self.G.shape[0]

        # Separates bits into blocks
        tx_signal_block = np.reshape(information_vector, (int(n_symbols_sequence/n_symbols_block), self.G.shape[0]))

        # Applies parity bits to each block according to the G matrix
        tx_signal_wordblock = dot_mod(tx_signal_block, self.G)

        # Place symbols to be sent side by side in a list
        tx_signal_with_paritybits = np.reshape(tx_signal_wordblock,(1,int((n_symbols_sequence/n_symbols_block) * self.G.shape[1])))[0]
        
        return tx_signal_with_paritybits

    
    def modulation_BPSK(self, bit_vector: np.array) -> np.array:
        """Convert bits to BPSK modulation

        Args:
            bit_vector (np.array): Information bits

        Returns:
            np.array: BPSK information bits
        """
        return (2 * bit_vector - 1)

    
    def add_RAGB(self, bit_vector: np.array, mu_AWGN: float, sigma_AWGN: float, EcN0dBs: float) -> np.array:
        """Add white Gaussian noise to a bit vector

        Args:
            bit_vector (np.array): Bit vector
            mu_AWGN (float): Mean AWGN
            sigma_AWGN (float): Sigma AWGN
            EcN0dBs (float): 

        Returns:
            np.array: Bit vector with noise
        """
        ## Creates the AWGN channel
        # German number based on a Gaussian distribution
        real_part_random = np.random.normal(mu_AWGN, sigma_AWGN, (1, len(bit_vector)))

        # German number based on a Gaussian distribution multiplied by j to create an imaginary number
        imaginary_part_random = 1j*np.random.normal(mu_AWGN, sigma_AWGN, (1, len(bit_vector)))

        # Channel with real and imaginary component
        channel = (1/np.sqrt(2))*(real_part_random + imaginary_part_random)

        ## Adds noise with a power
        rx_signal = bit_vector + (10**(-EcN0dBs/20)) * channel

        return rx_signal[0]