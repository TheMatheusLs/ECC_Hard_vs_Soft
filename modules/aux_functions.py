import numpy as np

def gen_all_binary(num_bits: int, right: bool = True) -> list:
    """Create a list with numbers in binary that contains 'num_bits' elements
    

    Example:
        gen_all_binary(3) => [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        gen_all_binary(3, False) => [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    Args:
        num_bits (int): Number of bits for the binary
        right (bool, optional): Most significant bit on the right. Defaults to True.

    Returns:
        list: All binary numbers for 'num_bits' elements
    """
    bits = []
    for i in range(2**num_bits):
        bits.append([int(bit) for bit in '{:0{}b}'.format(i, num_bits)][::1 if right else -1])
    return bits


def dot_mod(matrix_A: list, matrix_B: list, mod: int =2) -> np.array:
    """Multiplies matrix A by matrix B in 'mod' module

    Args:
        matrix_A (list): Matrix A
        matrix_B (list): Matrix B
        mod (int, optional): Operation module. Defaults to 2.

    Raises:
        ValueError: The matrix format must respect the multiplication property

    Returns:
        np.array: New matrix with the result of multiplication in the form of numpy array
    """
    mA = np.array(matrix_A)
    mB = np.array(matrix_B)

    try:
        product = np.array( mA @ mB) # Tenta realizar a multiplicação

        product_mod = np.array([value  % mod for value in product.reshape(-1)])

        if product.ndim == 1:
            return product_mod
        else:
            return product_mod.reshape((product.shape[0], product.shape[1]))
        
    except:
        raise ValueError(f"The matrix format must respect the multiplication property. matrix_A = {mA.shape} e matrix_B = {mB.shape}")


def sum_mod(matrix_A: list, matrix_B: list, mod: int =2) -> np.array:
    """Add matrix A to matrix B in 'mod' module

    Args:
        matrix_A (list): Matrix A
        matrix_B (list): Matrix B
        mod (int, optional): Operation module. Defaults to 2.

    Raises:
        ValueError: The matrix format must respect the multiplication property

    Returns:
        np.array: New matrix with the sum result in numpy array format
    """
    if type(matrix_A) == str:
        mA = np.array(str2bit(matrix_A))
        mB = np.array(str2bit(matrix_B))
    else:
        mA = np.array(matrix_A)
        mB = np.array(matrix_B)

    if mA.shape != mB.shape:
        raise ValueError(f"The matrix format must respect the multiplication property. matrix_A = {mA.shape} e matrix_B = {mB.shape}")
        
    vec_sum = [(bit_mA + bit_mB) for bit_mA, bit_mB in zip(mA, mB)]
    return [1 if bit % mod else 0 for bit in vec_sum]


def bit2str(vec: list) -> str:
    """Converts a bit list to a string.

    Example:
        str2bit([1, 1, 0, 0, 1, 0]) => '110010'

    Args:
        vec (list): Bit list

    Returns:
        str: Bit string
    """
    return "".join([str(bit) for bit in vec])


def str2bit(vec: str) -> list:
    """Convert a bit string to a list
    
    Example:
        str2bit('110010') => [1, 1, 0, 0, 1, 0]

    Args:
        vec (str): Bit string

    Returns:
        list: Bit list
    """
    return [int(bit) for bit in vec]


def hamming_distance(vec_a: list, vec_b: list) -> int:
    """Calculates the Hamming distance 

    Example:
        hamming_distance('110010', '101110') => 3

    Args:
        vec_a (list): Vector A
        vec_b (list): Vector B

    Raises:
        ValueError: "The vectors must have the same dimension !!!"

    Returns:
        int: Hamming Distance
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("The vectors must have the same dimension !!!")

    distance = 0

    for bit_a, bit_b in zip(vec_a, vec_b):
        if bit_a != bit_b:
            distance += 1
    
    return distance


def hamming_weight(vec: list) -> int:
    """Calculates the Hamming weight for the vector

    Example:
        hamming_weight('100110') => 3
        hamming_weight('121110') => 5

    Args:
        vec (list): Bit Vector

    Returns:
        int: Hamming weight
    """
    return sum([1 if bit != '0' else 0 for bit in vec])