# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import font # Explicitly import font
import math
import traceback # Keep for general error handling

# --- Helper for Row Transposition (Unchanged) ---
def get_key_order(key):
    key_lower = key.lower()
    indexed_key_chars = []
    for i, char_val in enumerate(key_lower):
        indexed_key_chars.append((char_val, i))
    
    sorted_indexed_key_chars = sorted(indexed_key_chars, key=lambda x: (x[0], x[1]))
    
    order = [0] * len(key)
    for rank, (char_val, original_index) in enumerate(sorted_indexed_key_chars):
        order[original_index] = rank
    return order

# --- Helpers for Playfair (Unchanged) ---
def clean_playfair_key(key_input):
    if not key_input:
        return ""
    cleaned = ""
    seen_chars = set()
    key_processed_for_playfair = key_input.upper().replace("J", "I")
    for char_val in key_processed_for_playfair:
        if char_val.isalpha() and char_val not in seen_chars:
            cleaned += char_val
            seen_chars.add(char_val)
    return cleaned

def create_playfair_matrix(valid_cleaned_key):
    matrix_chars = list(valid_cleaned_key)
    seen_in_matrix = set(matrix_chars)
    alphabet_playfair = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    for char_val in alphabet_playfair:
        if char_val not in seen_in_matrix:
            matrix_chars.append(char_val)
    if len(matrix_chars) != 25:
        raise RuntimeError(f"Internal error: Playfair matrix size is incorrect ({len(matrix_chars)}).")
    
    playfair_grid = []
    for i in range(0, 25, 5):
        playfair_grid.append(matrix_chars[i:i+5])
    return playfair_grid

def find_char_position_playfair(matrix_5x5, target_char):
    for r_idx, row_list in enumerate(matrix_5x5):
        try:
            c_idx = row_list.index(target_char)
            return r_idx, c_idx
        except ValueError:
            continue
    raise ValueError(f"Character '{target_char}' not found in Playfair matrix.")

def prepare_playfair_text_pairs(raw_text):
    text_filtered_alpha = ''.join(filter(str.isalpha, raw_text)).upper().replace("J", "I")
    if not text_filtered_alpha:
        return []
    
    prepared_char_list = []
    i = 0
    len_filtered_text = len(text_filtered_alpha)
    while i < len_filtered_text:
        char1 = text_filtered_alpha[i]
        if i + 1 == len_filtered_text:
            char2_fill = 'X' if char1 != 'X' else 'Z'
            prepared_char_list.append(char1 + char2_fill)
            i += 1
        else:
            char2 = text_filtered_alpha[i+1]
            if char1 == char2:
                char2_fill_for_duplicate = 'X' if char1 != 'X' else 'Z'
                prepared_char_list.append(char1 + char2_fill_for_duplicate)
                i += 1
            else:
                prepared_char_list.append(char1 + char2)
                i += 2
    return prepared_char_list

# --- Classical Cipher Functions (Unchanged) ---
def caesar_encrypt(text, key_str):
    try:
        key = int(key_str)
    except ValueError:
        raise ValueError("Caesar key must be an integer.")
    result = ""
    A_ord = ord('A')
    a_ord = ord('a')
    for char_val in text:
        if 'a' <= char_val <= 'z':
            shifted_char_code = ((ord(char_val) - a_ord + key) % 26) + a_ord
            result += chr(shifted_char_code)
        elif 'A' <= char_val <= 'Z':
            shifted_char_code = ((ord(char_val) - A_ord + key) % 26) + A_ord
            result += chr(shifted_char_code)
        else:
            result += char_val
    return result

def caesar_decrypt(text, key_str):
    try:
        key = int(key_str)
        return caesar_encrypt(text, str(-key))
    except ValueError:
        raise ValueError("Caesar key must be an integer.")

# Vigenere Cipher
def vigenere_encrypt(text, key_input):
    if not key_input or not key_input.isalpha():
        raise ValueError("Vigenere key must contain only letters and cannot be empty.")
    key_upper = key_input.upper()
    result = ""
    key_idx = 0
    key_len = len(key_upper)
    A_ord = ord('A')
    a_ord = ord('a')
    for char_val in text:
        if 'a' <= char_val <= 'z':
            shift = ord(key_upper[key_idx % key_len]) - A_ord 
            shifted_char_code = ((ord(char_val) - a_ord + shift) % 26) + a_ord
            result += chr(shifted_char_code)
            key_idx += 1
        elif 'A' <= char_val <= 'Z':
            shift = ord(key_upper[key_idx % key_len]) - A_ord
            shifted_char_code = ((ord(char_val) - A_ord + shift) % 26) + A_ord
            result += chr(shifted_char_code)
            key_idx += 1
        else:
            result += char_val
    return result

def vigenere_decrypt(text, key_input):
    if not key_input or not key_input.isalpha():
        raise ValueError("Vigenere key must contain only letters and cannot be empty.")
    key_upper = key_input.upper()
    result = ""
    key_idx = 0
    key_len = len(key_upper)
    A_ord = ord('A')
    a_ord = ord('a')
    for char_val in text:
        if 'a' <= char_val <= 'z':
            shift = ord(key_upper[key_idx % key_len]) - A_ord
            shifted_char_code = ((ord(char_val) - a_ord - shift + 26) % 26) + a_ord
            result += chr(shifted_char_code)
            key_idx += 1
        elif 'A' <= char_val <= 'Z':
            shift = ord(key_upper[key_idx % key_len]) - A_ord
            shifted_char_code = ((ord(char_val) - A_ord - shift + 26) % 26) + A_ord
            result += chr(shifted_char_code)
            key_idx += 1
        else:
            result += char_val
    return result

def monoalphabetic_encrypt(text, key_input):
    alpha_lower = "abcdefghijklmnopqrstuvwxyz"
    alpha_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    key_input_lower = key_input.lower()

    if len(key_input_lower) != 26:
         raise ValueError(f"Monoalphabetic key must be 26 characters (current: {len(key_input_lower)}).")
    if not key_input_lower.isalpha():
        original_non_alpha = sorted(list(set(c for c in key_input if not c.isalpha())))
        raise ValueError(f"Monoalphabetic key must contain only letters. Invalid chars: {', '.join(original_non_alpha)}")
    if len(set(key_input_lower)) != 26:
        counts = {}
        for char_val in key_input_lower:
            counts[char_val] = counts.get(char_val, 0) + 1
        duplicates = sorted([c for c, count in counts.items() if count > 1])
        raise ValueError(f"Monoalphabetic key must contain unique letters. Duplicates: {', '.join(duplicates)}")

    encrypt_map = {alpha_lower[i]: key_input_lower[i] for i in range(26)}
    encrypt_map.update({alpha_upper[i]: key_input_lower[i].upper() for i in range(26)})
    
    encrypted_text = []
    for char_val in text:
        encrypted_text.append(encrypt_map.get(char_val, char_val))
    return "".join(encrypted_text)

def monoalphabetic_decrypt(cipher_text, key_input):
    alpha_lower = "abcdefghijklmnopqrstuvwxyz"
    alpha_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    key_input_lower = key_input.lower()

    if len(key_input_lower) != 26:
         raise ValueError(f"Monoalphabetic key must be 26 characters (current: {len(key_input_lower)}).")
    if not key_input_lower.isalpha():
        original_non_alpha = sorted(list(set(c for c in key_input if not c.isalpha())))
        raise ValueError(f"Monoalphabetic key must contain only letters. Invalid chars: {', '.join(original_non_alpha)}")
    if len(set(key_input_lower)) != 26:
        counts = {}
        for char_val in key_input_lower:
            counts[char_val] = counts.get(char_val, 0) + 1
        duplicates = sorted([c for c, count in counts.items() if count > 1])
        raise ValueError(f"Monoalphabetic key must contain unique letters. Duplicates: {', '.join(duplicates)}")

    decrypt_map = {key_input_lower[i]: alpha_lower[i] for i in range(26)}
    decrypt_map.update({key_input_lower[i].upper(): alpha_upper[i] for i in range(26)})

    decrypted_text = []
    for char_val in cipher_text:
        decrypted_text.append(decrypt_map.get(char_val, char_val))
    return "".join(decrypted_text)

def rail_fence_encrypt(text, rails_str):
    try:
        rails = int(rails_str)
        if rails < 2:
            raise ValueError() # Will be caught by the except block
    except (ValueError, AssertionError): # AssertionError for consistency, though int() raises ValueError
        raise ValueError("Rail Fence key (rails) must be an integer >= 2.")

    clean_text_no_spaces = text.replace(" ", "")
    if not clean_text_no_spaces or rails >= len(clean_text_no_spaces) or rails == 1:
        return clean_text_no_spaces

    fence_matrix = [''] * rails # Initialize with empty strings
    current_rail = 0
    direction = 1 

    for char_val in clean_text_no_spaces:
        fence_matrix[current_rail] += char_val
        if current_rail == 0:
            direction = 1
        elif current_rail == rails - 1:
            direction = -1
        current_rail += direction
    return ''.join(fence_matrix)

def rail_fence_decrypt(cipher_text, rails_str):
    try:
        rails = int(rails_str)
        if rails < 2:
            raise ValueError()
    except (ValueError, AssertionError):
        raise ValueError("Rail Fence key (rails) must be an integer >= 2.")

    cipher_len = len(cipher_text)
    if not cipher_text or rails >= cipher_len or rails == 1:
        return cipher_text

    rail_pattern_indices = []
    current_rail = 0
    direction = 1
    for _ in range(cipher_len):
        rail_pattern_indices.append(current_rail)
        if current_rail == 0:
            direction = 1
        elif current_rail == rails - 1:
            direction = -1
        current_rail += direction
    
    chars_per_rail = [0] * rails
    for rail_idx_in_pattern in rail_pattern_indices:
        chars_per_rail[rail_idx_in_pattern] +=1

    cipher_idx = 0
    rail_strings_list = []
    for count in chars_per_rail:
        rail_strings_list.append(list(cipher_text[cipher_idx : cipher_idx + count]))
        cipher_idx += count
    
    decrypted_text_list = []
    current_indices_in_rail_strings = [0] * rails

    for rail_idx_in_pattern in rail_pattern_indices:
        char_to_append = rail_strings_list[rail_idx_in_pattern][current_indices_in_rail_strings[rail_idx_in_pattern]]
        decrypted_text_list.append(char_to_append)
        current_indices_in_rail_strings[rail_idx_in_pattern] += 1

    return ''.join(decrypted_text_list)

def row_transposition_encrypt(text, key_input):
    if not key_input:
        raise ValueError("Row Transposition key cannot be empty.")
    if not text:
        return ""
    try:
        column_order_ranks = get_key_order(key_input)#order columns
    except Exception as e:
        raise ValueError(f"Error processing Row Transposition key: {e}")

    text_no_spaces = text.replace(" ", "")
    num_cols = len(key_input)
    num_rows = math.ceil(len(text_no_spaces) / num_cols)
    padding_char = '*' # Using a common padding character
    num_padding_chars = (num_rows * num_cols) - len(text_no_spaces)
    text_padded = text_no_spaces + padding_char * num_padding_chars

    matrix = [['' for _ in range(num_cols)] for _ in range(num_rows)]
    char_idx = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if char_idx < len(text_padded):
                matrix[r][c] = text_padded[char_idx]
                char_idx += 1
    
    cipher_text_list = []
    for rank_to_read in range(num_cols):
        try:
            actual_col_idx = column_order_ranks.index(rank_to_read)
        except ValueError:
            raise RuntimeError(f"Internal error: Rank {rank_to_read} not found in column order.")
        for r in range(num_rows):
            cipher_text_list.append(matrix[r][actual_col_idx])
    return "".join(cipher_text_list)

def row_transposition_decrypt(cipher_text_input, key_input):
    if not key_input:
        raise ValueError("Row Transposition key cannot be empty.")
    if not cipher_text_input:
        return ""
    try:
        column_order_ranks = get_key_order(key_input)
    except Exception as e:
        raise ValueError(f"Error processing Row Transposition key: {e}")

    cipher_text_no_spaces = cipher_text_input.replace(" ", "")
    num_cols = len(key_input)
    num_rows = math.ceil(len(cipher_text_no_spaces) / num_cols)
    
    matrix = [['' for _ in range(num_cols)] for _ in range(num_rows)]
    
    ranked_cols = [0] * num_cols # To store original column index for each rank
    for original_idx, rank_val in enumerate(column_order_ranks):
        ranked_cols[rank_val] = original_idx
        
    char_idx = 0
    for rank_to_fill in range(num_cols): # Iterate through ranks 0, 1, 2...
        actual_col_to_fill = ranked_cols[rank_to_fill]
        for r in range(num_rows):
            if char_idx < len(cipher_text_no_spaces):
                matrix[r][actual_col_to_fill] = cipher_text_no_spaces[char_idx]
                char_idx += 1
            else:
                matrix[r][actual_col_to_fill] = '' # Should ideally not happen if padding was correct
                
    plain_text_list = []
    for r in range(num_rows):
        for c in range(num_cols):
            plain_text_list.append(matrix[r][c])
            
    return "".join(plain_text_list).rstrip('*')

def playfair_encrypt(plaintext_input, key_input):
    if not key_input:
        raise ValueError("Playfair key cannot be empty.")
    cleaned_valid_key = clean_playfair_key(key_input)
    if not cleaned_valid_key:
        raise ValueError("Playfair key must contain at least one letter (A-Z) after cleaning.")
    try:
        playfair_matrix = create_playfair_matrix(cleaned_valid_key)
        prepared_pairs = prepare_playfair_text_pairs(plaintext_input)
        if not prepared_pairs:
            return ""
        
        cipher_text_parts = []
        for char1, char2 in prepared_pairs:
            r1, c1 = find_char_position_playfair(playfair_matrix, char1)
            r2, c2 = find_char_position_playfair(playfair_matrix, char2)
            
            if r1 == r2: # Same row
                cipher_text_parts.append(playfair_matrix[r1][(c1 + 1) % 5])
                cipher_text_parts.append(playfair_matrix[r2][(c2 + 1) % 5])
            elif c1 == c2: # Same column
                cipher_text_parts.append(playfair_matrix[(r1 + 1) % 5][c1])
                cipher_text_parts.append(playfair_matrix[(r2 + 1) % 5][c2])
            else: # Rectangle
                cipher_text_parts.append(playfair_matrix[r1][c2])
                cipher_text_parts.append(playfair_matrix[r2][c1])
        return "".join(cipher_text_parts)
    except ValueError as ve:
        raise ValueError(f"Playfair Encryption Error: {ve}")
    except Exception as e:
        traceback.print_exc() # For unexpected errors
        raise RuntimeError(f"Unexpected Playfair encryption error: {e}")

def playfair_decrypt(ciphertext_input, key_input):
    if not key_input:
        raise ValueError("Playfair key cannot be empty.")
    cleaned_valid_key = clean_playfair_key(key_input)
    if not cleaned_valid_key:
        raise ValueError("Playfair key must contain at least one letter (A-Z) after cleaning.")

    ciphertext_cleaned_alpha = ''.join(filter(str.isalpha, ciphertext_input)).upper().replace("J", "I")
    if not ciphertext_cleaned_alpha:
        return ""
    if len(ciphertext_cleaned_alpha) % 2 != 0:
        raise ValueError("Invalid Playfair ciphertext: Length must be even after cleaning non-alphabetic characters.")
    
    try:
        playfair_matrix = create_playfair_matrix(cleaned_valid_key)
        cipher_pairs = [ciphertext_cleaned_alpha[i:i+2] for i in range(0, len(ciphertext_cleaned_alpha), 2)]
        
        plaintext_parts = []
        for char1, char2 in cipher_pairs:
            r1, c1 = find_char_position_playfair(playfair_matrix, char1)
            r2, c2 = find_char_position_playfair(playfair_matrix, char2)

            if r1 == r2: # Same row
                plaintext_parts.append(playfair_matrix[r1][(c1 - 1 + 5) % 5]) # +5 for true modulo
                plaintext_parts.append(playfair_matrix[r2][(c2 - 1 + 5) % 5])
            elif c1 == c2: # Same column
                plaintext_parts.append(playfair_matrix[(r1 - 1 + 5) % 5][c1])
                plaintext_parts.append(playfair_matrix[(r2 - 1 + 5) % 5][c2])
            else: # Rectangle
                plaintext_parts.append(playfair_matrix[r1][c2])
                plaintext_parts.append(playfair_matrix[r2][c1])
        return "".join(plaintext_parts)
    except ValueError as ve:
        raise ValueError(f"Playfair Decryption Error: {ve}")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Unexpected Playfair decryption error: {e}")

# --- Simplified DES (S-DES) ---
# Permutation and shifting functions
def permute(bits_str, pattern_list): # Renamed for clarity
    """Permutes bits according to the pattern."""
    # pattern_list items are indices (0-based) into bits_str
    return ''.join(bits_str[i] for i in pattern_list)

def left_shift(bits_str, n_shifts): # Renamed for clarity
    """Performs a circular left shift on bits."""
    return bits_str[n_shifts:] + bits_str[:n_shifts]

def xor(bits1_str, bits2_str): # Renamed for clarity
    """Performs XOR operation on two bit strings."""
    return ''.join('0' if b1 == b2 else '1' for b1, b2 in zip(bits1_str, bits2_str))

# S-Boxes
S0_TABLE = [ # Renamed for clarity
    ['01', '00', '11', '10'],
    ['11', '10', '01', '00'],
    ['00', '10', '01', '11'],
    ['11', '01', '11', '10'] # Note: Original S-DES has '11', '01', '00', '10'
]                           # I will use your provided S0 table.

S1_TABLE = [ # Renamed for clarity
    ['00', '01', '10', '11'],
    ['10', '00', '01', '11'],
    ['11', '00', '01', '00'], # Note: Original S-DES has '11', '10', '01', '00'
    ['10', '01', '00', '11'] # I will use your provided S1 table.
]

def sbox_lookup(four_bit_input, sbox_table): # Renamed for clarity
    """Looks up value in S-Box table."""
    # Input is a 4-bit string
    row = int(four_bit_input[0] + four_bit_input[3], 2) # Bit 1 and Bit 4 for row
    col = int(four_bit_input[1] + four_bit_input[2], 2) # Bit 2 and Bit 3 for column
    return sbox_table[row][col]

# S-DES Fk function (round function)
def sdes_fk_function(eight_bit_input, round_key_8bit): # Renamed for clarity
    """S-DES Fk function (f_K)."""
    # Expansion/Permutation table for right 4 bits
    EP_TABLE = [3, 0, 1, 2, 1, 2, 3, 0] # Input bit indices (0-3 for right half)
    # Permutation P4 after S-Box lookup
    P4_TABLE = [1, 3, 2, 0]            # Input bit indices (0-3 for combined S-Box output)

    left_half, right_half = eight_bit_input[:4], eight_bit_input[4:]
    
    expanded_right = permute(right_half, EP_TABLE)
    xor_with_key = xor(expanded_right, round_key_8bit)
    
    s0_input = xor_with_key[:4]
    s1_input = xor_with_key[4:]
    
    s0_output = sbox_lookup(s0_input, S0_TABLE)
    s1_output = sbox_lookup(s1_input, S1_TABLE)
    
    combined_sbox_output = s0_output + s1_output
    p4_output = permute(combined_sbox_output, P4_TABLE)
    
    # XOR P4 output with the original left half
    new_left_half = xor(left_half, p4_output)
    
    # Right half remains unchanged in this part of Fk
    return new_left_half + right_half

# S-DES Key Generation
def sdes_generate_keys(key_10bit_str):
    """Generates K1 and K2 from the 10-bit key."""
    # P10: Permutation for 10-bit key
    P10_TABLE = [2, 4, 1, 6, 3, 9, 0, 8, 7, 5]
    # P8: Permutation to select 8 bits for round keys
    P8_TABLE = [5, 2, 6, 3, 7, 4, 9, 8]

    permuted_key_10 = permute(key_10bit_str, P10_TABLE)
    
    left_5_bits = permuted_key_10[:5]
    right_5_bits = permuted_key_10[5:]
    
    # Generate K1
    ls1_left = left_shift(left_5_bits, 1)
    ls1_right = left_shift(right_5_bits, 1)
    k1 = permute(ls1_left + ls1_right, P8_TABLE)
    
    # Generate K2 (shift the LS1 results by 2 more bits)
    ls2_left = left_shift(ls1_left, 2) # Shift the already shifted part
    ls2_right = left_shift(ls1_right, 2) # Shift the already shifted part
    k2 = permute(ls2_left + ls2_right, P8_TABLE)
    
    return k1, k2

# S-DES Encryption
def sdes_encrypt_function(plaintext_8bit_str, key_10bit_str): # Renamed
    """Encrypts 8-bit plaintext using S-DES with a 10-bit key."""
    # Initial Permutation
    IP_TABLE = [1, 5, 2, 0, 3, 7, 4, 6]
    # Inverse Initial Permutation (Final Permutation)
    IP_INV_TABLE = [3, 0, 2, 4, 6, 1, 7, 5]

    # Validate inputs
    if not all(c in '01' for c in plaintext_8bit_str) or len(plaintext_8bit_str) != 8:
        raise ValueError("S-DES plaintext must be an 8-bit binary string.")
    
    if not  (c in '01' for c in key_10bit_str) or len(key_10bit_str) != 10:
        raise ValueError("S-DES key must be a 10-bit binary string.")

    k1, k2 = sdes_generate_keys(key_10bit_str)
    
    permuted_plaintext = permute(plaintext_8bit_str, IP_TABLE)
    
    # Round 1
    round1_output = sdes_fk_function(permuted_plaintext, k1)
    
    # Swap halves after round 1
    swapped_halves = round1_output[4:] + round1_output[:4]
    
    # Round 2
    round2_output = sdes_fk_function(swapped_halves, k2) # Use K2 for round 2
    
    ciphertext = permute(round2_output, IP_INV_TABLE)
    return ciphertext

# S-DES Decryption
def sdes_decrypt_function(ciphertext_8bit_str, key_10bit_str): # Renamed
    """Decrypts 8-bit ciphertext using S-DES with a 10-bit key."""
    # Initial Permutation
    IP_TABLE = [1, 5, 2, 0, 3, 7, 4, 6]
    # Inverse Initial Permutation (Final Permutation)
    IP_INV_TABLE = [3, 0, 2, 4, 6, 1, 7, 5]

    # Validate inputs
    if not all(c in '01' for c in ciphertext_8bit_str) or len(ciphertext_8bit_str) != 8:
        raise ValueError("S-DES ciphertext must be an 8-bit binary string.")
    
    if not all(c in '01' for c in key_10bit_str) or len(key_10bit_str) != 10:
        raise ValueError("S-DES key must be a 10-bit binary string.")

    # Key generation is the same, but keys are used in reverse order for decryption
    k1, k2 = sdes_generate_keys(key_10bit_str)
    
    permuted_ciphertext = permute(ciphertext_8bit_str, IP_TABLE)
    
    # Round 1 of decryption (uses K2)
    round1_dec_output = sdes_fk_function(permuted_ciphertext, k2)
    
    # Swap halves after round 1 of decryption
    swapped_halves_dec = round1_dec_output[4:] + round1_dec_output[:4]
    
    # Round 2 of decryption (uses K1)
    round2_dec_output = sdes_fk_function(swapped_halves_dec, k1)
    
    plaintext = permute(round2_dec_output, IP_INV_TABLE)
    return plaintext

# ==============================================================
# ===                  Tkinter GUI Application               ===
# ==============================================================
class CryptoApp:
    def __init__(self, master):
        self.master = master
        master.title("Cryptography Tool")
        master.geometry("850x850") # Adjusted geometry
        self.BG_COLOR="#F5F5F5"
        self.FRAME_COLOR="#F5F5F5"
        self.TEXT_BG_COLOR="#FFFFFF"
        self.TEXT_FG_COLOR="#333333"
        self.BORDER_COLOR="#BDBDBD"
        self.BUTTON_GREEN="#4CAF50"
        self.BUTTON_GREEN_ACTIVE="#45a049"
        self.BUTTON_FG_COLOR="#FFFFFF"
        
        try:
            self.title_font = font.Font(family="Segoe UI", size=16, weight="bold")
            self.label_font = font.Font(family="Segoe UI Semibold", size=11)
            self.widget_font = font.Font(family="Segoe UI", size=10)
            self.output_font = font.Font(family="Consolas", size=11)
        except tk.TclError: # Fallback fonts if Segoe UI is not available
            self.title_font = font.Font(family="Arial", size=16, weight="bold")
            self.label_font = font.Font(family="Arial", size=11, weight="bold")
            self.widget_font = font.Font(family="Arial", size=10)
            self.output_font = font.Font(family="Courier New", size=11)
            
        master.configure(bg=self.BG_COLOR)
        self.setup_styles()

        self.algorithms = {
            "Caesar": {"encrypt": caesar_encrypt, "decrypt": caesar_decrypt, "key_type": "number"},
            "Vigenere": {"encrypt": vigenere_encrypt, "decrypt": vigenere_decrypt, "key_type": "text_alpha_only"},
            "Monoalphabetic": {"encrypt": monoalphabetic_encrypt, "decrypt": monoalphabetic_decrypt, "key_type": "mono_key"},
            "Rail Fence": {"encrypt": rail_fence_encrypt, "decrypt": rail_fence_decrypt, "key_type": "number_ge_2"},
            "Row Transposition": {"encrypt": row_transposition_encrypt, "decrypt": row_transposition_decrypt, "key_type": "text_any"},
            "Playfair": {"encrypt": playfair_encrypt, "decrypt": playfair_decrypt, "key_type": "text_playfair"},
            "S-DES": { # Added S-DES
                "encrypt": sdes_encrypt_function,
                "decrypt": sdes_decrypt_function,
                "key_type": "sdes_key_10bit", # New key type for S-DES
                "input_type": "binary_8bit",  # S-DES expects 8-bit binary input
                "output_type": "binary_8bit" # S-DES produces 8-bit binary output
            }
        }
        self.algo_names = sorted(list(self.algorithms.keys()))
        self.setup_ui()

    def setup_styles(self):
        self.style = ttk.Style()
        available_themes = self.style.theme_names()
        preferred_themes = ['clam', 'alt', 'default'] # Added more theme fallbacks
        for theme in preferred_themes:
            if theme in available_themes:
                try:
                    self.style.theme_use(theme)
                    break
                except tk.TclError:
                    continue
        
        self.style.configure('.', background=self.BG_COLOR, foreground=self.TEXT_FG_COLOR, font=self.widget_font)
        self.style.configure('TFrame', background=self.FRAME_COLOR)
        self.style.configure('TLabel', background=self.FRAME_COLOR, foreground=self.TEXT_FG_COLOR, font=self.label_font)
        self.style.configure('TCheckbutton', background=self.FRAME_COLOR, foreground=self.TEXT_FG_COLOR, font=self.widget_font)
        self.style.configure('TCombobox', font=self.widget_font, fieldbackground=self.TEXT_BG_COLOR, background=self.TEXT_BG_COLOR, foreground=self.TEXT_FG_COLOR, arrowcolor=self.TEXT_FG_COLOR, selectbackground=self.TEXT_BG_COLOR, selectforeground=self.TEXT_FG_COLOR)
        
        root = self.master
        root.option_add('*TCombobox*Listbox*Background', self.TEXT_BG_COLOR)
        root.option_add('*TCombobox*Listbox*Foreground', self.TEXT_FG_COLOR)
        root.option_add('*TCombobox*Listbox*Font', self.widget_font)
        root.option_add('*TCombobox*Listbox*selectBackground', self.BUTTON_GREEN)
        root.option_add('*TCombobox*Listbox*selectForeground', self.BUTTON_FG_COLOR)
        
        self.style.configure('TEntry', font=self.widget_font, fieldbackground=self.TEXT_BG_COLOR, foreground=self.TEXT_FG_COLOR, insertcolor=self.TEXT_FG_COLOR)
        self.style.configure('TLabelframe', background=self.FRAME_COLOR, bordercolor=self.BORDER_COLOR, borderwidth=1)
        self.style.configure('TLabelframe.Label', background=self.FRAME_COLOR, foreground=self.TEXT_FG_COLOR, font=self.label_font)
        self.style.configure("Apply.TButton", font=self.widget_font, background=self.BUTTON_GREEN, foreground=self.BUTTON_FG_COLOR, padding=(12, 6), borderwidth=0)
        self.style.map("Apply.TButton", background=[('active', self.BUTTON_GREEN_ACTIVE)], relief=[('pressed', 'sunken'), ('!pressed', 'raised')])

    def setup_ui(self):
        main_frame = ttk.Frame(self.master, padding="25 15 25 15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_text_ui = "Classical Cipher Tool & S-DES" # Updated title
        title_label = ttk.Label(main_frame, text=title_text_ui, font=self.title_font, anchor=tk.CENTER)
        title_label.pack(pady=(0, 20))
        
        input_frame = ttk.LabelFrame(main_frame, text=" Input ", padding="15 10")
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.input_text_label = ttk.Label(input_frame, text="Enter text (or 8-bit binary for S-DES):")
        self.input_text_label.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=(0,2))
        
        self.input_text = tk.Text(input_frame, height=6, width=80, font=self.widget_font, wrap=tk.WORD,
                                  relief=tk.SOLID, borderwidth=1, bd=1, bg=self.TEXT_BG_COLOR, fg=self.TEXT_FG_COLOR,
                                  highlightthickness=1, highlightbackground=self.BORDER_COLOR,
                                  highlightcolor=self.BUTTON_GREEN, insertbackground=self.TEXT_FG_COLOR)
        input_scroll = ttk.Scrollbar(input_frame, orient=tk.VERTICAL, command=self.input_text.yview)
        input_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,5), pady=(5,5))
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,0), pady=5)
        self.input_text['yscrollcommand'] = input_scroll.set
        
        control_frame = ttk.LabelFrame(main_frame, text=" Controls ", padding="15 10")
        control_frame.pack(fill=tk.X, pady=(0, 20))
        control_frame.columnconfigure(1, weight=1) # Make combobox/entry expand

        ttk.Label(control_frame, text="Algorithm 1:").grid(row=0, column=0, sticky=tk.W, padx=(5,10), pady=5)
        self.algo1_var = tk.StringVar()
        self.algo1_combo = ttk.Combobox(control_frame, textvariable=self.algo1_var, values=self.algo_names,
                                        state="readonly", width=25)
        self.algo1_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        self.algo1_combo.bind("<<ComboboxSelected>>", self.update_controls_for_algo1)
        if self.algo_names:
            self.algo1_combo.current(0)

        ttk.Label(control_frame, text="Operation 1:").grid(row=1, column=0, sticky=tk.W, padx=(5,10), pady=5)
        self.op1_var = tk.StringVar(value="Encrypt")
        self.op1_combo = ttk.Combobox(control_frame, textvariable=self.op1_var, values=["Encrypt", "Decrypt"],
                                      state="readonly", width=15)
        self.op1_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.key1_label = ttk.Label(control_frame, text="Key 1:")
        self.key1_label.grid(row=2, column=0, sticky=tk.W, padx=(5,10), pady=5)
        self.key1_entry = ttk.Entry(control_frame, width=30)
        self.key1_entry.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=3, column=0, columnspan=3, sticky='ew', pady=15)
        
        self.use_dual_algo_var = tk.BooleanVar()
        self.dual_algo_check = ttk.Checkbutton(control_frame, text=" Use Second Algorithm (operates on original input)",
                                               variable=self.use_dual_algo_var, command=self.toggle_dual_algo_widgets)
        self.dual_algo_check.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(0, 10))
        
        self.algo2_outer_frame = ttk.Frame(control_frame, padding=0)
        self.algo2_outer_frame.grid(row=5, column=0, columnspan=3, sticky=tk.EW, pady=(10, 0))
        self.algo2_outer_frame.columnconfigure(1, weight=1)

        self.algo2_label_widget = ttk.Label(self.algo2_outer_frame, text="Algorithm 2:")
        self.algo2_label_widget.grid(row=0, column=0, sticky=tk.W, padx=(5,10), pady=5)
        self.algo2_var = tk.StringVar()
        self.algo2_combo = ttk.Combobox(self.algo2_outer_frame, textvariable=self.algo2_var, values=self.algo_names,
                                        state="readonly", width=25)
        self.algo2_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        self.algo2_combo.bind("<<ComboboxSelected>>", self.update_controls_for_algo2)
        
        self.op2_label_widget = ttk.Label(self.algo2_outer_frame, text="Operation 2:")
        self.op2_label_widget.grid(row=1, column=0, sticky=tk.W, padx=(5,10), pady=5)
        self.op2_var = tk.StringVar(value="Encrypt")
        self.op2_combo = ttk.Combobox(self.algo2_outer_frame, textvariable=self.op2_var, values=["Encrypt", "Decrypt"],
                                      state="readonly", width=15)
        self.op2_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.key2_label_widget = ttk.Label(self.algo2_outer_frame, text="Key 2:")
        self.key2_label_widget.grid(row=2, column=0, sticky=tk.W, padx=(5,10), pady=5)
        self.key2_entry = ttk.Entry(self.algo2_outer_frame, width=30)
        self.key2_entry.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        action_button_frame = ttk.Frame(main_frame)
        action_button_frame.pack(pady=20)
        self.process_button = ttk.Button(action_button_frame, text="Process Input",
                                         command=self.process_text, width=20, style="Apply.TButton")
        self.process_button.pack()
        
        output_frame = ttk.LabelFrame(main_frame, text=" Result ", padding="15 10")
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output_text = tk.Text(output_frame, height=12, width=80, state=tk.DISABLED, font=self.output_font,
                                   wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, bd=1, bg=self.TEXT_BG_COLOR,
                                   fg=self.TEXT_FG_COLOR, highlightthickness=1, highlightbackground=self.BORDER_COLOR)
        output_scroll = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        output_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,5), pady=(5,5))
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,0), pady=5)
        self.output_text['yscrollcommand'] = output_scroll.set
        
        self.update_controls_for_algo1() # Initial update for algo1 controls
        self.toggle_dual_algo_widgets()  # Initial state for algo2 controls

    def get_key_hint(self, algo_name_selected):
        default_hint = ":"
        if not algo_name_selected:
            return default_hint
        algo_data = self.algorithms.get(algo_name_selected)
        if not algo_data:
            return default_hint
            
        key_type = algo_data.get("key_type")
        if key_type == "number": return " (Integer Shift):"
        elif key_type == "number_ge_2": return " (Rails ≥ 2):"
        elif key_type == "mono_key": return " (26 unique letters a-z):"
        elif key_type == "text_alpha_only": return " (Keyword - letters only):"
        elif key_type == "text_playfair": return " (Keyword - letters, J->I):"
        elif key_type == "text_any": return " (Keyword - any chars):"
        elif key_type == "sdes_key_10bit": return " (10-bit binary e.g. 0101010101):" # S-DES Key
        else: return default_hint

    def update_controls_for_algo1(self, event=None):
        selected_algo = self.algo1_var.get()
        hint = self.get_key_hint(selected_algo)
        self.key1_label.config(text=f"Key 1{hint}")
        
        algo_data = self.algorithms.get(selected_algo)
        if algo_data and algo_data.get("input_type") == "binary_8bit":
            self.input_text_label.config(text="Enter 8-bit binary string (e.g., 01101000):")
        else:
            self.input_text_label.config(text="Enter text:")

    def update_controls_for_algo2(self, event=None):
        if self.use_dual_algo_var.get():
            selected_algo = self.algo2_var.get()
            hint = self.get_key_hint(selected_algo)
            if hasattr(self, 'key2_label_widget') and self.key2_label_widget.winfo_exists() and self.key2_label_widget.winfo_ismapped():
                self.key2_label_widget.config(text=f"Key 2{hint}")
            # Input label for algo2 does not change as it operates on the original input_text content

    def toggle_dual_algo_widgets(self):
        if not hasattr(self, 'algo2_outer_frame'): # Safety check
            return
        if self.use_dual_algo_var.get():
            self.algo2_outer_frame.grid()
            if not self.algo2_var.get() and self.algo_names: # If no algo2 selected, select first one
                 self.algo2_combo.current(0)
            self.update_controls_for_algo2() # Update hint if shown
        else:
            self.algo2_outer_frame.grid_remove()
            self.algo2_var.set("") # Clear selection
            if hasattr(self, 'key2_entry') and self.key2_entry.winfo_exists():
                try:
                    self.key2_entry.delete(0, tk.END)
                except tk.TclError:
                    pass # Widget might be destroyed or not fully initialized

    def process_text(self):
        try:
            text_input_val = self.input_text.get("1.0", tk.END).rstrip("\n")
            algo1_name = self.algo1_var.get()
            op1_str = self.op1_var.get()
            key1 = self.key1_entry.get()

            use_dual = self.use_dual_algo_var.get()
            algo2_name = self.algo2_var.get() if use_dual else None
            op2_str = self.op2_var.get() if use_dual else None
            key2 = self.key2_entry.get() if use_dual else None

            if not text_input_val:
                messagebox.showwarning("Input Missing", "Please enter input to process.")
                self.input_text.focus_set()
                return
            if not algo1_name:
                messagebox.showwarning("Selection Missing", "Please select Algorithm 1.")
                self.algo1_combo.focus_set()
                return
            if use_dual and not algo2_name:
                messagebox.showwarning("Selection Missing", "Please select Algorithm 2.")
                self.algo2_combo.focus_set()
                return
            
            output_lines = []
            
            # Stage 1
            algo1_data = self.algorithms.get(algo1_name)
            if not algo1_data:
                messagebox.showerror("Error", f"Algorithm '{algo1_name}' is not defined.")
                return
            op1_func = algo1_data.get('encrypt' if op1_str == "Encrypt" else 'decrypt')
            if not op1_func:
                raise NotImplementedError(f"Operation '{op1_str}' not available for {algo1_name}.")
            
            intermediate_result = op1_func(text_input_val, key1)
            
            output_lines.append(f"--- Stage 1: {algo1_name} ({op1_str} on Original Input) ---")
            display_key1 = key1 if len(key1) < 40 else key1[:37] + "..."
            output_lines.append(f"Key 1: '{display_key1}'")
            display_text1 = text_input_val if len(text_input_val) < 100 else text_input_val[:97] + "..."
            output_lines.append(f"Original Input for Stage 1: {display_text1}")
            output_lines.append(f"Result of Stage 1: {intermediate_result}")

            # Stage 2
            if use_dual:
                algo2_data = self.algorithms.get(algo2_name)
                if not algo2_data:
                    messagebox.showerror("Error", f"Algorithm '{algo2_name}' is not defined.")
                    return
                
                # Second algorithm also uses the original text_input_val
                op2_func = algo2_data.get('encrypt' if op2_str == "Encrypt" else 'decrypt')
                if not op2_func:
                    raise NotImplementedError(f"Operation '{op2_str}' not available for {algo2_name}.")
                
                final_result_stage2 = op2_func(text_input_val, key2) # Use original input
                
                output_lines.append(f"\n--- Stage 2: {algo2_name} ({op2_str} on Original Input) ---")
                display_key2 = key2 if len(key2) < 40 else key2[:37] + "..."
                output_lines.append(f"Key 2: '{display_key2}'")
                display_text2 = text_input_val if len(text_input_val) < 100 else text_input_val[:97] + "..."
                output_lines.append(f"Original Input for Stage 2: {display_text2}")
                output_lines.append(f"Result of Stage 2: {final_result_stage2}")

            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", "\n\n".join(output_lines)) # Add extra newline
            self.output_text.config(state=tk.DISABLED)

        except ValueError as ve:
            messagebox.showerror("Input/Key Error", f"{ve}")
        except NotImplementedError as nie:
            messagebox.showerror("Function Error", f"{nie}")
        except RuntimeError as rte: # Catch other runtime errors
            messagebox.showerror("Processing Error", f"Internal error:\n{rte}")
            traceback.print_exc()
        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An unexpected error occurred:\n{e}")
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = CryptoApp(root)
    # No need for DES_AVAILABLE check here anymore as S-DES is self-contained
    root.mainloop()