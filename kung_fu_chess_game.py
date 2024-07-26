import chess
from fentoimage.board import BoardImage
import cv2
import numpy as np

class ChessBoard:

    # object that stores board state
    class BitBoard:
        def __init__(self):
            # Initialize piece bitmaps
            self.pawns = 0x0000000000000000
            self.rooks = 0x0000000000000000
            self.knights = 0x0000000000000000
            self.bishops = 0x0000000000000000
            self.queens = 0x0000000000000000
            self.kings = 0x0000000000000000

            # Additional maps
            self.occupied = 0x0000000000000000
            self.white = 0x0000000000000000
            self.black = 0x0000000000000000
            self.enPassant = 0x0000000000000000

            # Kung fu maps
            self.frozen = 0x0000000000000000
            self.moving = 0x0000000000000000

   
    def __init__(self):

        ### CONSTANTS
        # Following map constants used to identify ranks and files on chess board
        self.FILE_MAP = {
            'a': 0x0101010101010101,
            'b': 0x0101010101010101 << 1,
            'c': 0x0101010101010101 << 2,
            'd': 0x0101010101010101 << 3,
            'e': 0x0101010101010101 << 4,
            'f': 0x0101010101010101 << 5,
            'g': 0x0101010101010101 << 6,
            'h': 0x0101010101010101 << 7,
        }

        self.RANK_MAP = {
            1: 0x00000000000000FF,
            2: 0x000000000000FF00,
            3: 0x0000000000FF0000,
            4: 0x00000000FF000000,
            5: 0x000000FF00000000,
            6: 0x0000FF0000000000,
            7: 0x00FF000000000000,
            8: 0xFF00000000000000,
        }

        self.RANKS_TO_Y = {
            1:0,
            2:1,
            3:2,
            4:3,
            5:4,
            6:5,
            7:6,
            8:7
        }

        self.FILES_TO_X = {
            'a':0,
            'b':1,
            'c':2,
            'd':3,
            'e':4,
            'f':5,
            'g':6,
            'h':7
        }

        # Inverse mappings
        self.X_TO_FILES = {v: k for k, v in self.FILES_TO_X.items()}
        self.Y_TO_RANKS = {v: k for k, v in self.RANKS_TO_Y.items()}


        self.bit_board = self.BitBoard()

        return
    
    def bitboard_to_fen(self) -> str:
        rows = []
        for rank in range(7,-1,-1):
            row = ''
            empty_squares = 0
            for file in range(8):
                square = self.xy_to_square_bitmap(file, rank)
                piece = self.piece_at_square(square)
                if piece:
                    if empty_squares > 0:
                        row += str(empty_squares)
                        empty_squares = 0
                    row += piece
                else:
                    empty_squares += 1
            if empty_squares > 0:
                row += str(empty_squares)
            rows.append(row)
        
        fen_board = '/'.join(rows)

        # Placeholder for FEN parts we do not handle yet (turn, castling, en passant, halfmove, fullmove)
        fen_parts = [
            fen_board,
            'w',    # It's white's turn (default)
            '-',    # No castling available (default)
            '-',    # No en passant square (default)
            '0',    # Halfmove clock (default)
            '1'     # Fullmove number (default)
        ]

        return ' '.join(fen_parts)

    def chess_notation_to_bitmap(self, notation):

        # Extract file and rank from the square notation
        file = notation[0]
        rank = int(notation[1])

        if file not in self.FILES_TO_X or rank not in self.RANKS_TO_Y:
            raise ValueError("Invalid square notation: " + notation)

        return self.FILE_MAP[file] & self.RANK_MAP[rank]
    
    def piece_at_square(self,bit_square):

        if self.bit_board.pawns & bit_square != 0:
            return 'P' if self.bit_board.white & bit_square != 0 else 'p'
        if self.bit_board.rooks & bit_square != 0:
            return 'R' if self.bit_board.white & bit_square != 0 else 'r'
        if self.bit_board.knights & bit_square != 0:
            return 'N' if self.bit_board.white & bit_square != 0 else 'n'
        if self.bit_board.bishops & bit_square != 0:
            return 'B' if self.bit_board.white & bit_square != 0 else 'b'
        if self.bit_board.queens & bit_square != 0:
            return 'Q' if self.bit_board.white & bit_square != 0 else 'q'
        if self.bit_board.kings & bit_square != 0:
            return 'K' if self.bit_board.white & bit_square != 0 else 'k'
        return None

    def xy_to_square_bitmap(self, x, y):
        """
        Converts input (x,y) to bitmap 
        """
        return self.FILE_MAP[self.X_TO_FILES[x]] & self.RANK_MAP[self.Y_TO_RANKS[y]]

    def init_starting_position(self):
        
        ### Initialize piece bitmaps
        self.bit_board.pawns = self.RANK_MAP[2] | self.RANK_MAP[7]
        self.bit_board.rooks = ((self.FILE_MAP['a'] | self.FILE_MAP['h']) & (self.RANK_MAP[1] | self.RANK_MAP[8])) 
        self.bit_board.knights = ((self.FILE_MAP['b'] | self.FILE_MAP['g']) & (self.RANK_MAP[1] | self.RANK_MAP[8])) 
        self.bit_board.bishops = ((self.FILE_MAP['c'] | self.FILE_MAP['f']) & (self.RANK_MAP[1] | self.RANK_MAP[8])) 
        self.bit_board.queens = (self.FILE_MAP['d'] & (self.RANK_MAP[1] | self.RANK_MAP[8])) 
        self.bit_board.kings = (self.FILE_MAP['e'] & (self.RANK_MAP[1] | self.RANK_MAP[8])) 

        # Additional maps
        self.bit_board.occupied = self.RANK_MAP[1] | self.RANK_MAP[2] | self.RANK_MAP[7] | self.RANK_MAP[8]
        self.bit_board.white = self.RANK_MAP[1] | self.RANK_MAP[2]
        self.bit_board.black = self.RANK_MAP[7] | self.RANK_MAP[8]
        self.bit_board.enPassant = 0x0000000000000000

        # Kung fu maps
        self.bit_board.frozen = 0x0000000000000000
        self.bit_board.moving = 0x0000000000000000

        # TODO add castling

        return

    def bitmap_to_string(self, bitmap):
        """
        Converts a 64-bit bitmap into an 8x8 string representation.
        
        Parameters:
        -----------
        bitmap : int
            A 64-bit integer representing the bitmap.

        Returns:
        --------
        str
            A string representing the bitmap in an 8x8 grid format.
        """
        rows = []
        for i in range(8):
            # Extract the bits for the current row
            row_bits = (bitmap >> (56 - i * 8)) & 0xFF
            # Convert to a binary string and pad with leading zeros
            row_string = format(row_bits, '08b')
            rows.append(row_string)
        # Join rows with newline characters
        return '\n'.join(rows)

    def move_piece(self, start_square, end_square):
        # TODO kung fu slow moving piece

        start_square_bitmap = self.chess_notation_to_bitmap(start_square)
        end_square_bitmap = self.chess_notation_to_bitmap(end_square)

        # check if piece exists on start square
        if self.bit_board.occupied & start_square_bitmap == 0:
            return
        
        #TODO check if legal move

        # Check if end_square is occupied
            # if occupied remove captured piece
        if self.bit_board.occupied & end_square_bitmap != 0:
            self.remove_square(end_square_bitmap) 
        
        # Move piece
        piece = self.piece_at_square(start_square_bitmap)
        self.add_piece_to_square(piece, end_square_bitmap)
        self.remove_square(start_square_bitmap)

        return
    
    def remove_square(self, square):
        # clears piece from square
        self.bit_board.occupied &= ~square
        self.bit_board.white &= ~square
        self.bit_board.black &= ~square
        self.bit_board.pawns &= ~square
        self.bit_board.knights &= ~square
        self.bit_board.bishops &= ~square
        self.bit_board.queens &= ~square
        self.bit_board.kings &= ~square
        self.bit_board.frozen &= ~square
        self.bit_board.moving &= ~square
    
    def add_piece_to_square(self, piece, square):
        
        self.bit_board.occupied |= square
        
        # check colour and add bit
        if piece.isupper():
            self.bit_board.white |= square
        else:
            self.bit_board.black |= square

        if piece.upper() == 'P':
            self.bit_board.pawns |= square
        elif piece.upper() == 'R':
            self.bit_board.rooks |= square
        elif piece.upper() == 'N':
            self.bit_board.knights |= square
        elif piece.upper() == 'B':
            self.bit_board.bishops |= square
        elif piece.upper() == 'Q':
            self.bit_board.queens |= square
        elif piece.upper() == 'K':
            self.bit_board.kings |= square

def main():
    """
    The main function of the script.
    """
    board = ChessBoard()
    board.init_starting_position()

    


    move = ""
    # Loop to continuously prompt for moves and update the image
    while move != "q":
        
        fen = board.bitboard_to_fen()

        renderer = BoardImage(fen)
        image = renderer.render()

         # Convert PIL image to a format OpenCV can display
        image_np = np.array(image)  # Convert PIL image to NumPy array
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        resized_image = cv2.resize(image_bgr, (256,256), interpolation=cv2.INTER_AREA)

        cv2.imshow('Resized Chess Board', resized_image)
        cv2.waitKey(1)  # Wait 1 millisecond to allow the image to be updated

        move = input()

        start_square = move[:2]
        end_square = move[2:4]
        board.move_piece(start_square, end_square)

    # finish up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
