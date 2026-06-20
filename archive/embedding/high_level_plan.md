Here is the high level plan for create a self-supervised chess embedding.

Idea: use a MAE (Masked AutoEncoder) to create an embedding, from the board state vector
to a fixed size embedding.

# Steps for Data Collection:
- create a script that takes in a pgn file with games to process
- each game is randomly assigned to either train, validation, or test (80/10/10 split)
- within each game, look at the board states played
  - create a function that turns a board state into an 8x8x17 one-hot vector. Use python-chess library for this
    - the first 6 represent positions of each type of piece for white
    - the next 6 represent positions for each type of piece for blac
    - the next 1 gives the color who's turn it is
    - the next 4 give castling rights for each color on each side (full layer 0 or 1)
    - the next 1 gives en passant righs, 0 everywhere exect for the position where en passant can happen
  - skip the opening board. No need for every game to add this to the data set
  - for the first 10 moves, skip each with 90% chance. There will be lots of duplicates
  - for the remaining, skip with 50% chance. Just need a sampling of options
  - for ones that are not skipped, compute
    - the board tensor (8x8x17). This is the main thing used to create the embedding
    - the elo of each color. Will be used to validate embedding later
    - the number of pieces each side has. Will be used to validate embedding later
    - the final game outcome (-1 for black wins, 0 for draw, 1 for white wins). Will be used to validate embedding later.
    - is the current player in check. Will be used to validate embedding later.
  - save these to an hdf5 file

Some things to note:
- likely need to save to hdf5 in batches. Needs to support millions of boards.
- should report progress along the way with tqdm
- should save training splits to literally different files to avoid data leakage mistakes
- should put all hardcoded values at top to be tweaked later.


# Steps for model training
- Use a CNN based autoencoder
- Encoder Input: 8x8x18 board: the board vector plus a mask
  - pieces where the mask is 0 should be zerod out
- Encoder Output: 128 value embedding
- Decoder Input: the 8x8 mask and 128 value embedding
- Decoder Ouptut: 8x8x12 piece mask, showing where white and black pieces are
- Decoder Loss: Difference from the true non-masked 8x8x12 piece mask. Only apply loss to the masked regions.
