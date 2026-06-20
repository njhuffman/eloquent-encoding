# Move Predictor

# High Level Idea

Ultimately what I want is a model that takes in the last N board states and predicts what the player's
next move will be. To achieve this, I am expecting a model architecture that has
- A GRU taking in the last N moves, compressing that down to a small style vector
- An MLP taking in the current board, a proposed move, and the style embedding. It outputs a single compatibility value

To use the predictor, you would give multiple proposed moves. The one with the highest output value is the
most likely move.

# Training
To start training, I will need to sample a bunch of games, store consecutive moves and possible
moves that were not chosen. To bootstrap, here is my plan:
- each training sample will have 3 moves: the chosen one, and two random non-chosen moves
- train the model to minimize cross entropy
Once that is done, use that model to create a new dataset
- each training sample will have 3 moves: the chosen one, the top non-chosen move based on the first predictor, and a random non-chosen move
- train the new model using these hard and easy negatives

To keep the data processing simple, run through a pgn file
- skip games with some probability
- for kept games, run through moves, skipping with some probability
- also skip any states where the player has only 1 or 2 moves
- for the kept moves, generate an entry for the database
  - the current board embedding using a model from embeddings/
  - the last N board embeddings using the same model
  - index pairs for the chosen and non-chosen moves, as [from, to] pairs in range [0, 63]

To train, a "move" is always represented as a concatenated [board_embedding, from_embedding, to_embedding]
The board_embedding uses a learned model, trained in the embedding/ module. The from and to embeddings
are just simple nn.Embedding with 64 different options.