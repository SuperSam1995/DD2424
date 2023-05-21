# %% [markdown]
# ### Needed imports

# %%
import numpy as np

import matplotlib.pyplot as plt


# %% [markdown]
# ### Read in the data

# %%
with open('datasets/goblet_book.txt', 'r') as f:
    #Read the text file
    book_data = f.read()

    #Find the unique characters in the book and the length K
    book_chars = sorted(set(book_data))
    K = len(book_chars)

    #Create dictionaries for character to index and index to character conversion
    char_to_ind = {char: i for i, char in enumerate(book_chars)}
    ind_to_char = {i: char for i, char in enumerate(book_chars)}


''' convert a sequence of characters into a sequence of vectors 
    of one-hot encodings and vice versa
'''
def char_seq_to_one_hot(char_seq, char_to_ind, K):
    N = len(char_seq)
    one_hot_seq = np.zeros((K, N))
    for i, char in enumerate(char_seq):
        one_hot_seq[char_to_ind[char], i] = 1
    return one_hot_seq

def one_hot_seq_to_char_seq(one_hot_seq, ind_to_char):
    N = one_hot_seq.shape[1]
    char_seq = ''.join([ind_to_char[np.argmax(one_hot_seq[:, i])] for i in range(N)])
    return char_seq

# Example usage:
char_seq = 'hello'
one_hot_seq = char_seq_to_one_hot(char_seq, char_to_ind, K)
decoded_char_seq = one_hot_seq_to_char_seq(one_hot_seq, ind_to_char)

print(decoded_char_seq)  # Should print 'hello'






# %% [markdown]
# ### Set hyper-parameters & initialize the RNN’s parameters

# %%
import numpy as np

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1, seq_length=25, sigma=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.seq_length = seq_length
        
        self.b = np.zeros((hidden_dim, 1))
        self.c = np.zeros((output_dim, 1))
        
        self.U = np.random.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
        self.W = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.V = np.random.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim)

        # Define the __getstate__ method to handle pickling
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    # Define the __setstate__ method to handle unpickling
    def __setstate__(self, state):
        self.__dict__.update(state)

# Set hyperparameters
m = 100
eta = 0.1
seq_length = 25

# Initialize the RNN
rnn = RNN(K, m, K, learning_rate=eta, seq_length=seq_length)


# %% [markdown]
# ### Synthesize text from your randomly initialized RNN

# %%
def synthesize_sequence(rnn, h0, x0, n):
    x = x0
    h = h0
    Y = np.zeros((rnn.output_dim, n))

    for t in range(n):
        h = np.tanh(np.dot(rnn.U, x) + np.dot(rnn.W, h) + rnn.b)
        o = np.dot(rnn.V, h) + rnn.c
        p = np.exp(o) / np.sum(np.exp(o))

        ix = np.random.choice(range(rnn.output_dim), p=p.ravel())
        x_next = np.zeros((rnn.input_dim, 1))
        x_next[ix] = 1

        Y[:, t] = x_next.ravel()
        x = x_next

    return Y

# Example usage:
h0 = np.zeros((m, 1))
x0 = char_seq_to_one_hot('.', char_to_ind, K)
n = 100

Y = synthesize_sequence(rnn, h0, x0, n)
char_seq_synthesized = one_hot_seq_to_char_seq(Y, ind_to_char)

print(char_seq_synthesized)


# %% [markdown]
# ### Implement the forward & backward pass of back-prop

# %%
#forwards pass
def forward_pass(rnn, X, Y, h0):
    n = X.shape[1]
    h, o, p = {}, {}, {}
    h[-1] = np.copy(h0)
    loss = 0

    for t in range(n):
        h[t] = np.tanh(np.dot(rnn.U, X[:, t:t+1]) + np.dot(rnn.W, h[t-1]) + rnn.b)
        o[t] = np.dot(rnn.V, h[t]) + rnn.c
        p[t] = np.exp(o[t]) / np.sum(np.exp(o[t]))
        loss += -np.log(p[t][Y[:, t].argmax(), 0])

    return loss, h, p

#backwards pass
def backward_pass(rnn, X, Y, h, p):
    grads = {
        'U': np.zeros_like(rnn.U),
        'W': np.zeros_like(rnn.W),
        'V': np.zeros_like(rnn.V),
        'b': np.zeros_like(rnn.b),
        'c': np.zeros_like(rnn.c)
    }

    dh_next = np.zeros_like(h[0])
    n = X.shape[1]

    for t in reversed(range(n)):
        dL_do = np.copy(p[t])
        dL_do[Y[:, t].argmax()] -= 1

        grads['V'] += np.dot(dL_do, h[t].T)
        grads['c'] += dL_do

        dL_dh = np.dot(rnn.V.T, dL_do) + dh_next
        dL_da = (1 - h[t] * h[t]) * dL_dh

        grads['U'] += np.dot(dL_da, X[:, t:t+1].T)
        grads['W'] += np.dot(dL_da, h[t-1].T)
        grads['b'] += dL_da

        dh_next = np.dot(rnn.W.T, dL_da)

    return grads

#gradient clipping
def clip_gradients(grads, threshold=5):
    for k in grads:
        grads[k] = np.clip(grads[k], -threshold, threshold)
    return grads

# Example usage:
X_chars = book_data[:seq_length]
Y_chars = book_data[1:seq_length + 1]

X = char_seq_to_one_hot(X_chars, char_to_ind, K)
Y = char_seq_to_one_hot(Y_chars, char_to_ind, K)

h0 = np.zeros((m, 1))

loss, h, p = forward_pass(rnn, X, Y, h0)
grads = backward_pass(rnn, X, Y, h, p)
grads = clip_gradients(grads)

print(loss)



# %% [markdown]
# ### Train your RNN using AdaGrad

# %% [markdown]
# include a graph of the smooth loss function for a longish training
# run (at least 2 epochs)

# %%
import matplotlib.pyplot as plt
import pickle

def train_rnn(rnn, book_data, char_to_ind, ind_to_char, n_epochs=7):
    K = rnn.output_dim
    seq_length = rnn.seq_length

    iter_per_epoch = len(book_data) // seq_length
    n_updates = n_epochs * iter_per_epoch

    smooth_loss = -np.log(1.0 / K) * seq_length
    hprev = np.zeros((rnn.hidden_dim, 1))
    ada_params = {k: np.zeros_like(getattr(rnn, k)) for k in ['U', 'W', 'V']}

    e = 0
    best_rnn = None
    best_loss = np.inf
    
    smooth_losses = []
    for update_step in range(n_updates):
        if e == 0 or e + seq_length + 1 > len(book_data):
            e = 1
            hprev = np.zeros((rnn.hidden_dim, 1))

        X_chars = book_data[e - 1 : e - 1 + seq_length]
        Y_chars = book_data[e : e + seq_length]

        X = char_seq_to_one_hot(X_chars, char_to_ind, K)
        Y = char_seq_to_one_hot(Y_chars, char_to_ind, K)

        loss, h, p = forward_pass(rnn, X, Y, hprev)
        grads = backward_pass(rnn, X, Y, h, p)
        grads = clip_gradients(grads)

        for k in ada_params.keys():
            ada_params[k] += grads[k] * grads[k]
            updated_param = getattr(rnn, k) - rnn.learning_rate * grads[k] / (np.sqrt(ada_params[k]) + 1e-8)
            setattr(rnn, k, updated_param)

        smooth_loss = 0.999 * smooth_loss + 0.001 * loss
        smooth_losses.append(smooth_loss)

        #if update_step % 100 == 0:
            #print("Smooth loss at step {}: {}".format(update_step, smooth_loss))

        #if update_step % 500 == 0:
            #Y_synthesized = synthesize_sequence(rnn, hprev, X[:, :1], 200)
            #synthesized_seq = one_hot_seq_to_char_seq(Y_synthesized, ind_to_char)
            #print("Synthesized text:\n", synthesized_seq)

        if smooth_loss < best_loss:
            best_loss = smooth_loss
            best_rnn = rnn

            with open('best_rnn.pkl', 'wb') as f:
                pickle.dump(best_rnn, f)

        e += seq_length

    return smooth_losses

# Call the train_rnn function to train the RNN and get smooth losses
smooth_losses = train_rnn(rnn, book_data, char_to_ind, ind_to_char, n_epochs=2)

# Plot the smooth loss function
plt.plot(smooth_losses)
plt.xlabel('Update step')
plt.ylabel('Smooth loss')
plt.title('Smooth loss function over 2 epochs')
plt.show()


# %% [markdown]
# Show the evolution of the text synthesized by your RNN during
# training by including a sample of synthesized text (200 characters
# long) before the first and before every 10,000th update steps when
# you train for 100,000 update steps

# %%
import pickle

def train_rnn(rnn, book_data, char_to_ind, ind_to_char, n_updates=100000):
    K = rnn.output_dim
    seq_length = rnn.seq_length

    iter_per_epoch = len(book_data) // seq_length
    n_epochs = n_updates // iter_per_epoch

    smooth_loss = -np.log(1.0 / K) * seq_length
    hprev = np.zeros((rnn.hidden_dim, 1))
    ada_params = {k: np.zeros_like(getattr(rnn, k)) for k in ['U', 'W', 'V']}

    e = 0
    best_rnn = None
    lowest_smooth_loss = float('inf')
    for update_step in range(n_updates):
        if e == 0 or e + seq_length + 1 > len(book_data):
            e = 1
            hprev = np.zeros((rnn.hidden_dim, 1))

        X_chars = book_data[e - 1 : e - 1 + seq_length]
        Y_chars = book_data[e : e + seq_length]

        X = char_seq_to_one_hot(X_chars, char_to_ind, K)
        Y = char_seq_to_one_hot(Y_chars, char_to_ind, K)

        if update_step % 10000 == 0:
            Y_synthesized = synthesize_sequence(rnn, hprev, X[:, :1], 200)
            synthesized_seq = one_hot_seq_to_char_seq(Y_synthesized, ind_to_char)
            print("Synthesized text at update step {}:\n{}".format(update_step, synthesized_seq))

        loss, h, p = forward_pass(rnn, X, Y, hprev)
        grads = backward_pass(rnn, X, Y, h, p)
        grads = clip_gradients(grads)

        for k in ada_params.keys():
            ada_params[k] += grads[k] * grads[k]
            updated_param = getattr(rnn, k) - rnn.learning_rate * grads[k] / (np.sqrt(ada_params[k]) + 1e-8)
            setattr(rnn, k, updated_param)

        smooth_loss = 0.999 * smooth_loss + 0.001 * loss

        if update_step % 10000 == 0:
            print("Smooth loss at step {}: {}".format(update_step, smooth_loss))

        if smooth_loss < lowest_smooth_loss:
            lowest_smooth_loss = smooth_loss
            best_rnn = rnn

            with open('best_rnn.pkl', 'wb') as f:
                pickle.dump(best_rnn, f)

        e += seq_length

# Call the train_rnn function to train the RNN and synthesize text at specified steps
train_rnn(rnn, book_data, char_to_ind, ind_to_char, n_updates=100000)


# %%
import pickle

def synthesize_best_model_text(n_chars=1000):
    # Load the best model
    with open('best_rnn.pkl', 'rb') as f:
        best_rnn = pickle.load(f)

    # Choose an initial character (you can replace this with any character from the dataset)
    initial_char = 'T'

    # Convert the initial character to one-hot representation
    x0 = char_seq_to_one_hot(initial_char, char_to_ind, best_rnn.output_dim)

    # Synthesize the text using the best model
    h0 = np.zeros((best_rnn.hidden_dim, 1))
    Y = synthesize_sequence(best_rnn, h0, x0, n_chars)

    # Convert the one-hot representation back to a character sequence
    synthesized_text = one_hot_seq_to_char_seq(Y, ind_to_char)
    
    return synthesized_text

# Generate a passage of length 1000 characters from the best model
synthesized_text = synthesize_best_model_text(1000)
print(synthesized_text)



