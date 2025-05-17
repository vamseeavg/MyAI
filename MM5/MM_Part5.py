import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

#read all the words
words = open('MyAI/data/tnames.txt', 'r').read().splitlines()
# random.seed(44)
# random.shuffle(words)

#build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

#shuffle up the words
random.seed(42)
random.shuffle(words)

#Build the data set
block_size = 8

def build_dataset(words):

    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            data1 = (''.join(itos[i] for i in context), '---->', itos[ix])
            #print(data1[0])
            #file.write(data1[0] + '-->' + data1[2] + '\n')
            context = context[1:] + [ix] #crop and append
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return(X, Y)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
#-------------------------------------------------------------------------------------------------
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out))/ fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
#------------------------------------------------------------------------------------------------- 
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        #parameters trained with backprop
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        #buffers (trained with a running momentum update)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        #calculate the forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True, unbiased=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) #normalize the unit variance
        self.out = self.gamma * xhat + self.beta
        #update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return[self.gamma, self.beta]
#-------------------------------------------------------------------------------------------------   
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return[]
#-------------------------------------------------------------------------------------------------
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]
#-------------------------------------------------------------------------------------------------
class FlattenConsequtive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    
    def parameters(self):
        return []
#-------------------------------------------------------------------------------------------------
class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
#-------------------------------------------------------------------------------------------------
torch.manual_seed(42)

#MLP revisited
n_embd = 24 # the dimentionality of the character embedding vectors
n_hidden = 128 # the number of neurons in the hidden layer of MLP

C = torch.randn((vocab_size,n_embd))

model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsequtive(2), Linear(n_embd *2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsequtive(2), Linear(n_hidden *2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsequtive(2), Linear(n_hidden *2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])

#parameter init
with torch.no_grad():
    model.layers[-1].weight *= 0.1 #last layer: make less confident

parameters = model.parameters()
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

#same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    #minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X, Y

    #Forward pass
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    #Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #update
    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    #track stats
    if i % 10000 == 0: # print once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

#plt.plot(lossi)
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
plt.savefig("MakeMore/MM5/Lossi.png")

#put layers into eval mode (needed for batch norm especially)
for layer in model.layers:
    layer.training = False

# evaluate the loss
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')

#sample from the modal
for _ in range(20):
    out = []
    context = [0] * block_size #initialize with all
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        #sample for m the distribution
        ix = torch.multinomial(probs, num_samples=1).item()
        # shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)
        # if we sample '.' break
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))