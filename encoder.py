#%%
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
#%% Token and Encoding

s = '''
cat sat on couch
mat is red
cat is white
couch is brown
dog sat on mat
dog is golden
'''

tokens = s.split()
token_ids = {token: idx for idx, token in enumerate(set(tokens))}
encoded = [token_ids[token] for token in tokens]
encoded

#%% Creating Embedding Matrix
def embedding_matrix(encoded,embedding_dim=8):
    return torch.randn(size=(len(encoded),embedding_dim),generator=torch.random.manual_seed(40))
    
embedding_matrix = embedding_matrix(encoded,64)
#%%
#%%
print("tokens:",tokens)
print("token_ids:",token_ids)
print("encoded:",encoded)
embeddings = [embedding_matrix[token_id] for token_id in encoded]
print("Embeddings for the itorchut tokens:")
for token, embed in zip(tokens, embeddings):
    print(f"{token}: {embed}")
#%% Postional ecoding given matrix
def positional_encoding(m): # m = embedding_matrix
    pe=torch.zeros(m.shape)
    d=m.shape[1]
    for pos in range(len(m)):
        for i in range(len(m[pos])):
            rad = torch.tensor(pos / (10000 ** (2 * (i // 2) / d)))
            if i%2==0:
                pe[pos][i] = torch.sin(rad)
            else:
                pe[pos][i] = torch.cos(rad)
    return pe

positional_matrix = positional_encoding(embedding_matrix)
positional_matrix
#%%
plt.style.use('default')
plt.subplot(2,1,1)
plt.imshow(embedding_matrix)
plt.colorbar(label="Encoding Value")
plt.xlabel("Dimensions")
plt.ylabel("Position in Sequence")
plt.title("Embedding Matrix Heatmap")

plt.subplot(2,1,2)
plt.imshow(positional_matrix)
plt.colorbar(label="Encoding Value")
plt.xlabel("Dimensions")
plt.ylabel("Position in Sequence")
plt.title("Positional Encoding Heatmap")

plt.tight_layout() 
plt.show()
# %% Itorchut to Encoder
input_matrix = embedding_matrix + positional_matrix
plt.subplot(2,1,2)
plt.imshow(input_matrix)
plt.colorbar()
plt.xlabel("Dimensions")
plt.ylabel("Position in Sequence")
plt.title("Itorchut Heatmap")
plt.tight_layout() 
plt.show()

# %% Cosine Similarity
def cos_sim(a,b):
    return a@b / (torch.linalg.norm(a)*torch.linalg.norm(b))

a = embedding_matrix[token_ids['red']]
b = embedding_matrix[token_ids['white']]
print("embeddings:",cos_sim(a,b))
a = positional_matrix[token_ids['red']]
b = positional_matrix[token_ids['white']]
print("positional:",cos_sim(a,b))
a = input_matrix[token_ids['red']]
b = input_matrix[token_ids['white']]
print("itorchut:",cos_sim(a,b))
# %%
# x = embedding_matrix[1]
# plt.plot(x,torch.sin(2/100**(2*x/8)) )
# plt.show()
# %%
dmodel = 64
heads = 8
dk = dmodel/heads
# %%
def MHA(H,dmodel,heads):
    dk = torch.tensor(dmodel//heads)
    Wq = torch.randn(size=(heads,dmodel,dk),generator=torch.random.manual_seed(41))
    print("Wq:", Wq.shape)
    Wk = torch.randn(size=(heads,dmodel,dk),generator=torch.random.manual_seed(42))
    Wv = torch.randn(size=(heads,dmodel,dk),generator=torch.random.manual_seed(43))

    Q = torch.einsum('td,hdq->htq',H,Wq)
    print("Q:", Q.shape)
    K = torch.einsum('td,hdk->htk',H,Wk)
    V = torch.einsum('td,hdv->htv',H,Wv)

    sc_qk_transpose = torch.einsum('htq,hsk->hts',Q,K) / torch.sqrt(dk)
    print("sc_qk_transpose:", sc_qk_transpose.shape)
    soft = torch.softmax(sc_qk_transpose, dim=-1)
    attn = torch.einsum('hij,hjk->hik',soft,V)
    print("attn:", attn.shape)

    z = attn.permute(1,0,2).reshape(20,-1)
    print("z:", z.shape)

    Wo = torch.randn(size=(dk*heads,dmodel),generator=torch.random.manual_seed(44))
    out = torch.einsum('tp,pd->td',z,Wo)
    print("out:", out.shape)

    return out

mha_out = MHA(input_matrix,dmodel,heads)
plt.style.use('default')
plt.subplot(3,1,1)
plt.imshow(input_matrix)
plt.colorbar(label="Encoding Value")
plt.xlabel("Dimensions")
plt.ylabel("Position in Sequence")
plt.title("input_matrix Heatmap")

plt.subplot(3,1,2)
plt.imshow(mha_out)
plt.colorbar(label="Encoding Value")
plt.xlabel("Dimensions")
plt.ylabel("Position in Sequence")
plt.title("mha_out Heatmap")

plt.subplot(3,1,3)
plt.imshow(input_matrix+mha_out)
plt.colorbar(label="Encoding Value")
plt.xlabel("Dimensions")
plt.ylabel("Position in Sequence")
plt.title("input_matrix+mha_out Heatmap")

plt.tight_layout() 
plt.show()
#%%
j = torch.tensor([[1,2,3],[1,1,1]], dtype=float)
print(j,j.shape)
mu=torch.mean(j,dim=-1)
print("mu:",mu)
var=torch.var(j,dim=-1)
print("var:",var)
(j-mu.reshape(j.shape[0],1))/ var.reshape(j.shape[0],1)

# %% Layer Normalisation
v = embedding_matrix + mha_out

def layer_normalization(v):
    mu = torch.mean(v,dim=-1)
    var = torch.var(v,dim=-1)
    alpha = (v-mu.reshape(v.shape[0],1))/ var.reshape(v.shape[0],1)
    gama,beta = 0.1,0.01
    return gama*alpha + beta
    
norm_mha = layer_normalization(v)
plt.subplot(2,1,1)
plt.imshow(v)
plt.colorbar()
plt.xlabel("Dimensions")
plt.ylabel("Position in Sequence")
plt.title("v Heatmap")

plt.subplot(2,1,2)
plt.imshow(norm_mha)
plt.colorbar()
plt.xlabel("Dimensions")
plt.ylabel("Position in Sequence")
plt.title("norm_mha Heatmap")

plt.tight_layout() 
plt.show()
# %%
def ffn(x):
    in_dim = x.shape[1]
    x = nn.Linear(in_dim,128)(x)
    x = nn.ReLU()(x)
    x = nn.Linear(128,in_dim)(x)
    return x
ffn_out = ffn(norm_mha)
plt.subplot(2,1,1)
plt.imshow(norm_mha)
plt.colorbar()
plt.xlabel("Dimensions")
plt.ylabel("Position in Sequence")
plt.title("v Heatmap")

plt.subplot(2,1,2)
plt.imshow(ffn_out.detach().numpy())
plt.colorbar()
plt.xlabel("Dimensions")
plt.ylabel("Position in Sequence")
plt.title("norm_mha Heatmap")

plt.tight_layout() 
plt.show()
# %%
norm_ffn = layer_normalization(ffn_out+norm_mha)
plt.subplot(2,1,1)
plt.imshow(ffn_out.detach().numpy())
plt.colorbar()
plt.xlabel("Dimensions")
plt.ylabel("Position in Sequence")
plt.title("v Heatmap")

plt.subplot(2,1,2)
plt.imshow(norm_ffn.detach().numpy())
plt.colorbar()
plt.xlabel("Dimensions")
plt.ylabel("Position in Sequence")
plt.title("norm_mha Heatmap")

plt.tight_layout() 
plt.show()
# %%