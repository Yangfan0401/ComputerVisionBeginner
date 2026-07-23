> RNN(Recurrent Neural Network): Process Sequences
>
> One input to many output: 
>
> Eg. Image Captioning: Image-> sequence of words
>
> Many input to one output:
>
> Eg. Video classification: Sequence of images -> label
>
> Many input to many ouput:
>
> Eg. Machine Translation or Pre-frame video classification

> Key idea
>
> An `internal state` updated as s sequence 

**Fomula：**

$h_t=f_W(h_{t-1},\ x_t)$

1. $h_t$ denotes a new state
2. $f_W$ denotes some function with parameters W (Sharing)
3. $h_{t-1}$ denotes old state
4. $x_t$ denotes input vector at now

<font color="red">**Notice: **</font>The same function f and the same set of partameters W in the whole sequence.

## Vanilla RNN

$h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$

$y_t = W_{hy}h_t + b_y$

$\sigma()$ denotes a activation function using `tanh()`

> A simple RNN Computational Graph(Many to Many)

![EgVanillaRNN](/Users/yangfan/Documents/LocalProjects/Computer Vision/Image Caption/assets/EgVanillaRNN.png)

> Vanilla RNN Gradient Flow

![VanillaGradientFlow](/Users/yangfan/Documents/LocalProjects/Computer Vision/Image Caption/assets/VanillaGradientFlow.png)

* Largest singular value > 1: **Exploding gradients**

**Grdient clipping**

* Largest singular value < 1: **Vanishing gradients**

**Change RNN Architecture**

### Image Captiopn

[COCO Captions dataset](http://cocodataset.org/) a standard testbed for image captioning.

$Image \xrightarrow{CNN} Features \xrightarrow[Fully\  Connected]{Linear \ Transformation} expanded\_features$**$(h_0)$**

$Words\_Index \xrightarrow{Words\ Embedding} words\_vector$**(x with (B, T, W) shape)**

T：the dimension of sequence also denotes the tokens number

W：the words number of one token 


## LSTM(Long Short Term Memory)

- LSTMs can solve training on long sequences due to vanishing and exploding gradients caused by repeated matrix multiplication by replacing the simple update rule with a **gating mechanism**

> **Update rule：**
> Similar to the canilla RNN receive input and previous hidden state
> LSTM also maintains an H-dimensional cell state, so also receive the previous cell state
> Rather than, the learnable parameters of the LSTM are **4H dimension** such an input-to-hidden matrix $W_x \in \mathbb{R}^{4H \times D}$, a hidden-hidden matrix $W_h \in \mathbb{R}^{4H \times H}$, and a bias vector $b \in \mathbb{4H}$
$$
\left(
\begin{matrix}
i_t \\
f_t \\
o_t \\
g_t \\
\end{matrix}
\right) =
\left(
\begin{matrix}
\sigma \\
\sigma \\
\sigma \\
tanh \\
\end{matrix}
\right)
\left(
\begin{matrix}
W
\left(
\begin{matrix}
h_{t-1} \\
x_t
\end{matrix}
\right)
+ b_h \\
\end{matrix}
\right)
$$

A gate mechanism algorithm using 

* **$i_t：$**Input gate
* **$f_t：$**Forget gate
* **$o_t：$**Output gate
* **$g_t：$**Gate gate

Chunk with hidden layer for gates each H 

> **Summary：**
> $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ 
> $h_t = o_t \odot tanh(c_t)$

![LSTM Gradients Flow](/Users/yangfan/Documents/LocalProjects/Computer Vision/Image Caption/assets/LSTM Gradients Flow.png)

### One steptime LSTM forward

```python
out = x @ self.Wx + prev_h @ self.Wh + self.b # N, 4H
i, f, o, g = torch.chunk(out, chunks=4, dim=1)
i = F.sigmoid(i) # N, H
f = F.sigmoid(f) # N, H
o = F.sigmoid(o) # N, H
g = F.tanh(g) # N, H
next_c = f * prev_c + i * g
next_h = o * F.tanh(next_c)
```

## Sequence to Sequence with RNNS
**Input: Sequence **
**Output: Sequence**

- Encoder: $h_t = f_W(x_t, h_{t-1})$
  - final hidden state predict: $s_i$(decoder state) as decoder hidden vector
  - Context vector c(often choose $h_t$)

- Decoder: $s_t = g_U(y_{t-1}, s_{t-1}, c)$

## Attention

Use $s_t$ and $h_t$ computing alignment scores(attention scores),like $e_{t, i} =f_{att}(s_{t-1}, h_i)$

$f_{att}$ is an MLP

- Softmax alignment scores to get attention weights $a_{t,i}$
- Compute context vector as linear combination of hidden states $c_{t} = \sum_i a_{t,i} h_{i}$

New Decoder: $s_t = g_{U}(y_{t-1}, s_{t-1}, c_t)$

Attention Network 把cell state从gate输出转为Context Attention Mechanism

### Attention Layer

**Input：**

- Query vector：q
- Input vector：X
- Similarity function：$f_{att}$(**Vector Dot Product** also using **scaled** dot product)
- **Key Matrix：**$W_k$
- **Value Matrix：**$W_v$

**Computation：**

- Similarities：$e_i = f_{att}(q, X_i)=q \cdot X_i$ or equals to  $\frac{1}{\sqrt{D_Q}}q \cdot X_i$
- Attention weights：a = softmax(e)
- Output vector：y=$\sum_i a_i X_i$
  - **Key vectors：** $K = XW_k$ $\rightarrow$like Hidden vectors on Attention RNNs
  - **Value vectors：**
  -  $V = XW_v$ $\rightarrow$like Hidden vectors on Attention RNNS
  - **Similariteis：** $E = QK^T / \sqrt{D_Q}, E_{i, j} = \frac{1}{\sqrt{D_Q}}((Q_i \cdot K_j))$
  - **Attention weights：** A = softmax(E)
  - **Output vectos：**Y = AV

### Attention-RNN

Same encoder

* Diffirential decoder

see last encoder output as $s_i$ as state vectors 

dot product between $s_i$(last input in encoder layer) and $h_t$ hidden vectors from encoder layer produce `attention scores` and then computing softmax gets `attention weights`

using attention weights compute dot product with hidden vectos and then sum up getting the `cell state`

![Attention-RNN](/Users/yangfan/Documents/LocalProjects/Computer Vision/Image Caption/assets/Attention-RNN.png)

### Attention Mechanism

引入query、key、values these three vectors

#### Self-Attention Layer

One `query` per `input vector`

- Similarities：$e_i = f_{att}(q, X_i)=q \cdot X_i$ or equals to  $\frac{1}{\sqrt{D_Q}}q \cdot X_i$

- Attention weights：a = softmax(e)

- Output vector：y=$\sum_i a_i X_i$

  - **Qeury vectos：**$Q=XW_q$ $\rightarrow$like Encoder last output vectos on Attention RNNs
  - **Key vectors：** $K = XW_k$ $\rightarrow$like Hidden vectors on Attention RNNs
  - **Value vectors：**
  -  $V = XW_v$ $\rightarrow$like Hidden vectors on Attention RNNS
  - **Similariteis：** $E = QK^T / \sqrt{D_Q}, E_{i, j} = \frac{1}{\sqrt{D_Q}}((Q_i \cdot K_j))$
  - **Attention weights：** A = softmax(E)
  - **Output vectos：**Y = AV

  通过三个Channels两个Channels通过Linear Transformation做投影 再做Matrix Transpose(vectors dot product)以及softmax得到注意力权重，用第三个Channel与注意力权重在进行Matrix Transpose得到输出结果

  三个Channels同一个input不共享Don't Sharing Weights

#### Masked Self-Attention Layer

**Don't let vectos 'look ahead' in the sequence**

Use some mechanism like Dropout to drop out some attention weights

#### Multihead Self-Attention

Use **H independent 'Attention Heads'** in parallel

![Multihead-SelfAttention](/Users/yangfan/Documents/LocalProjects/Computer Vision/Image Caption/assets/Multihead-SelfAttention.png)

Split input vectors along dim=1 and concat along dim=0(split along row and then concat along column) to compute independent attention results

## Transformer

### **Framework:**

![Transformer frame](/Users/yangfan/Documents/LocalProjects/Computer Vision/Image Caption/assets/Transformer frame.png)

依赖3个核心的Queries, keys and values(Q, K, V) sharing inputs x and  three different **weights**

- Compared to every other vector to establish the weights for its own output $y_i$	$Q = W_q X$
- Compared to every other vector to establish the weights for the output of the j-th vector $y_j$	$K = W_k X$
- Used as part of the weighted sum to compute each output vector once the weights have been established	$V = W_v X$



Distll how "self-attention" it works

`The animal didn't cross the street because it was too tired`

Algorithm is hard to understand `it`  meanig, but `self-attetion`  allows  it to  associate `it` with `animal`



### Part I. Preparation
#### Pre-processing

input sequence(string) convert them into a sequence of tokens known as **tokenizations**.

* Build a vocabulary a list of all unique tokens mapping into unique integer value(`consequetive`)

```python
token_dict = {} ########################################################################TTODO: Use this function to assign a unique whole number element to each element present in the vocab list. To do this, map the first element in the vocab to 0 and the last element in the vocab to len(vocab), and the elements in between as consequetive number.    ########################################################################
    i = 0
    for word in vocab:
        token_dict[word] = i
        i += 1
```

* Preprocess_input_sequence



### MultiHead Attention on Transformers

#### 

#### Masked Attention

Prevent the decoder from looking ahead into the future




























