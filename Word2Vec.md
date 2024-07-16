Paper: [*NIPS-2013-distributed-representations-of-words-and-phrases-and-their-compositionality-Paper.pdf](file:///C:/Users/gigm2/OneDrive/Desktop/NLP/Sep.30/NIPS-2013-distributed-representations-of-words-and-phrases-and-their-compositionality-Paper.pdf)
Summary: http://jalammar.github.io/illustrated-word2vec/


At that time, for distributed representation of words in vector space helped leaning-algorithm to achieve better performance by ___grouping similar words.___
But then it's hard to find idiomatic phrases for example like it's hard to find connections between Air and Canada to get Air Canada. So this paper is presenting several extensions that improve both quality of vectors and training speed.

Skip-gram model: An efficient model that for learning high-quality vector representations of words from unstructured text data. Since it doesn't involve dense matrix multiplication this makes training extremely efficient.(Able to train 100 billion words in a day)
	Training: to find word representations that are useful for predicting the surrounding words in sentence or document.

Key is using phrase vector instead of word vector. Treat phrases as individual token when training.
And by using them it was able to get meaningful result just by using simple vector addition.
	Ex) result for vector(Russia) + vector(River) was close to vector(Volga river), 
	Or vector(capital) + vector(Germany) results was close to vector(Berlin)
From this, this made able to understand non-obvious language for computer just using basic mathematical operations on word vector.

Subsampling of frequent words
Since for the frequently used words like "the", "in" etc. There is not much important data from it. Which means for rarely used word there is plenty of data. So they focused to find that by using simple subsampling approach. By that it accelerated learning.

Learning phrases
	To learn vector representation for phrases, they found words that appears frequently together, and infrequently in other context. Ex) Like "New York Times" or "Toronto Maple Leafs" appeared as unique token in training data.
