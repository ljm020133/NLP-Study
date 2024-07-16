https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf


Abstract:
	Since, dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include encoder and decoder. Even best performing models also connect encoder and decoder. So they propose a new simple network architecture called "Transformer". On experiment about two machine translation task, we can see this model have superior quality while being parallelizable and requiring significantly reduced amount of time to train.

Introduction:
	Recurrent neural network have been firmly established as best approaches in sequence modeling and transduction problems like language modeling and machine translation. Plus, there were many efforts to push boundaries of it's field like recurrent language model and encoder-decoder architecture. When recurrent model aligning the positions to steps in computation time, they generate "sequence of hidden state" ,$$h_t$$as a function of the "previous hidden state" $$h_t-1$$and input for position *t* . This sequential nature prevent parallelization within training examples, which become critical at longer sequence lengths, as memory constraint limit batching across examples. And this recent work achieved significant improvement in computational efficiency through factorization tricks and conditional computation, plus improving model performance in case of plan B. But, the fundamental limitation of sequential computation still remains.
	$$$$Attention mechanism have become an essential part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequence. But in few of all cases, some attention mechanisms are used by combine with recurrent network.

Background
	Self-attention
		which also called intra-attention, is attention mechanism relating different positions of single sequence in order to compute a representation of the sequence. It has been used successfully in many fields like reading comprehension, abstractive summarization, textual entailment, and learning task-independent sentence representation.$$$$
	End-to-end memory networks
		It is based on recurrent attention mechanism which showed high performance on simple language question answering and language modeling tasks.$$$$
		Transformer is first transduction model relying entirely on self-attention to compute representations of its inputs and outputs without using sequence aligning RNNs or convolution.

Model architecture
	Most competitive neural sequence transduction model have an encoder-decoder structure.
		Encoder maps an input sequence of symbol representation $$(x_1, \dots, x_n)$$to a sequence of continuous representations. $$z = (z_1, \dots, x_n)$$Given *z*, the decoder then generates an output sequence $$(y_1, \dots, y_n)$$of symbols one element at a time. At every step, the model is auto-regressive, and consume the previously generated symbol as additional input when generating next.$$$$
		Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.


Convolution(합성곱)
	확률을 계산하는데 쓰이는 방식으로 결과가 정해져있을경우 같은 결과값이 나올 확률을 모두 더한 확률을 구하는데 쓰이는 방식.
		예) 공을 바닥에 수직으로 떨어뜨릴때, 시작지점에서 처음으로 떨어진곳을 A 그리고 그 자리에서 다시한번 똑같이 떨어뜨려서 튀어나간 곳을 B라고 하고, 총 A에서 B까지 거리를 C 라고 할때, C라는 결과값이 10으로 정해져있다면, 총 10중에 A가 1 B가 9이건, A가 5 B가 5이건 결국 결과가 같으면 되는것이므로 결과가 10이되는 A와 B의 모든 확률을 다 더한 확률을 구하는 방법이 합성곱이다.
			확률분포를 f라고 하면 A에 도달할 확률은 f(A)가 될것이고 똑같이 확률분포 g가 있을때 B에 도달할 확률은 g(B)가 될것이다. 그렇다면 C에 도달할 확률은 $$f(A)*g(B)$$가 된다.
				그렇기에 C를 구하는 공식은 $$(f*g)(c) = \displaystyle\sum_{a+b=c} f(A)*g(B)$$
Softmax function()
	softmax function converts vector K the real number into probability distribution of K possible outcomes. 확률은 퍼센트의 영역이고 그말은 즉 최고값이 1이다 1 == 100% 소프트맥스 함수는 많은 확률값들중에  1이 넘는 값들을 다 1 안쪽으로 넣어주는 함수라고 생각하면 됨.
	