Long-LoRA(Brand new)
Prompt tuning
데이터양으로 밀어붙이는게 잘 되다보니 거기에 들어가는 돈과 시간이 너무 많이 들어가기때문에 그걸 줄여보자 에서 시작하게 되었고 그것과 연결되어서 Prompt tuning이라는 분야가 나온것이다.(LoRA와 거의 한두달 차이로 관련된 논문이 나왔음)
	최근 chatGPT 열풍에 이어 LLM에 대한 인기가 매우 뜨겁다. 여기서 **문제점으로 언급된 것이 fine-tuning할 때 모델의 파라미터가 너무 많다보니 리소스 제약이 크다는 점**이였다.
Low Rank = 선형대수에서 R-dimension의 Rank

수백차원의 데이터를 2차원으로 줄여도 잘 작동이 되었다
Billion짜리를 17만개로 줄여서 실험을 했는데 더 잘된 결과도 있음
![[Pasted image 20231028131859.png]]

![[Pasted image 20231028132314.png]]
Transformer 각각의 Weight 어디에 적용할때 제일 효과가 좋냐를 실험한것
Wq는 Query Wk는 Key Wv는 Value, Wo는 Fully-connected layer
