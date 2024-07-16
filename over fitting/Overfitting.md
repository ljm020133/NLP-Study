For 1D when you try to find best fitting graph you are missing some datas.
![[Overfiting1D 1.png]]
Even for 2D you found better graph than 1D but still missing some datas
![[Overfiting2D 1.png]]
So when you go up to n dimension you can find best fitting graph that connects every data. 
![[OverfitingnD.png]]
But... the problem is that the data you used to train model is "train data" and the data you actually use when you run model is "running data"
![[OverfitingnDrd 1.png]]
Which means best fitting graph is only for training data. So it's not best fitting graph for running data. That's what overfitting is. And to fix this issue there is two way to fix it.
	1. put more data that can fit to best fitting graph
	2. Down scale your dimension to lower dimension
