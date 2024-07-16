![[2311.03285.pdf]]

S-LoRA: system designed for the scalable serving of many LoRA adapters.

stores all adapters in the main memory and fetches the adapters used by the currently running queries to the GPU memory

LoRa enables efficient fine-tuning by updating only low-rank additive matrices.
LoRa showed that by fine-tuning just these adapter weights, it is possible to achieve performance on par with full-weight fine-tuning(adapter weights를 fine-tuning 하는것만으로도 full-wright fine-tuning과 비슷한 성능을 내는게 가능하도록 만듬)
One of the key innovation of LoRA was elimination of adapter inference latency by directly merging the adapter with the model parameters. 
In the training phase, LoRA freezes the weights of a pre-trained base model and adds trainable low-rank matrices to each layer. This approach significantly reduces the number of trainable parameters and memory consumption.

In this paper, S-LoRA, they will show how to scalably serve thousands of LoRA adapters on single machine.

KV Caching: 모델의 중간 출력을 캐시 메모리에 저장해놓고 이후 처리 요청이 들어올 시 이를 재사용할 수 있게 해주는 기술. 이는 메모리 관리를 더 효율적으로 수행하게 하여 메모리 사용량을 줄이는 동시에 처리 속도를 높일수 있는데, 특히 큰 모델이나 높은 볼륨의 수행요청 처리시 메모리부족 문제를 해결할 수 있다.
(https://contents.premium.naver.com/banya/banyacompany/contents/231025111858328hy)

3 contribution of S-LoRA
	*Unified paging: To reduce memory fragmentation and increase batch size, introduces unified memory pool. This pool manages dynamic adapter weights and KV cache tensors by unified page mechanism. 
	Heterogeneous Batching: To minimize latency when batching different adapters of varying ranks, S-LoRA employs highly optimized custom CUDA kernels. And these operates directly non-contiguous memory and align with the memory pool design
	*S-LoRA TP: for effective parallelization across multiple GPUs, introduces novel tensor parallelism strategy. this approach incurs minimal communication costs for the added LoRA computation compared to base model. 
	
