


## 单机训练

下面我们给一个微调 Baichuan2-7B-Base 的单机训练例子。

训练数据：`data/belle_chat_ramdon_10k.json`，该样例数据是从 [multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M) 采样出 1 万条，并且做了格式转换。主要是展示多轮数据怎么训练，不保证效果。

```shell
cd fine-tune

hostfile=""
deepspeed --hostfile=$hostfile --include localhost:0,1,2,3,6 --master_port 29501  fine-tune.py  \
    --report_to "none" \
    --data_path "data/belle_chat_ramdon_10k.json" \
    --model_name_or_path "/ssd1/share/Baichuan2-13B-Chat" \
    --output_dir "output/13B" \
    --model_max_length 4096 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --bf16 True \
    --tf32 True
```


## 命令行工具方式

```shell
python cli_demo.py
```
本命令行工具是为 Chat 场景设计，因此我们不支持使用该工具调用 Base 模型。

## 网页 demo 方式

依靠 streamlit 运行以下命令，会在本地启动一个 web 服务，把控制台给出的地址放入浏览器即可访问。本网页 demo 工具是为 Chat 场景设计，因此我们不支持使用该工具调用 Base 模型。

```shell
streamlit run web_demo.py
```