如果你现在刚开始学，不要直接上完整大工程。先学这个缩减版：

mini_llm/
├─ configs/
│  ├─ model.yaml
│  └─ train.yaml
├─ src/
│  ├─ model.py
│  ├─ dataset.py
│  ├─ trainer.py
│  ├─ generate.py
│  └─ utils.py
├─ scripts/
│  ├─ run_pretrain.py
│  └─ run_chat.py
├─ outputs/
└─ README.md
这个版本适合你先建立认知：

model.py：模型结构
dataset.py：样本构造
trainer.py：训练循环
generate.py：推理生成
scripts/：启动入口
configs/：实验配置
等你把这一版吃透，再升级到我前面给你的完整版。