from mlx_lm import load
from mlx_lm.tuner.trainer import train, TrainingArgs
from mlx_lm.tuner import datasets
import mlx.optimizers as optim

# Load model/tokenizer
model, tokenizer = load("meta-llama/Llama-3.2-8B-Instruct")

# Load dataset
train_set = datasets.load_jsonl_dataset("./jsonl/non-diverse/train.jsonl", tokenizer, seq_len=512)

# Training args
args = TrainingArgs(
    iters=5000,
    batch_size=1,
    grad_accum_steps=8,  # <- simula batch=8
    max_seq_length=512,
    steps_per_report=10,
    steps_per_eval=100,
)

optimizer = optim.Adam(learning_rate=1e-5)

train(
    model=model,
    train_dataset=train_set,
    args=args,
    optimizer=optimizer,
)