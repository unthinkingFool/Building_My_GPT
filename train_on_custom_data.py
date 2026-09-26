import torch
import torch.nn.functional as F
from pathlib import Path

from model.gpt import GPT
from data.vocab import Solution as VocabBuilder
from generate import Solution as TextGenerator


# ----------------------------------------------------------------------
# STEP 0: All the settings for this training run, in one place.
# Feel free to tweak these numbers and re-run the script.
# ----------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "data" / "train_data.txt"
CHECKPOINT_FILE = Path(__file__).parent / "trained_gpt.pt"

CONTEXT_LENGTH = 256     # how many characters of history the model looks at
MODEL_DIM = 512          # size of each token's internal representation
NUM_BLOCKS = 6           # how many transformer blocks are stacked
NUM_HEADS = 8            # how many attention heads per block

BATCH_SIZE = 64          # how many examples we train on at once
LEARNING_RATE = 3e-4     # how big a step the optimizer takes each time
TRAINING_STEPS = 1200    # how many batches of training we run
PRINT_EVERY = 200        # how often (in steps) we print the current loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# STEP 1: Load the raw text file.
# ----------------------------------------------------------------------
print(f"Loading text from {DATA_FILE} ...")
text = DATA_FILE.read_text(encoding="utf-8")
print(f"Loaded {len(text):,} characters.")


# ----------------------------------------------------------------------
# STEP 2: Turn the text into numbers.
# A GPT model only understands numbers, not letters, so every unique
# character gets its own ID (e.g. 'a' -> 12, 'b' -> 5, ...).
# ----------------------------------------------------------------------
vocab_builder = VocabBuilder()
char_to_int, int_to_char = vocab_builder.build_vocab(text)
vocab_size = len(char_to_int)
print(f"Vocabulary size: {vocab_size} unique characters.")

token_ids = vocab_builder.encode(text, char_to_int)
data = torch.tensor(token_ids, dtype=torch.long)

# Hold back the last 10% of the text to check the model on text
# it has never trained on ("validation data").
split_point = int(0.9 * len(data))
train_data = data[:split_point]
val_data = data[split_point:]
print(f"Training on {len(train_data):,} characters, "
      f"validating on {len(val_data):,} characters.")


# ----------------------------------------------------------------------
# STEP 3: Build the model.
# ----------------------------------------------------------------------
model = GPT(
    vocab_size=vocab_size,
    context_length=CONTEXT_LENGTH,
    model_dim=MODEL_DIM,
    num_blocks=NUM_BLOCKS,
    num_heads=NUM_HEADS,
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model created with {num_params:,} parameters, running on {DEVICE}.")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# ----------------------------------------------------------------------
# STEP 4: A helper that grabs one random batch of training examples.
#
# Each example is a short chunk of text (length CONTEXT_LENGTH).
# The "answer" for each example is the same chunk, shifted one
# character to the right -- i.e. "predict the next character".
#
#   text:    "hello world"
#   input:   "hello worl"
#   target:  "ello world"   (each position's target is the NEXT character)
#
# Note: model/gpt.py resets PyTorch's global random seed every time the
# model runs (it does this for reproducible test grading). To make sure
# that doesn't force us to pick the exact same batch on every step, we
# use our own private random generator here, just for picking examples.
# ----------------------------------------------------------------------
batch_picker = torch.Generator().manual_seed(1337)


def get_batch(split: str):
    source = train_data if split == "train" else val_data

    start_positions = torch.randint(
        low=0,
        high=len(source) - CONTEXT_LENGTH,
        size=(BATCH_SIZE,),
        generator=batch_picker,
    )

    inputs = torch.stack(
        [source[i: i + CONTEXT_LENGTH] for i in start_positions]
    )
    targets = torch.stack(
        [source[i + 1: i + 1 + CONTEXT_LENGTH] for i in start_positions]
    )

    return inputs.to(DEVICE), targets.to(DEVICE)


# ----------------------------------------------------------------------
# STEP 5: The training loop.
#
# On every step we:
#   a) grab a random batch of examples
#   b) ask the model to predict the next character for each position
#   c) measure how wrong it was (the "loss")
#   d) nudge the model's weights to make it a little less wrong
# Repeating this thousands of times is what "training" means.
# ----------------------------------------------------------------------
print("\nStarting training...\n")

for step in range(1, TRAINING_STEPS + 1):
    inputs, targets = get_batch("train")

    logits = model(inputs)                       # model's predictions
    logits = logits.view(-1, vocab_size)          # flatten for the loss function
    targets = targets.view(-1)                    # flatten to match

    loss = F.cross_entropy(logits, targets)       # how wrong were we?

    optimizer.zero_grad()                         # clear old gradients
    loss.backward()                               # compute new gradients
    optimizer.step()                              # update the weights

    if step % PRINT_EVERY == 0 or step == 1:
        with torch.no_grad():
            val_inputs, val_targets = get_batch("val")
            val_logits = model(val_inputs).view(-1, vocab_size)
            val_loss = F.cross_entropy(val_logits, val_targets.view(-1))

        print(f"step {step:5d}/{TRAINING_STEPS}  "
              f"train_loss={loss.item():.4f}  val_loss={val_loss.item():.4f}")

print("\nTraining finished!")


# ----------------------------------------------------------------------
# STEP 6: Save the trained model so it can be reused later without
# having to retrain from scratch.
# ----------------------------------------------------------------------
torch.save({
    "model_state_dict": model.state_dict(),
    "char_to_int": char_to_int,
    "int_to_char": int_to_char,
    "config": {
        "vocab_size": vocab_size,
        "context_length": CONTEXT_LENGTH,
        "model_dim": MODEL_DIM,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS,
    },
}, CHECKPOINT_FILE)
print(f"Saved trained model to {CHECKPOINT_FILE}")


# ----------------------------------------------------------------------
# STEP 7: Generate a short sample of text, just to see what the model
# has learned so far.
# ----------------------------------------------------------------------
print("\nGenerating a sample of text:\n")

model.eval()
starting_character = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # start token

generated_text = TextGenerator().generate(
    model=model,
    new_chars=300,
    context=starting_character,
    context_length=CONTEXT_LENGTH,
    int_to_char=int_to_char,
)

print(generated_text)
