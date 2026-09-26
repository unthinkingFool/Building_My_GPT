import sys
import torch
from pathlib import Path

from model.gpt import GPT
from data.vocab import Solution as VocabBuilder
from generate import Solution as TextGenerator


CHECKPOINT_FILE = Path(__file__).parent / "trained_gpt.pt"
NEW_CHARS_TO_GENERATE = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# STEP 1: Load the checkpoint saved during training.
# It contains the trained weights, the vocabulary, and the model's config.
# ----------------------------------------------------------------------
print(f"Loading checkpoint from {CHECKPOINT_FILE} ...")
checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)

char_to_int = checkpoint["char_to_int"]
int_to_char = checkpoint["int_to_char"]
config = checkpoint["config"]


# ----------------------------------------------------------------------
# STEP 2: Rebuild the model with the exact same shape it was trained
# with, then load the trained weights into it.
# ----------------------------------------------------------------------
model = GPT(
    vocab_size=config["vocab_size"],
    context_length=config["context_length"],
    model_dim=config["model_dim"],
    num_blocks=config["num_blocks"],
    num_heads=config["num_heads"],
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # turns off dropout, since we're generating, not training
print("Model loaded and ready.")


# ----------------------------------------------------------------------
# STEP 3: Build the starting context.
# If you pass a prompt on the command line, we encode it into token IDs.
# Otherwise we start from a single blank token.
# ----------------------------------------------------------------------
vocab_builder = VocabBuilder()

if len(sys.argv) > 1:
    prompt = sys.argv[1]
    prompt_ids = vocab_builder.encode(prompt, char_to_int)
    context = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
else:
    prompt = ""
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)


# ----------------------------------------------------------------------
# STEP 4: Generate new characters, one at a time, and print the result.
# ----------------------------------------------------------------------
print(f"\nPrompt: {prompt!r}\n")
print("Generating...\n")

generated_text = TextGenerator().generate(
    model=model,
    new_chars=NEW_CHARS_TO_GENERATE,
    context=context,
    context_length=config["context_length"],
    int_to_char=int_to_char,
)

print(prompt + generated_text)
