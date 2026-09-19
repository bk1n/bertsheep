import os

# huggingface_hub reads this once at import, so it has to be set before any
# test module imports transformers. Tests use the locally cached model and must
# never hit the network.
os.environ["HF_HUB_OFFLINE"] = "1"
