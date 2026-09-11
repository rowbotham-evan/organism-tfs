import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_WIDTHS = {"600m": 1152}


def main():
    parser = argparse.ArgumentParser(description="Smoke-test a local ESMC model")
    parser.add_argument("--model", choices=MODEL_WIDTHS, default="600m")
    args = parser.parse_args()

    model_path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / f"esmc-{args.model.upper()}"
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using ESMC-{args.model.upper()} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    inputs = tokenizer("MKTIIALSYIFCLVFADYKDDDDK", return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}

    with torch.inference_mode():
        embeddings = model(**inputs).last_hidden_state

    print("Embedding shape:", embeddings.shape)
    assert embeddings.shape[-1] == MODEL_WIDTHS[args.model]


if __name__ == "__main__":
    main()
