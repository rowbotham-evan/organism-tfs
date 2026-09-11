"""Mean-pool ESMC-600M residue embeddings at every second layer for every TF.

`output_hidden_states=True` returns 37 tensors: index 0 is the token embedding
(no context), indices 1-35 are the raw outputs of transformer blocks 1-35, and
index 36 is block 36 after the final LayerNorm, identical to the
`last_hidden_state` saved by encode_all_tfs.py. Only the pooled vectors are
kept: storing every residue at 18 layers would take roughly 60 GB.
"""

import csv

import torch
from transformers import AutoModel, AutoTokenizer

from encode_all_tfs import DATASET_CONFIGS, MODEL_CONFIGS, PROJECT_DIR


MODEL = MODEL_CONFIGS["600m"]
LAYERS = tuple(range(2, 37, 2))
DATASETS = ("ecoli", "pao1", "hvolc", "nrc1")
OUTPUT = (
    PROJECT_DIR / "embedding" / MODEL["slug"]
    / f"{MODEL['slug']}_layer_pooled_tf_embeddings.pt"
)


def check_final_layer(key, pooled):
    """Layer 36 must match the mean of the saved final-layer embeddings."""
    records = torch.load(
        PROJECT_DIR / "embedding" / MODEL["slug"]
        / f"{MODEL['slug']}_{DATASET_CONFIGS[key]['output_suffix']}",
        map_location="cpu",
        weights_only=False,
    )
    saved = torch.stack([record["embedding"].float().mean(dim=0) for record in records])
    final = pooled[:, LAYERS.index(36)]
    if saved.shape != final.shape or not torch.allclose(saved, final, atol=1e-4):
        difference = (saved - final).abs().max() if saved.shape == final.shape else None
        raise ValueError(f"{key}: layer 36 differs from saved embeddings ({difference})")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {MODEL['name']} on {device}; layers {LAYERS}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL["path"])
    model = AutoModel.from_pretrained(MODEL["path"]).to(device).eval()
    if model.config.num_hidden_layers != LAYERS[-1]:
        raise ValueError(f"Expected {LAYERS[-1]} layers, got {model.config.num_hidden_layers}")

    results = {"model": MODEL["name"], "layers": list(LAYERS), "datasets": {}}
    for key in DATASETS:
        with DATASET_CONFIGS[key]["path"].open(newline="") as source:
            proteins = list(csv.DictReader(source))

        pooled = []
        for protein in proteins:
            sequence = protein["sequence"]
            inputs = tokenizer(
                sequence,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            special_tokens = inputs.pop("special_tokens_mask").to(device)
            inputs = {name: value.to(device) for name, value in inputs.items()}

            with torch.inference_mode():
                hidden_states = model(**inputs, output_hidden_states=True).hidden_states

            residue_mask = inputs["attention_mask"].bool() & ~special_tokens.bool()
            if int(residue_mask.sum()) != len(sequence):
                raise ValueError(f"{protein['gene']}: residue count mismatch")
            pooled.append(torch.stack([
                hidden_states[layer][residue_mask].float().mean(dim=0).cpu()
                for layer in LAYERS
            ]))

        pooled = torch.stack(pooled)
        if not torch.isfinite(pooled).all():
            raise ValueError(f"{key}: non-finite pooled vectors")
        check_final_layer(key, pooled)
        results["datasets"][key] = {
            "genes": [protein["gene"] for protein in proteins],
            "pooled": pooled,
        }
        print(f"  {key}: {pooled.shape[0]} proteins -> {tuple(pooled.shape)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, OUTPUT)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
