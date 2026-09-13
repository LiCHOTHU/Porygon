import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from real_robot import PROMPT


MODEL = "openai/clip-vit-base-patch32"


def encode_prompt(prompt, output):
    tokenizer = CLIPTokenizer.from_pretrained(MODEL)
    model = CLIPTextModelWithProjection.from_pretrained(MODEL).eval()
    tokens = tokenizer([prompt], padding=True, return_tensors="pt")
    with torch.inference_mode():
        embedding = model(**tokens).text_embeds[0]
        embedding = torch.nn.functional.normalize(embedding, dim=0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embedding.cpu().numpy().astype(np.float32))
    print(f"Saved {embedding.numel()}D CLIP embedding for '{prompt}' to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--output", default="real_robot/cache/prompt_embedding.npy")
    args = parser.parse_args()
    encode_prompt(args.prompt, args.output)


if __name__ == "__main__":
    main()
