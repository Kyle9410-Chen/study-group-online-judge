import pickle
import gc
import numpy as np

def main():
    with open("./data/unlabeled_attn.pkl", "rb") as f:
        data = pickle.load(f, encoding="latin1")

    print(type(data), len(data))

    tokens: list[list[str]] = [ex["tokens"] for ex in data]
    del data; gc.collect()

    with open("./data/unlabeled_text.txt", "w") as f:
        for toks in tokens:
            words = []
            
            for t in toks:
                if t in ("[CLS]", "[SEP]"):
                    continue

                if t.startswith("##") and words:
                    words[-1] += t[2:]
                
                else:
                    words.append(t)

            f.write(" ".join(words) + "\n")

    print(len(tokens), "examples saved")

if __name__ == "__main__":
    main()