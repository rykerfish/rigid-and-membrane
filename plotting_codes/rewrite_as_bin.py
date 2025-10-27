import numpy as np
import json
import os


def main():

    fname = "blob_pos.csv"
    data = np.loadtxt(fname, delimiter=",")

    with open("binary_metadata.json", "w") as f:
        metadata = {
            "row_size": data.shape[1],
            "n_rows": data.shape[0],
            "dtype": "float32",
        }
        json.dump(metadata, f, indent=4)

    out_file = "blob_pos.bin"
    with open(out_file, "wb") as f:
        data.astype("float32").tofile(f)


def write_row_binary(out_dir, row, N, out_dtype="float32") -> None:
    pos_file = out_dir + "colloids.bin"
    meta_file = out_dir + "binary_metadata.json"

    row = np.array(row, dtype=out_dtype)
    if not os.path.exists(meta_file):
        metadata = {
            "row_size": row.size,
            "N": N,
            "n_rows": 1,  # account for row about to be written
            "dtype": out_dtype,
        }
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)
    else:
        with open(meta_file, "r") as f:
            metadata = json.load(f)
            metadata["n_rows"] += 1
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)

    with open(pos_file, "ab") as f:
        row.tofile(f)


if __name__ == "__main__":
    main()
