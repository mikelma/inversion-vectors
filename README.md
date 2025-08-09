# Inversion vectors 🔄 

I ❤️ $\overrightarrow{V}_\sigma^>$ 

<br>

This repository contains the source code and data of the paper *"A Unified View of Bijective Transformations for Optimizing Permutation Problems"* (ECAI 2025).

<br>

Abstract: 

*Many optimization algorithms represent solutions as permutations. However, despite their apparent simplicity, permutations pose significant challenges—especially for Global Random Search (GRS) algorithms—due to the mutual-exclusivity constraint. This constraint complicates both the learning and sampling of probability distributions over the permutation space, often leading to computationally expensive procedures.*
*A promising alternative involves transforming permutation-encoded solutions into integer vectors using bijective functions on the symmetric group* $\mathbb{S}_n$, *resulting in what are known as inversion vectors. While inversion vectors have been studied for centuries, a unified and formal framework encompassing all their codifications has been lacking.*
*In this paper, we introduce precise definitions and a unified notation for various types of inversion vector codifications. We establish bijective transformations between them, providing a formal characterization of their relationships and properties. Leveraging this theoretical foundation, we analyze and explain the behavior of GRS algorithms across different permutation problems when using different inversion vector representations.*

## Getting started 🤓

Although not required, this repository is ready to be used with [`uv`](https://docs.astral.sh/uv/). The first step is to install `uv` (if not already installed).

Once `uv` is working, clone the repo and run the scripts using the following commands in the repo's directory:

```bash
uv run main_umda_permus.py # see --help for CLI options
# or
uv run main_umda_inv.py  # for UMDA over inversion vectors
```

Note that the `umda.py` (contains the UMDA algorithm implementation) and `transformations.py` (implements all permutation transformations and their inverses) are standalone modules that can be used for other projects.

Finally, the main experiment scripts require the `pypermu` library for loading instances and evaluation. A binary of the library is included in the repo (should work on x86_64 Linux machines) together with its source code (in Rust).  

## License

This repository is distributed under the terms of the GLPv3 license. See [LICENSE](./LICENSE) for more details.
