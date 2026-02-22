# Vector

> Custom modern-C++ vector implementation.
> Small vector optimization.
> Exception safety and proper object lifecycle management.
> Alignment guarantee for the internal stack buffer.
> Efficient vector swaps.

Project for [Effective C++ Programming](https://bilakniha.cvut.cz/en/predmet6071806.html) course taught by [Daniel Langr](https://usermap.cvut.cz/profile/c217508e-1760-49d3-815c-108d8b2ff596), written in C++ in 2025.
The following tables show benchmark performance of my `epc::vector` compared to other well-known implementations.

| Implementation | GCC (nanoseconds) | Clang (nanoseconds) |
| ------ | ------ | ------ |
| `epc::vector` | 19,992 | **20,844** |
| `std::vector` | 43,050 | 46,205 |
| `boost::container::small_vector` | **19,983** | 23,145 |
| `llvm::SmallVector` | 23,749 | 24,507 |

| Implementation | GCC (%) | Clang (%) |
| ------ | ------ | ------ |
| `epc::vector` | +0.05 | baseline |
| `std::vector` | +115.43 | +121.67 |
| `boost::container::small_vector` | baseline | +11.04 |
| `llvm::SmallVector` | +18.85 | +17.57 |
