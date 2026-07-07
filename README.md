# かわいい (kawaii)

> [!NOTE]
> This is a work in progress and mainly for my own learning purposes.

`kawaii` is a Rust port of parts of NVIDIA's cute cutlass indexing math. It does not aim to be a complete port but rather just a rewrite of the layout and shape abstractions.

```bash
cargo run --example basic
```

```rust
use kawaii::{int, layout, print_2d, IntTuple, Layout};

fn main() {
    // Create a 4x6 matrix layout (column-major by default)
    let mut layout = Layout::new(int!(4, 6), Some(int!([1, 4])));
    // or, equivalently: layout!((4, 6)) / layout!((4, 6), (1, 4))

    println!("Column-major 4x3 layout:");
    println!("{}", print_2d(&layout));

    // now make it row-major by updating the strides
    layout.stride = int!([6, 1]);
    println!("Row-major 4x3 layout:");
    println!("{}", print_2d(&layout));

    // Access some linear indices
    let index = layout.call(&int!(1, 2)); // row 1, column 2
    println!("row 1, column 2 is at linear index: {}", index);
    let index = layout.call(&int!(3, 5)); // row 3, column 5
    println!("row 3, column 5 is at linear index: {}", index);
}
// Column-major 4x3 layout:
// (4,6):(1,4)
//        0    1    2    3    4    5
//     +----+----+----+----+----+----+
//  0  |  0 |  4 |  8 | 12 | 16 | 20 |
//     +----+----+----+----+----+----+
//  1  |  1 |  5 |  9 | 13 | 17 | 21 |
//     +----+----+----+----+----+----+
//  2  |  2 |  6 | 10 | 14 | 18 | 22 |
//     +----+----+----+----+----+----+
//  3  |  3 |  7 | 11 | 15 | 19 | 23 |
//     +----+----+----+----+----+----+
// Row-major 4x3 layout:
// (4,6):(6,1)
//        0    1    2    3    4    5
//     +----+----+----+----+----+----+
//  0  |  0 |  1 |  2 |  3 |  4 |  5 |
//     +----+----+----+----+----+----+
//  1  |  6 |  7 |  8 |  9 | 10 | 11 |
//     +----+----+----+----+----+----+
//  2  | 12 | 13 | 14 | 15 | 16 | 17 |
//     +----+----+----+----+----+----+
//  3  | 18 | 19 | 20 | 21 | 22 | 23 |
//     +----+----+----+----+----+----+
// row 1, column 2 is at linear index: 8
// row 3, column 5 is at linear index: 23
```

### Beyond layout algebra

On top of the CuTe layout/shape math, kawaii has the pieces needed to act as a
kernel-side layout language:

- **Swizzles** — `Swizzle` (runtime, for planning) and `StaticSwizzle<B, M, S>`
  (const-generic, `const fn`, for device code): XOR-based index permutations for
  bank-conflict-free shared memory. `SwizzledLayout` composes one with a layout,
  `print_2d_swizzled` visualizes the result.
- **Static lowering** — `Layout::to_static::<R>()` lowers a dynamic layout to a
  flat, `Copy`, heap-free `StaticLayout<R>`. The intended split: do layout
  algebra dynamically on the host, lower the result, and pass it to a kernel as
  a monomorphized parameter. Construction infers rank from plain arrays —
  `StaticLayout::new([32, 32], [96, 1])` — and `dims()` / `strides()` give the
  arrays back.
- **Thread/value ownership** — `TvLayout` is a rank-2 layout
  `(thread, value) -> offset` describing which thread owns which elements.
  `local_partition` / `local_tile` mirror CuTe's partitioning helpers.
- **Debug rendering** — `print_tv` draws the thread-ownership map of a tile:

```text
(T32,V2) ((8,4),(1,2)):((1,8),(0,32))
          0       1       2       3       4       5       6       7
    +-------+-------+-------+-------+-------+-------+-------+-------+
 0  |  T0V0 |  T8V0 | T16V0 | T24V0 |  T0V1 |  T8V1 | T16V1 | T24V1 |
    +-------+-------+-------+-------+-------+-------+-------+-------+
 1  |  T1V0 |  T9V0 | T17V0 | T25V0 |  T1V1 |  T9V1 | T17V1 | T25V1 |
    ...
```

See `cargo run --example tv` for the full tour.

### References 
https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/index.html

### License

This software includes code derived from NVIDIA Corporation,
licensed under the BSD 3-Clause License.
