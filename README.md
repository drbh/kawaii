# かわいい (kawaii)

> [!NOTE]
> This is a work in progress and mainly for my own learning purposes.

`kawaii` is a Rust port of parts of NVIDIA's cute cutlass indexing math. It does not aim to be a complete port but rather just a rewrite of the layout and shape abstractions.

```bash
cargo run --example basic
```

```rust
use kawaii::{int, print_2d, IntTuple, Layout};

fn main() {
    // Create a 4x6 matrix layout (column-major by default)
    let mut layout = Layout::new(int!(4, 6), Some(int!([1, 4])));

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

### References 
https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/index.html

### License

This software includes code derived from NVIDIA Corporation,
licensed under the BSD 3-Clause License.
