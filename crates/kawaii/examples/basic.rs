use kawaii::{int, print_2d, IntTuple, Layout};

// Based on quick example in
// https://youtu.be/vzUhbDO_0qk?t=2160

fn main() {
    // Create a 4x8 matrix layout (column-major by default)
    let mut layout = Layout::new(int!(4, 8), Some(int!(1, 4)));

    println!("Column-major");
    println!("{}", print_2d(&layout));

    // now make it row-major by updating the strides
    layout.stride = int!(8, 1);
    println!("Row-major");
    println!("{}", print_2d(&layout));

    // now make it col-major padded
    layout.stride = int!(1, 5);
    println!("Column-major padded");
    println!("{}", print_2d(&layout));

    // now make it col-major interleave
    layout.shape = int!(4, int!(4, 2));
    layout.stride = int!(4, int!(1, 16));
    println!("Column-major interleave");
    println!("{}", print_2d(&layout));

    // now make it mixed
    layout.shape = int!(int!(2, 2), int!(4, 2));
    layout.stride = int!(int!(1, 8), int!(2, 16));
    println!("Column-major interleave");
    println!("{}", print_2d(&layout));

    // Access some linear indices
    let index = layout.call(&int!(1, 2)); // row 1, column 2
    println!("row 1, column 2 is at linear index: {}", index);
    let index = layout.call(&int!(3, 5)); // row 3, column 5
    println!("row 3, column 5 is at linear index: {}", index);
}
