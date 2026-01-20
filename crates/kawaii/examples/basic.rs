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
