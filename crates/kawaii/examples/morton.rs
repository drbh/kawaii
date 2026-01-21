use kawaii::{blocked_product, int, print_2d, IntTuple, Layout};

// Based on
// https://youtu.be/vzUhbDO_0qk?t=2499

fn main() {
    // Create a 4x8 matrix layout (column-major by default)
    let morton1 = Layout::new(int!(2, 2), Some(int!(1, 2)));
    let morton2 = blocked_product(&morton1, &morton1);
    let morton3 = blocked_product(&morton1, &morton2);

    println!("Hierarchical morton code");
    println!("{}", print_2d(&morton3));

    // logical coords 1D, 2D, hD
    println!("{}", morton3.call(&int!(37))); // 49
    println!("{}", morton3.call(&int!(5, 4))); // 49
    println!("{}", morton3.call(&int!(int!(1, 2), int!(0, 2)))); // 49
    println!(
        "{}",
        morton3.call(&int!(int!(1, int!(0, 1)), int!(0, int!(0, 1))))
    ); // 49
}
