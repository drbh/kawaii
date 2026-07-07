use kawaii::{
    int, local_partition, print_2d_swizzled, print_tv, Layout, StaticLayout, Swizzle,
    SwizzledLayout, TvLayout,
};

fn main() {
    // An 8x8 column-major data tile partitioned among 32 threads arranged
    // as an 8x4 column-major grid: each thread owns 2 elements.
    let data = Layout::new(int!(8, 8), None);
    let threads = Layout::new(int!(8, 4), None);

    let tv = TvLayout::from_partition(&data, &threads);
    println!("thread ownership of the 8x8 tile:");
    println!("{}\n", print_tv(&tv, 8, 8));

    let (values, offset) = local_partition(&data, &threads, 5);
    println!(
        "thread 5 owns value layout {} at base offset {}\n",
        values, offset
    );

    // Shared-memory swizzling: an 8x8 row-major tile accessed by column hits
    // one bank 8 times; Sw<3,0,3> spreads each column across all 8 banks.
    let smem = Layout::new(int!(8, 8), Some(int!(8, 1)));
    let swizzled = SwizzledLayout::new(smem, Swizzle::new(3, 0, 3));
    println!("swizzled shared-memory tile (indices mod 8 = bank):");
    println!("{}\n", print_2d_swizzled(&swizzled));

    // Planning -> execution handoff: algebra runs on dynamic layouts, the
    // result lowers to a Copy, heap-free StaticLayout for the kernel.
    let lowered: StaticLayout<2> = data.to_static().unwrap();
    println!(
        "lowered static layout: shape {:?} stride {:?}, element (3,2) -> {}",
        lowered.shape.as_array(),
        lowered.stride.as_array(),
        lowered.index([3, 2])
    );
}
