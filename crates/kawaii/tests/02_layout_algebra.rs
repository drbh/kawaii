use kawaii::{
    coalesce, coalesce_by_mode, complement, composition, composition_with_tile, flat_divide, int,
    logical_divide, logical_divide_with_tile, logical_product, tiled_divide, zipped_divide,
    IntTuple, Layout, Tile,
};

// coalesce
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#coalesce

#[test]
fn test_layout_algebra_coalesce() {
    // (2,(1,6)):(1,(6,2)) => 12:1
    let layout = Layout::new(int!(2, int!(1, 6)), Some(int!(1, int!(6, 2))));
    let result = coalesce(&layout);
    assert_eq!(result.to_string(), "12:1");
}

// by-mode-coalesce
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#by-mode-coalesce

#[test]
fn test_layout_algebra_by_mode_coalesce() {
    // (2,(1,6)):(1,(6,2)) with profile (1,1) => (2,6):(1,2)
    let a = Layout::new(int!(2, int!(1, 6)), Some(int!(1, int!(6, 2))));
    let result = coalesce_by_mode(&a, &int!(1, 1));
    assert_eq!(result.to_string(), "(2,6):(1,2)");
}

// composition
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#composition

#[test]
fn test_layout_algebra_composition() {
    // A = (6,2):(8,2), B = (4,3):(3,1)
    // R = A o B = ((2,2),3):((24,2),8)
    let a = Layout::new(int!(6, 2), Some(int!(8, 2)));
    let b = Layout::new(int!(4, 3), Some(int!(3, 1)));
    let r = composition(&a, &b);
    assert_eq!(r.to_string(), "((2,2),3):((24,2),8)");
}

// computing-composition
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#computing-composition

#[test]
fn test_layout_algebra_computing_composition() {
    // Example 1: A = (6,2):(8,2), B = (4,3):(3,1)
    // R = ((2,2),3):((24,2),8)
    let a = Layout::new(int!(6, 2), Some(int!(8, 2)));
    let b = Layout::new(int!(4, 3), Some(int!(3, 1)));
    let r = composition(&a, &b);
    assert_eq!(r.to_string(), "((2,2),3):((24,2),8)");

    // Example 2: 20:2 o (5,4):(4,1) => (5,4):(8,2)
    // Reshape layout 20:2 as a 5x4 matrix in row-major order
    let a2 = Layout::new(int!(20), Some(int!(2)));
    let b2 = Layout::new(int!(5, 4), Some(int!(4, 1)));
    let r2 = composition(&a2, &b2);
    assert_eq!(r2.to_string(), "(5,4):(8,2)");

    // Example 3: (10,2):(16,4) o (5,4):(1,5)
    // Reshape as 5x4 matrix in column-major order
    // Coalesced result: (5,(2,2)):(16,(80,4))
    let a3 = Layout::new(int!(10, 2), Some(int!(16, 4)));
    let b3 = Layout::new(int!(5, 4), Some(int!(1, 5)));
    let r3 = composition(&a3, &b3);
    assert_eq!(r3.to_string(), "(5,(2,2)):(16,(80,4))");
}

// by-mode-composition
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#by-mode-composition

#[test]
fn test_layout_algebra_by_mode_composition() {
    // a = (12,(4,8)):(59,(13,1))
    let a = Layout::new(int!(12, int!(4, 8)), Some(int!(59, int!(13, 1))));

    // tiler = <3:4, 8:2> (apply 3:4 to mode-0, 8:2 to mode-1)
    let tiler = Tile::new(vec![
        Layout::new(int!(3), Some(int!(4))),
        Layout::new(int!(8), Some(int!(2))),
    ]);

    // result = (3,(2,4)):(236,(26,1))
    let result = composition_with_tile(&a, &tiler);
    assert_eq!(result.to_string(), "(3,(2,4)):(236,(26,1))");

    // a = (12,(4,8)):(59,(13,1))
    let a = Layout::new(int!(12, int!(4, 8)), Some(int!(59, int!(13, 1))));

    // tiler = <3:1, 8:1> (shape interpreted as stride-1 layouts)
    let tiler = Tile::new(vec![
        Layout::new(int!(3), None), // interpret as 3:1
        Layout::new(int!(8), None), // interpret as 8:1
    ]);

    // result = (3,(4,2)):(59,(13,1))
    let result = composition_with_tile(&a, &tiler);
    assert_eq!(result.to_string(), "(3,(4,2)):(59,(13,1))");
}

// complement
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#complement

#[test]
fn test_layout_algebra_complement() {
    // complement(4:1, 24) is 6:4
    let a = Layout::new(int!(4), Some(int!(1)));
    let r = complement(&a, Some(24));
    assert_eq!(r.to_string(), "6:4");

    // complement(6:4, 24) is 4:1
    let a = Layout::new(int!(6), Some(int!(4)));
    let r = complement(&a, Some(24));
    assert_eq!(r.to_string(), "4:1");

    // complement((4,6):(1,4), 24) is 1:0
    let a = Layout::new(int!(4, 6), Some(int!(1, 4)));
    let r = complement(&a, Some(24));
    assert_eq!(r.to_string(), "1:0");

    // complement(4:2, 24) is (2,3):(1,8)
    let a = Layout::new(int!(4), Some(int!(2)));
    let r = complement(&a, Some(24));
    assert_eq!(r.to_string(), "(2,3):(1,8)");

    // complement((2,4):(1,6), 24) is 3:2
    let a = Layout::new(int!(2, 4), Some(int!(1, 6)));
    let r = complement(&a, Some(24));
    assert_eq!(r.to_string(), "3:2");

    // complement((2,2):(1,6), 24) is (3,2):(2,12)
    let a = Layout::new(int!(2, 2), Some(int!(1, 6)));
    let r = complement(&a, Some(24));
    assert_eq!(r.to_string(), "(3,2):(2,12)");
}

// division-tiling
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#division-tiling

#[test]
fn test_layout_algebra_division_tiling() {
    // 1-D Example
    // A = (4,2,3):(2,1,8), B = 4:2
    // Result: ((2,2),(2,3)):((4,1),(2,8))
    let a = Layout::new(int!(4, 2, 3), Some(int!(2, 1, 8)));
    let b = Layout::new(int!(4), Some(int!(2)));
    let result = logical_divide(&a, &b);
    assert_eq!(result.to_string(), "((2,2),(2,3)):((4,1),(2,8))");

    // 2-D Example with Tile
    // A = (9,(4,8)):(59,(13,1))
    // B = <3:3, (2,4):(1,8)>
    // Result: ((3,3),((2,4),(2,2))):((177,59),((13,2),(26,1)))
    let a2 = Layout::new(int!(9, int!(4, 8)), Some(int!(59, int!(13, 1))));
    let tiler = Tile::new(vec![
        Layout::new(int!(3), Some(int!(3))),
        Layout::new(int!(2, 4), Some(int!(1, 8))),
    ]);
    let result2 = logical_divide_with_tile(&a2, &tiler);

    assert_eq!(
        result2.to_string(),
        "((3,3),((2,4),(2,2))):((177,59),((13,2),(26,1)))"
    );
}

// TODO: below require more inspection to verify correctness

// zipped-tiled-flat-divides
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#zipped-tiled-flat-divides

#[test]
fn test_layout_algebra_zipped_tiled_flat_divides() {
    // A: shape is (9,32) = (9,(4,8)):(59,(13,1))
    let layout_a = Layout::new(int!(9, int!(4, 8)), Some(int!(59, int!(13, 1))));

    // B: shape is (3,8) = <3:3, (2,4):(1,8)>
    let tiler = Tile::new(vec![
        Layout::new(int!(3), Some(int!(3))),
        Layout::new(int!(2, 4), Some(int!(1, 8))),
    ]);

    // logical_divide: ((TileM,RestM), (TileN,RestN)) with shape ((3,3), (8,4))
    let ld = logical_divide_with_tile(&layout_a, &tiler);
    assert_eq!(ld.shape.get(0).size(), 9); // 3*3
    assert_eq!(ld.shape.get(1).size(), 32); // 8*4

    // zipped_divide: ((TileM,TileN), (RestM,RestN)) with shape ((3,8), (3,4))
    let zd = zipped_divide(&layout_a, &tiler);
    assert_eq!(zd.shape.get(0).size(), 24); // 3*8 = tile size
    assert_eq!(zd.shape.get(1).size(), 12); // 3*4 = number of tiles

    // layout<0>(zipped_divide(a, b)) == composition(a, b)
    let zd_tile = zd.mode(0);
    let comp = composition_with_tile(&layout_a, &tiler);
    // Both should have same size (the tile)
    assert_eq!(zd_tile.size(), comp.size());

    // tiled_divide: ((TileM,TileN), RestM, RestN) with shape ((3,8), 3, 4)
    let td = tiled_divide(&layout_a, &tiler);
    assert_eq!(td.shape.get(0).size(), 24); // tile size
    assert_eq!(td.rank(), 3); // tile, RestM, RestN

    // flat_divide: (TileM, TileN, RestM, RestN) with shape (3, 8, 3, 4)
    let fd = flat_divide(&layout_a, &tiler);
    assert_eq!(fd.rank(), 4); // TileM, TileN, RestM, RestN
}

// product-tiling
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#product-tiling

#[test]
fn test_layout_algebra_product_tiling() {
    // 1-D Example
    // A = (2,2):(4,1), B = 6:1
    // A* = complement((2,2):(4,1), 24) = (2,3):(2,8)
    // Result = ((2,2),(2,3)):((4,1),(2,8))
    let a = Layout::new(int!(2, 2), Some(int!(4, 1)));
    let b = Layout::new(int!(6), Some(int!(1)));
    let result = logical_product(&a, &b);
    assert_eq!(result.to_string(), "((2,2),(2,3)):((4,1),(2,8))");

    // The first mode is the tile A
    assert_eq!(result.mode(0).to_string(), "(2,2):(4,1)");

    // The second mode iterates over 6 tiles
    assert_eq!(result.mode(1).size(), 6);

    // Second example with B = (4,2):(2,1) - 8 tiles in different order
    let b2 = Layout::new(int!(4, 2), Some(int!(2, 1)));
    let result2 = logical_product(&a, &b2);

    // First mode is still the tile A
    assert_eq!(result2.mode(0).to_string(), "(2,2):(4,1)");

    // Second mode iterates over 8 tiles
    assert_eq!(result2.mode(1).size(), 8);
}
