use kawaii::{
    coalesce, complement, composition, int, logical_divide, logical_product, IntTuple, Layout,
};

// Property tests for layout algebra
// Verify that operations compose correctly and preserve semantics

#[test]
fn test_divide_then_flatten_preserves_elements() {
    // Start with a layout
    let a = Layout::new(int!(4, 6), Some(int!(1, 4)));

    // Divide by a tiler
    let tiler = Layout::new(int!(2), Some(int!(2)));
    let divided = logical_divide(&a, &tiler);

    // Verify all original indices are still reachable
    let mut original_indices: Vec<i64> = (0..a.size()).map(|i| a.call_1d(i)).collect();
    let mut divided_indices: Vec<i64> = (0..divided.size()).map(|i| divided.call_1d(i)).collect();

    original_indices.sort();
    divided_indices.sort();

    assert_eq!(original_indices, divided_indices);
}

#[test]
fn test_product_then_divide_roundtrip() {
    // Start with a small tile
    let tile = Layout::new(int!(2, 2), Some(int!(1, 2)));

    // Create a product with 6 repetitions
    let reps = Layout::new(int!(6), Some(int!(1)));
    let product = logical_product(&tile, &reps);

    // The product should have size = tile.size * reps.size = 4 * 6 = 24
    assert_eq!(product.size(), 24);

    // First mode should be the original tile
    assert_eq!(product.mode(0).size(), tile.size());

    // Second mode should be the repetitions
    assert_eq!(product.mode(1).size(), reps.size());

    // Divide the product by the tile - should get back the repetition structure
    let divided = logical_divide(&product, &tile);

    // First mode of divided should match tile
    assert_eq!(divided.mode(0).size(), tile.size());
}

#[test]
fn test_composition_associativity() {
    // (A o B) o C == A o (B o C) when sizes are compatible
    let a = Layout::new(int!(12), Some(int!(1)));
    let b = Layout::new(int!(4), Some(int!(3)));
    let c = Layout::new(int!(2), Some(int!(2)));

    // (A o B) o C
    let ab = composition(&a, &b);
    let ab_c = composition(&ab, &c);

    // A o (B o C)
    let bc = composition(&b, &c);
    let a_bc = composition(&a, &bc);

    // Both should produce same indices
    for i in 0..ab_c.size() {
        assert_eq!(ab_c.call_1d(i), a_bc.call_1d(i), "Mismatch at index {}", i);
    }
}

#[test]
fn test_coalesce_preserves_function() {
    // Coalesce should not change the function, just simplify the representation
    let a = Layout::new(int!(2, int!(1, 6)), Some(int!(1, int!(6, 2))));
    let coalesced = coalesce(&a); // 12:1

    // Same size
    assert_eq!(a.size(), coalesced.size());

    // Same function values
    for i in 0..a.size() {
        assert_eq!(
            a.call_1d(i),
            coalesced.call_1d(i),
            "Mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_complement_fills_gaps() {
    // Layout A and its complement should together cover the cotarget
    let a = Layout::new(int!(4), Some(int!(2)));
    let cotarget = 24;
    let comp = complement(&a, Some(cotarget));

    // Use logical_product to combine A and its complement
    let combined = logical_product(&a, &comp);

    // Collect all indices from combined layout
    let mut indices: Vec<i64> = (0..combined.size()).map(|i| combined.call_1d(i)).collect();
    indices.sort();
    indices.dedup();

    // Should cover 0..cotarget
    assert_eq!(indices.len(), cotarget as usize);
}

#[test]
fn test_divide_preserves_all_indices() {
    // Dividing a layout should preserve all reachable indices
    let base = Layout::new(int!(6, 4), Some(int!(4, 1)));

    // Tile size
    let tile = Layout::new(int!(2), Some(int!(1)));

    // Divide
    let divided = logical_divide(&base, &tile);

    // Verify total size preserved
    assert_eq!(base.size(), divided.size());

    // Verify the divided layout reaches all the same indices
    let mut base_indices: Vec<i64> = (0..base.size()).map(|i| base.call_1d(i)).collect();
    let mut divided_indices: Vec<i64> = (0..divided.size()).map(|i| divided.call_1d(i)).collect();

    base_indices.sort();
    divided_indices.sort();

    assert_eq!(base_indices, divided_indices);
}

#[test]
fn test_composition_with_identity() {
    // Composing with identity (n:1) should preserve the layout function
    let a = Layout::new(int!(4, 3), Some(int!(3, 1)));
    let identity = Layout::new(int!(a.size()), Some(int!(1)));

    let result = composition(&a, &identity);

    // Same size
    assert_eq!(a.size(), result.size());

    // Same function
    for i in 0..a.size() {
        assert_eq!(a.call_1d(i), result.call_1d(i), "Mismatch at index {}", i);
    }
}
