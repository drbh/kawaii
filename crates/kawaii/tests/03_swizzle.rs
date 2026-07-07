use kawaii::{int, print_2d_swizzled, Layout, StaticSwizzle, Swizzle, SwizzledLayout};

#[test]
fn swizzle_is_an_involution() {
    for &(b, m, s) in &[(3u32, 0u32, 3i32), (2, 0, 2), (3, 4, 3), (2, 3, -2), (0, 0, 0)] {
        let sw = Swizzle::new(b, m, s);
        for idx in 0..1024i64 {
            assert_eq!(sw.apply(sw.apply(idx)), idx, "Sw<{},{},{}> at {}", b, m, s, idx);
        }
    }
}

#[test]
fn swizzle_keeps_base_bits_fixed() {
    // The low M bits are untouched, so vectorized access within 2^M stays contiguous.
    let sw = Swizzle::new(3, 4, 3);
    for idx in 0..4096i64 {
        assert_eq!(sw.apply(idx) & 0xF, idx & 0xF);
    }
}

#[test]
fn identity_swizzle_is_identity() {
    let sw = Swizzle::identity();
    for idx in 0..256i64 {
        assert_eq!(sw.apply(idx), idx);
    }
}

#[test]
#[should_panic]
fn overlapping_fields_are_rejected() {
    // |shift| < bits would break the involution property.
    let _ = Swizzle::new(3, 0, 2);
}

#[test]
#[should_panic]
fn fields_past_the_sign_bit_are_rejected() {
    // base 61 + shift 3 + bits 3 would shift the mask past bit 63.
    let _ = Swizzle::new(3, 61, 3);
}

#[test]
fn static_swizzle_matches_dynamic() {
    let dynamic = Swizzle::new(3, 0, 3);
    for idx in 0..1024usize {
        assert_eq!(
            StaticSwizzle::<3, 0, 3>::apply(idx),
            dynamic.apply(idx as i64) as usize
        );
    }

    let dynamic = Swizzle::new(2, 3, -2);
    for idx in 0..1024usize {
        assert_eq!(
            StaticSwizzle::<2, 3, -2>::apply(idx),
            dynamic.apply(idx as i64) as usize
        );
    }
}

#[test]
fn static_swizzle_is_const_evaluable() {
    const SWIZZLED: usize = StaticSwizzle::<2, 0, 2>::apply(5);
    // 5 = 0b0101: field bits (0b0100) xor into low bits: 0b0101 ^ 0b0001 = 0b0100
    assert_eq!(SWIZZLED, 4);
}

#[test]
fn swizzle_resolves_bank_conflicts_for_8x8_column_access() {
    // 8x8 row-major f32 tile in shared memory: without swizzling, a column
    // access hits offsets {c, 8+c, 16+c, ...} — all congruent mod 8, i.e. an
    // 8-way bank conflict (with 8 "banks" for this reduced example).
    // Swizzle<3,0,3> XORs the row bits into the column bits, making the 8
    // offsets of every column hit 8 distinct banks.
    let smem = Layout::new(int!(8, 8), Some(int!(8, 1))); // row-major
    let swizzled = SwizzledLayout::new(smem, Swizzle::new(3, 0, 3));

    for col in 0..8i64 {
        let mut banks: Vec<i64> = (0..8i64)
            .map(|row| swizzled.call(&int!(row, col)) % 8)
            .collect();
        banks.sort();
        assert_eq!(banks, vec![0, 1, 2, 3, 4, 5, 6, 7], "column {}", col);
    }

    // And it is still a bijection on the tile.
    let mut all: Vec<i64> = (0..64).map(|i| swizzled.call_1d(i)).collect();
    all.sort();
    assert_eq!(all, (0..64).collect::<Vec<_>>());
}

#[test]
fn swizzled_layout_prints() {
    let smem = Layout::new(int!(4, 4), Some(int!(4, 1)));
    let swizzled = SwizzledLayout::new(smem, Swizzle::new(2, 0, 2));
    let output = print_2d_swizzled(&swizzled);
    let expected = r#"Sw<2,0,2> o (4,4):(4,1)
       0    1    2    3
    +----+----+----+----+
 0  |  0 |  1 |  2 |  3 |
    +----+----+----+----+
 1  |  5 |  4 |  7 |  6 |
    +----+----+----+----+
 2  | 10 | 11 |  8 |  9 |
    +----+----+----+----+
 3  | 15 | 14 | 13 | 12 |
    +----+----+----+----+"#;
    assert_eq!(output, expected);
}
