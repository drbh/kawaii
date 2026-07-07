use std::collections::BTreeSet;

use kawaii::{
    int, local_partition, local_tile, print_tv, right_inverse, tile, Layout,
    TvLayout,
};

fn covered_indices(value_layout: &Layout, offset: i64) -> Vec<i64> {
    (0..value_layout.size())
        .map(|k| offset + value_layout.call_1d(k))
        .collect()
}

#[test]
fn from_thr_val_is_additive() {
    // thread t at stride 1, value v at stride 4: call(t, v) = t + 4v
    let tv = TvLayout::from_thr_val(
        &Layout::new(int!(4), Some(int!(1))),
        &Layout::new(int!(2), Some(int!(4))),
    );
    assert_eq!(tv.threads(), 4);
    assert_eq!(tv.values_per_thread(), 2);
    for t in 0..4 {
        for v in 0..2 {
            assert_eq!(tv.call(t, v), t + 4 * v);
        }
    }
}

#[test]
fn thread_slice_matches_call() {
    let tv = TvLayout::from_thr_val(
        &Layout::new(int!(8), Some(int!(4))),
        &Layout::new(int!(4), Some(int!(1))),
    );
    for t in 0..8 {
        let (values, offset) = tv.thread_slice(t);
        for v in 0..4 {
            assert_eq!(offset + values.call_1d(v), tv.call(t, v));
        }
    }
}

#[test]
fn local_partition_covers_layout_disjointly() {
    // 4x8 column-major data, 2x2 threads (row-major within the tile).
    let data = Layout::new(int!(4, 8), None);
    let threads = Layout::new(int!(2, 2), Some(int!(2, 1)));

    let mut seen = BTreeSet::new();
    for t in 0..4 {
        let (values, offset) = local_partition(&data, &threads, t);
        assert_eq!(values.size(), 8); // 32 elements / 4 threads
        for idx in covered_indices(&values, offset) {
            assert!(seen.insert(idx), "index {} owned twice", idx);
        }
    }
    assert_eq!(seen, (0..32).collect::<BTreeSet<i64>>());
}

#[test]
fn local_partition_respects_thread_arrangement() {
    // Column-major 4x4 data tiled by 2x2 threads. Thread 1 sits at position
    // threads(1) within each tile.
    let data = Layout::new(int!(4, 4), None);

    // Column-major threads: thread 1 is at (1, 0) of the tile -> offset 1.
    let col_threads = Layout::new(int!(2, 2), None);
    let (_, offset) = local_partition(&data, &col_threads, 1);
    assert_eq!(offset, 1);

    // Row-major threads: thread 1 is at (0, 1) of the tile -> offset 4.
    let row_threads = Layout::new(int!(2, 2), Some(int!(2, 1)));
    let (_, offset) = local_partition(&data, &row_threads, 1);
    assert_eq!(offset, 4);
}

#[test]
fn from_partition_matches_local_partition() {
    let data = Layout::new(int!(8, 8), None);
    let threads = Layout::new(int!(4, 2), Some(int!(2, 1)));

    let tv = TvLayout::from_partition(&data, &threads);
    assert_eq!(tv.threads(), 8);

    for t in 0..8 {
        let (values, offset) = local_partition(&data, &threads, t);
        let (tv_values, tv_offset) = tv.thread_slice(t);
        assert_eq!(offset, tv_offset, "thread {}", t);
        assert_eq!(
            covered_indices(&values, offset),
            covered_indices(&tv_values, tv_offset),
            "thread {}",
            t
        );
    }
}

#[test]
fn local_tile_extracts_tile_at_coord() {
    // 8x8 column-major divided into 4x4 tiles: 2x2 grid of tiles.
    let data = Layout::new(int!(8, 8), None);
    let tiler = tile!(
        Layout::new(int!(4), Some(int!(1))),
        Layout::new(int!(4), Some(int!(1)))
    );

    // Tile (0,0) starts at offset 0; tile (1,0) at row 4 -> offset 4;
    // tile (0,1) at column 4 -> offset 32.
    let (tile00, offset00) = local_tile(&data, &tiler, int!(0, 0));
    assert_eq!(offset00, 0);
    assert_eq!(tile00.size(), 16);

    let (_, offset10) = local_tile(&data, &tiler, int!(1, 0));
    assert_eq!(offset10, 4);

    let (_, offset01) = local_tile(&data, &tiler, int!(0, 1));
    assert_eq!(offset01, 32);

    // All four tiles together cover the layout exactly once.
    let mut seen = BTreeSet::new();
    for i in 0..2 {
        for j in 0..2 {
            let (tile_layout, offset) = local_tile(&data, &tiler, int!(i, j));
            for idx in covered_indices(&tile_layout, offset) {
                assert!(seen.insert(idx), "index {} covered twice", idx);
            }
        }
    }
    assert_eq!(seen, (0..64).collect::<BTreeSet<i64>>());
}

#[test]
fn right_inverse_inverts_permutation_layouts() {
    for layout in [
        Layout::new(int!(16), None),
        Layout::new(int!(4, 4), None),                     // col-major
        Layout::new(int!(4, 4), Some(int!(4, 1))),         // row-major
        Layout::new(int!(3, 2), Some(int!(2, 1))),         // non-involutive
        Layout::new(int!(2, 3, 4), Some(int!(12, 4, 1))),  // 3D row-major
        Layout::new(int!(2, 3, 4), Some(int!(3, 1, 6))),   // mixed permutation
    ] {
        let inv = right_inverse(&layout);
        for o in 0..layout.size() {
            assert_eq!(layout.call_1d(inv.call_1d(o)), o, "layout {}", layout);
        }
    }
}

#[test]
#[should_panic]
fn right_inverse_rejects_non_permutations() {
    // (4):(2) leaves gaps in [0, 8) — no right inverse exists.
    let _ = right_inverse(&Layout::new(int!(4), Some(int!(2))));
}

#[test]
fn local_partition_inverts_thread_layout_like_cute() {
    // Non-involutive thread layout (3,2):(2,1) over 6x2 col-major data.
    // CuTe semantics: thr_layout maps tile position -> thread id, so thread t
    // sits at the position where thr_layout evaluates to t:
    //   thr(c0, c1) = 2*c0 + c1, thread 1 -> (c0, c1) = (0, 1) -> offset 6.
    // The forward map would (wrongly) put thread 1 at coord (1, 0) = offset 1.
    let data = Layout::new(int!(6, 2), None);
    let threads = Layout::new(int!(3, 2), Some(int!(2, 1)));

    let expected_offsets = [0i64, 6, 1, 7, 2, 8]; // thread t at (t/2, t%2)
    for (t, &expected) in expected_offsets.iter().enumerate() {
        let (_, offset) = local_partition(&data, &threads, t as i64);
        assert_eq!(offset, expected, "thread {}", t);
    }

    let mut seen = BTreeSet::new();
    for t in 0..6 {
        let (values, offset) = local_partition(&data, &threads, t);
        for idx in covered_indices(&values, offset) {
            assert!(seen.insert(idx), "index {} owned twice", idx);
        }
    }
    assert_eq!(seen, (0..12).collect::<BTreeSet<i64>>());
}

#[test]
fn from_partition_matches_local_partition_without_power_of_two() {
    // Divisibility-sensitive case: composition with the raw (non-inverted)
    // thread layout used to corrupt this ownership map.
    let data = Layout::new(int!(6, 2), None);
    let threads = Layout::new(int!(3, 2), Some(int!(2, 1)));

    let tv = TvLayout::from_partition(&data, &threads);
    assert_eq!(tv.threads(), 6);

    let mut seen = BTreeSet::new();
    for t in 0..6 {
        let (values, offset) = local_partition(&data, &threads, t);
        let (tv_values, tv_offset) = tv.thread_slice(t);
        assert_eq!(offset, tv_offset, "thread {}", t);
        assert_eq!(
            covered_indices(&values, offset),
            covered_indices(&tv_values, tv_offset),
            "thread {}",
            t
        );
        for idx in covered_indices(&tv_values, tv_offset) {
            assert!(seen.insert(idx), "index {} owned twice", idx);
        }
    }
    assert_eq!(seen, (0..12).collect::<BTreeSet<i64>>());
}

#[test]
#[should_panic]
fn from_partition_rejects_non_permutation_thread_layouts() {
    // Stride 3 over shape 4 is not a compact permutation of [0, 4).
    let data = Layout::new(int!(4, 4), None);
    let threads = Layout::new(int!(2, 2), Some(int!(3, 1)));
    let _ = TvLayout::from_partition(&data, &threads);
}

#[test]
fn rank1_data_can_be_tiled_and_partitioned() {
    // Partitioning a plain vector is the most basic use case.
    let data = Layout::new(int!(16), None);

    let (tile_layout, offset) = local_tile(&data, &tile!(Layout::new(int!(4), None)), int!(1));
    assert_eq!(tile_layout.size(), 4);
    assert_eq!(offset, 4);

    let threads = Layout::new(int!(4), None);
    let mut seen = BTreeSet::new();
    for t in 0..4 {
        let (values, offset) = local_partition(&data, &threads, t);
        assert_eq!(values.size(), 4);
        for idx in covered_indices(&values, offset) {
            assert!(seen.insert(idx), "index {} owned twice", idx);
        }
    }
    assert_eq!(seen, (0..16).collect::<BTreeSet<i64>>());
}

#[test]
fn lower_rank_threads_leave_untiled_modes_to_each_thread() {
    // 4 threads over 4x8 data: each thread owns a whole 8-element row.
    let data = Layout::new(int!(4, 8), None);
    let threads = Layout::new(int!(4), None);

    let mut seen = BTreeSet::new();
    for t in 0..4 {
        let (values, offset) = local_partition(&data, &threads, t);
        assert_eq!(values.size(), 8, "thread {}", t);
        assert_eq!(offset, t);
        for idx in covered_indices(&values, offset) {
            assert!(seen.insert(idx), "index {} owned twice", idx);
        }
    }
    assert_eq!(seen, (0..32).collect::<BTreeSet<i64>>());
}

#[test]
fn owner_of_inverts_call() {
    let data = Layout::new(int!(4, 4), None);
    let threads = Layout::new(int!(2, 2), None);
    let tv = TvLayout::from_partition(&data, &threads);

    for t in 0..tv.threads() {
        for v in 0..tv.values_per_thread() {
            let idx = tv.call(t, v);
            assert_eq!(tv.owner_of(idx), Some((t, v)));
        }
    }
    assert_eq!(tv.owner_of(999), None);
}

#[test]
fn print_tv_shows_ownership() {
    // 2 threads, 2 values each over a 2x2 column-major tile:
    // call(t, v) = t + 2v, so column-major cells are T0V0, T1V0, T0V1, T1V1.
    let tv = TvLayout::from_thr_val(
        &Layout::new(int!(2), Some(int!(1))),
        &Layout::new(int!(2), Some(int!(2))),
    );
    let output = print_tv(&tv, 2, 2);
    let expected = r#"(T2,V2) (2,2):(1,2)
         0      1
    +------+------+
 0  | T0V0 | T0V1 |
    +------+------+
 1  | T1V0 | T1V1 |
    +------+------+"#;
    assert_eq!(output, expected);
}

#[test]
fn print_tv_reports_unowned_cells() {
    // One thread, one value on a 2x2 grid: 3 cells unowned.
    let tv = TvLayout::from_thr_val(
        &Layout::new(int!(1), Some(int!(1))),
        &Layout::new(int!(1), Some(int!(1))),
    );
    let output = print_tv(&tv, 2, 2);
    assert!(output.contains("3 cell(s) unowned"), "{}", output);
}
