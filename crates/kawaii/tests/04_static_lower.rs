use kawaii::{coalesce, int, Layout, LowerLayoutError, StaticLayout};

// The planning→execution handoff: layout algebra runs dynamically on the
// host, then the result is lowered to a Copy, heap-free StaticLayout that a
// kernel can take as a (monomorphized) parameter.

#[test]
fn lowered_layout_matches_dynamic_indexing() {
    let dynamic = Layout::new(int!(4, 6), Some(int!(6, 1))); // row-major 4x6
    let lowered: StaticLayout<2> = dynamic.to_static().unwrap();

    for i in 0..4 {
        for j in 0..6 {
            assert_eq!(
                dynamic.call(&int!(i as i64, j as i64)),
                lowered.index([i, j]) as i64
            );
        }
    }
    assert_eq!(dynamic.cosize(), lowered.cosize() as i64);
    assert_eq!(dynamic.size(), lowered.size() as i64);
}

#[test]
fn nested_layouts_flatten_before_lowering() {
    // ((2,2),4):((1,8),2) has flat rank 3
    let nested = Layout::new(
        int!(int!(2, 2), 4),
        Some(int!(int!(1, 8), 2)),
    );
    let lowered: StaticLayout<3> = nested.to_static().unwrap();
    let flat = nested.flatten();

    for i in 0..2 {
        for j in 0..2 {
            for k in 0..4 {
                assert_eq!(
                    flat.call(&int!(i as i64, j as i64, k as i64)),
                    lowered.index([i, j, k]) as i64
                );
            }
        }
    }
}

#[test]
fn rank_mismatch_is_rejected() {
    let layout = Layout::new(int!(4, 6), None);
    assert_eq!(
        layout.to_static::<3>(),
        Err(LowerLayoutError::RankMismatch {
            expected: 3,
            actual: 2
        })
    );
}

#[test]
fn negative_strides_are_rejected() {
    let layout = Layout::new(int!(4), Some(int!(-1)));
    assert_eq!(layout.to_static::<1>(), Err(LowerLayoutError::NegativeValue));
}

#[test]
fn static_layout_round_trips_to_dynamic() {
    let lowered = StaticLayout::<2>::col_major(3, 5);
    let dynamic: Layout = lowered.into();
    assert_eq!(dynamic.to_string(), "(3,5):(1,3)");
    assert_eq!(dynamic.to_static::<2>().unwrap(), lowered);
}

#[test]
fn coalesce_then_lower_to_smaller_rank() {
    // A dense nested layout coalesces to rank 1, then lowers to R=1:
    // the "check the contract, pick the fast path" flow in a planner.
    let nested = Layout::new(int!(int!(2, 4), 8), None);
    let coalesced = coalesce(&nested);
    let lowered: StaticLayout<1> = coalesced.to_static().unwrap();
    assert_eq!(lowered.shape.as_array(), [64]);
    assert_eq!(lowered.stride.as_array(), [1]);
}

#[test]
fn try_from_reference_works() {
    let layout = Layout::new(int!(8, 8), None);
    let lowered = StaticLayout::<2>::try_from(&layout).unwrap();
    assert_eq!(lowered.shape.as_array(), [8, 8]);
    assert_eq!(lowered.stride.as_array(), [1, 8]);
}
