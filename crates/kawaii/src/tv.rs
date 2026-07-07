//! Thread/value ownership layouts and partitioning helpers.
//!
//! A TV-layout is a rank-2 layout `((threads...), (values...)) -> offset`:
//! it answers not just "where is this element?" but "which thread owns it,
//! and which of that thread's values is it?". This is the same idea as
//! CuTe's TV-layouts used by `local_partition` / tiled copy atoms.

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt::{self, Display};

use crate::{
    composition, idx2crd_with_stride, make_layout, render_grid, right_inverse, zipped_divide,
    IntTuple, Layout, Tile,
};

/// A rank-2 layout mapping `(thread, value)` to a linear offset.
#[derive(Debug, Clone, PartialEq)]
pub struct TvLayout(pub Layout);

impl TvLayout {
    /// Wrap a rank-2 layout; mode 0 is threads, mode 1 is values.
    pub fn new(layout: Layout) -> Self {
        assert!(
            layout.rank() == 2,
            "TV layout must have rank 2: (threads, values)"
        );
        TvLayout(layout)
    }

    /// TV layout from separate thread and value arrangements:
    /// `call(t, v) = thr(t) + val(v)`.
    pub fn from_thr_val(thr: &Layout, val: &Layout) -> Self {
        Self::new(make_layout(&[thr, val]))
    }

    /// TV layout of partitioning `layout` into tiles the shape of the thread
    /// arrangement (CuTe's `local_partition`, but returning the full
    /// ownership map). `thr_layout` maps tile position -> thread id, so
    /// thread `t` owns the position where `thr_layout` evaluates to `t`;
    /// it must be a compact permutation (panics otherwise).
    pub fn from_partition(layout: &Layout, thr_layout: &Layout) -> Self {
        let (tile_mode, rest) = partition_modes(layout, thr_layout);
        let thread_mode = composition(&tile_mode, &right_inverse(thr_layout));
        Self::new(make_layout(&[&thread_mode, &rest]))
    }

    /// Number of threads.
    pub fn threads(&self) -> i64 {
        self.0.shape.get(0).size()
    }

    /// Number of values owned by each thread.
    pub fn values_per_thread(&self) -> i64 {
        self.0.shape.get(1).size()
    }

    /// Offset owned by `(thread, value)`.
    pub fn call(&self, thread: i64, value: i64) -> i64 {
        self.0
            .call(&IntTuple::Tuple(vec![
                IntTuple::Int(thread),
                IntTuple::Int(value),
            ]))
    }

    /// One thread's slice: its value layout and base offset.
    pub fn thread_slice(&self, thread: i64) -> (Layout, i64) {
        (self.0.mode(1), self.0.mode(0).call_1d(thread))
    }

    /// Which `(thread, value)` produces `offset`, if any. Scans the whole
    /// domain — a debugging aid, not device code.
    pub fn owner_of(&self, offset: i64) -> Option<(i64, i64)> {
        for t in 0..self.threads() {
            for v in 0..self.values_per_thread() {
                if self.call(t, v) == offset {
                    return Some((t, v));
                }
            }
        }
        None
    }
}

impl Display for TvLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Lift a rank-1 Int-shape layout like `16:1` to the single-mode tuple form
/// `((16)):((1))` that the divide operations require.
fn as_multi_mode(layout: &Layout) -> Layout {
    if layout.shape.is_int() {
        Layout {
            shape: layout.shape.clone().wrap(),
            stride: layout.stride.clone().wrap(),
        }
    } else {
        layout.clone()
    }
}

/// The tile mode and rest mode of dividing `layout` by the shape of
/// `thr_layout` (one tiler mode per thread mode; data modes beyond the
/// thread rank are left whole and land in the rest mode).
fn partition_modes(layout: &Layout, thr_layout: &Layout) -> (Layout, Layout) {
    let layout = as_multi_mode(layout);
    let thr_rank = thr_layout.rank();
    let layout_rank = layout.rank();
    assert!(
        thr_rank <= layout_rank,
        "thread arrangement has higher rank than the layout"
    );
    let tiler = Tile::new(
        (0..layout_rank)
            .map(|i| {
                if i >= thr_rank {
                    return Layout::new(1i64, None);
                }
                let shape = if thr_layout.shape.is_int() {
                    thr_layout.shape.clone()
                } else {
                    thr_layout.shape.get(i).clone()
                };
                Layout::new(shape, None)
            })
            .collect(),
    );
    let zd = zipped_divide(&layout, &tiler);
    (zd.mode(0), zd.mode(1))
}

/// Layout and base offset of the tile at `tile_coord` (colexicographical for
/// integer coordinates), like CuTe's `local_tile`.
pub fn local_tile(layout: &Layout, tile: &Tile, tile_coord: impl Into<IntTuple>) -> (Layout, i64) {
    let zd = zipped_divide(&as_multi_mode(layout), tile);
    let offset = zd.mode(1).call(&tile_coord.into());
    (zd.mode(0), offset)
}

/// Per-thread value layout and base offset when `layout` is partitioned among
/// threads, like CuTe's `local_partition`. `thr_layout` maps tile position ->
/// thread id (CuTe convention), so thread `t` sits at the position within
/// each tile where `thr_layout` evaluates to `t`.
pub fn local_partition(layout: &Layout, thr_layout: &Layout, thread_idx: i64) -> (Layout, i64) {
    let (tile_mode, rest) = partition_modes(layout, thr_layout);
    let coord = idx2crd_with_stride(thread_idx, &thr_layout.shape, &thr_layout.stride);
    let offset = tile_mode.call(&coord);
    (rest, offset)
}

/// ASCII map of which `T{t}V{v}` owns each cell of a `rows`×`cols`
/// column-major tile. CuTe `print_latex` analog for terminals.
pub fn print_tv(tv: &TvLayout, rows: usize, cols: usize) -> String {
    print_tv_with(tv, rows, cols, |idx| idx)
}

/// Like [`print_tv`], but post-transforms each offset first — e.g. pass a
/// [`crate::Swizzle`]'s `apply` to see the swizzled shared-memory picture.
pub fn print_tv_with(
    tv: &TvLayout,
    rows: usize,
    cols: usize,
    f: impl Fn(i64) -> i64,
) -> String {
    let total = rows * cols;
    let mut owners: Vec<Option<(i64, i64)>> = vec![None; total];
    let mut multiply_owned = 0usize;
    let mut out_of_range = 0usize;

    for t in 0..tv.threads() {
        for v in 0..tv.values_per_thread() {
            let idx = f(tv.call(t, v));
            if idx >= 0 && (idx as usize) < total {
                let slot = &mut owners[idx as usize];
                if slot.is_some() {
                    multiply_owned += 1;
                } else {
                    *slot = Some((t, v));
                }
            } else {
                out_of_range += 1;
            }
        }
    }
    let unowned = owners.iter().filter(|o| o.is_none()).count();

    let title = format!(
        "(T{},V{}) {}",
        tv.threads(),
        tv.values_per_thread(),
        tv.0
    );
    let cells: Vec<Vec<String>> = (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| match owners[r + c * rows] {
                    Some((t, v)) => format!("T{}V{}", t, v),
                    None => ".".to_string(),
                })
                .collect()
        })
        .collect();

    let mut out = render_grid(&title, &cells);
    if multiply_owned > 0 {
        out.push_str(&format!(
            "\n{} cell(s) multiply-owned (first owner shown)",
            multiply_owned
        ));
    }
    if unowned > 0 {
        out.push_str(&format!("\n{} cell(s) unowned", unowned));
    }
    if out_of_range > 0 {
        out.push_str(&format!(
            "\n{} mapping(s) fall outside the {}x{} grid",
            out_of_range, rows, cols
        ));
    }
    out
}
