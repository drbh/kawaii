//! Heap-free layout types for device/embedded use.
//!
//! These types use fixed-size arrays instead of `Vec`, making them
//! `Copy`, `repr(C)`, and usable without an allocator. Do your layout
//! algebra on the host with [`crate::Layout`], then call
//! [`crate::Layout::to_flat`] to get a [`FlatLayout`] you can pass to
//! device code.

/// Maximum number of leaf elements in a flat tuple.
pub const MAX_RANK: usize = 8;

/// A fixed-size, heap-free integer tuple storing flattened leaf values.
///
/// Created by flattening an [`crate::IntTuple`] — all hierarchical
/// nesting is collapsed into a single level of `i64` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FlatIntTuple {
    values: [i64; MAX_RANK],
    len: u8,
}

impl FlatIntTuple {
    /// Build from a slice of values. Panics if `values.len() > MAX_RANK`.
    pub const fn new(values: &[i64]) -> Self {
        assert!(values.len() <= MAX_RANK, "too many elements for FlatIntTuple");
        let mut v = [0i64; MAX_RANK];
        let mut i = 0;
        while i < values.len() {
            v[i] = values[i];
            i += 1;
        }
        Self {
            values: v,
            len: values.len() as u8,
        }
    }

    pub const fn rank(&self) -> usize {
        self.len as usize
    }

    pub const fn get(&self, i: usize) -> i64 {
        self.values[i]
    }

    /// Product of all elements.
    pub const fn size(&self) -> i64 {
        let mut product = 1i64;
        let mut i = 0;
        while i < self.len as usize {
            product *= self.values[i];
            i += 1;
        }
        product
    }

    pub const fn as_array(&self) -> &[i64; MAX_RANK] {
        &self.values
    }

    pub const fn len(&self) -> usize {
        self.len as usize
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A heap-free layout: a pair of flat shape and stride arrays.
///
/// All methods are branch-free or loop-only over the fixed-size arrays,
/// making this suitable for GPU kernels and `#![no_std]` without `alloc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FlatLayout {
    shape: FlatIntTuple,
    stride: FlatIntTuple,
}

impl FlatLayout {
    pub const fn new(shape: FlatIntTuple, stride: FlatIntTuple) -> Self {
        Self { shape, stride }
    }

    /// Row-major 2D layout: stride = `[cols, 1]`.
    pub const fn row_major(rows: i64, cols: i64) -> Self {
        Self {
            shape: FlatIntTuple {
                values: [rows, cols, 0, 0, 0, 0, 0, 0],
                len: 2,
            },
            stride: FlatIntTuple {
                values: [cols, 1, 0, 0, 0, 0, 0, 0],
                len: 2,
            },
        }
    }

    /// Column-major 2D layout: stride = `[1, rows]`.
    pub const fn col_major(rows: i64, cols: i64) -> Self {
        Self {
            shape: FlatIntTuple {
                values: [rows, cols, 0, 0, 0, 0, 0, 0],
                len: 2,
            },
            stride: FlatIntTuple {
                values: [1, rows, 0, 0, 0, 0, 0, 0],
                len: 2,
            },
        }
    }

    pub const fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub const fn size(&self) -> i64 {
        self.shape.size()
    }

    pub const fn shape(&self) -> &FlatIntTuple {
        &self.shape
    }

    pub const fn stride(&self) -> &FlatIntTuple {
        &self.stride
    }

    /// Compute linear index from a flat coordinate slice.
    pub const fn call(&self, coords: &[i64]) -> i64 {
        let mut result = 0i64;
        let mut i = 0;
        let len = if coords.len() < self.stride.len as usize {
            coords.len()
        } else {
            self.stride.len as usize
        };
        while i < len {
            result += coords[i] * self.stride.values[i];
            i += 1;
        }
        result
    }

    /// Compute linear index from a 1D index via colexicographic decomposition.
    pub const fn call_1d(&self, idx: i64) -> i64 {
        let mut result = 0i64;
        let mut remaining = idx;
        let mut i = 0;
        while i < self.shape.len as usize {
            let s = self.shape.values[i];
            let d = self.stride.values[i];
            result += (remaining % s) * d;
            remaining /= s;
            i += 1;
        }
        result
    }

    /// Index with `(row, col)` — direct multiply-add, no decomposition.
    pub const fn index(&self, row: i64, col: i64) -> i64 {
        row * self.stride.values[0] + col * self.stride.values[1]
    }

    /// Bounds check for `(row, col)`.
    pub const fn contains(&self, row: i64, col: i64) -> bool {
        row >= 0 && col >= 0 && row < self.shape.values[0] && col < self.shape.values[1]
    }

    /// Maximum index + 1 (the range of the layout function).
    pub const fn cosize(&self) -> i64 {
        if self.shape.len == 0 {
            return 1;
        }
        let mut max_idx = 0i64;
        let mut i = 0;
        while i < self.shape.len as usize {
            let s = self.shape.values[i];
            let d = self.stride.values[i];
            if s > 0 {
                max_idx += (s - 1) * d;
            }
            i += 1;
        }
        max_idx + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_row_major() {
        let layout = FlatLayout::row_major(3, 4);
        assert_eq!(layout.index(2, 1), 9);
        assert_eq!(layout.cosize(), 12);
        assert!(layout.contains(2, 3));
        assert!(!layout.contains(3, 0));
    }

    #[test]
    fn flat_col_major() {
        let layout = FlatLayout::col_major(3, 4);
        assert_eq!(layout.index(2, 1), 5);
        assert_eq!(layout.cosize(), 12);
    }

    #[test]
    fn flat_call_1d() {
        let layout = FlatLayout::row_major(3, 4);
        // 1D index 5 in a (3,4) row-major layout:
        // coords = (5 % 3, 5 / 3) = (2, 1) in colexicographic order
        // linear = 2 * 4 + 1 * 1 = 9
        assert_eq!(layout.call_1d(5), 9);
    }

    #[test]
    fn flat_call_coords() {
        let layout = FlatLayout::row_major(3, 4);
        assert_eq!(layout.call(&[2, 1]), 9);
        assert_eq!(layout.call(&[0, 0]), 0);
        assert_eq!(layout.call(&[0, 3]), 3);
    }

    #[test]
    fn flat_is_copy_and_repr_c() {
        let a = FlatLayout::row_major(4, 8);
        let b = a; // Copy
        assert_eq!(a, b);
        assert_eq!(core::mem::size_of::<FlatLayout>(), core::mem::size_of::<FlatIntTuple>() * 2);
    }

    #[test]
    fn flat_const_constructible() {
        const LAYOUT: FlatLayout = FlatLayout::row_major(16, 16);
        assert_eq!(LAYOUT.size(), 256);
        assert_eq!(LAYOUT.cosize(), 256);
    }

    #[test]
    fn flat_from_slice() {
        let shape = FlatIntTuple::new(&[2, 3, 5]);
        let stride = FlatIntTuple::new(&[15, 5, 1]);
        let layout = FlatLayout::new(shape, stride);
        assert_eq!(layout.rank(), 3);
        assert_eq!(layout.size(), 30);
        assert_eq!(layout.call(&[1, 2, 3]), 28); // 1*15 + 2*5 + 3*1
    }
}
