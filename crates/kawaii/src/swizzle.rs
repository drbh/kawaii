//! XOR-based index swizzles, after CuTe's `Swizzle<B, M, S>`.
//!
//! A swizzle permutes linear indices by XOR-ing a field of high bits into a
//! field of low bits. Applied to shared-memory offsets it spreads accesses
//! across banks (avoiding conflicts) without breaking the contiguity of the
//! low `M` bits, so vectorized loads stay legal.
//!
//! Bit picture of `apply(idx)`:
//!
//! ```text
//!            S >= 0                    (B = swizzled bits, M = kept base bits)
//!   idx  = 0bxxxYYYzzzzMMMM
//!                 |    ^
//!                 v    |            YYY field (B bits at M + S)
//!   out  = 0bxxxYYYwwwwMMMM         wwww = zzzz ^ YYY
//! ```
//!
//! Requires `|S| >= B` so the source and target fields do not overlap, which
//! makes the swizzle an involution: `apply(apply(x)) == x`.

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt::{self, Display};

use crate::{render_grid, IntTuple, Layout};

/// Runtime-parameterized swizzle for host-side planning and inspection.
///
/// `bits` (B) is the width of the swizzled field, `base` (M) the number of
/// low bits kept fixed, `shift` (S) the signed distance between the source
/// field and the bits it is XOR-ed into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Swizzle {
    bits: u32,
    base: u32,
    shift: i32,
}

impl Swizzle {
    /// Create a swizzle. Panics unless `|shift| >= bits` (fields must not
    /// overlap, otherwise the swizzle is not an involution) and the fields
    /// fit below the sign bit of an `i64` index.
    pub fn new(bits: u32, base: u32, shift: i32) -> Self {
        assert!(
            shift.unsigned_abs() >= bits,
            "swizzle requires |shift| >= bits so source and target fields do not overlap"
        );
        assert!(
            bits + base + shift.unsigned_abs() <= 63,
            "swizzle fields must fit below the sign bit of an i64 index"
        );
        Swizzle { bits, base, shift }
    }

    /// The identity swizzle.
    pub fn identity() -> Self {
        Swizzle {
            bits: 0,
            base: 0,
            shift: 0,
        }
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    pub fn base(&self) -> u32 {
        self.base
    }

    pub fn shift(&self) -> i32 {
        self.shift
    }

    /// Apply the swizzle to a non-negative index.
    pub fn apply(&self, idx: i64) -> i64 {
        debug_assert!(idx >= 0, "swizzle indices must be non-negative");
        let bit_msk = (1i64 << self.bits) - 1;
        let up = self.base + self.shift.max(0) as u32;
        let masked = idx & (bit_msk << up);
        if self.shift >= 0 {
            idx ^ (masked >> self.shift as u32)
        } else {
            idx ^ (masked << (-self.shift) as u32)
        }
    }
}

impl Display for Swizzle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sw<{},{},{}>", self.bits, self.base, self.shift)
    }
}

/// A layout whose output indices are post-transformed by a swizzle:
/// `call(coord) = swizzle(layout(coord))`.
#[derive(Debug, Clone, PartialEq)]
pub struct SwizzledLayout {
    pub layout: Layout,
    pub swizzle: Swizzle,
}

impl SwizzledLayout {
    pub fn new(layout: Layout, swizzle: Swizzle) -> Self {
        SwizzledLayout { layout, swizzle }
    }

    pub fn call(&self, coord: &IntTuple) -> i64 {
        self.swizzle.apply(self.layout.call(coord))
    }

    pub fn call_1d(&self, idx: i64) -> i64 {
        self.swizzle.apply(self.layout.call_1d(idx))
    }

    pub fn rank(&self) -> usize {
        self.layout.rank()
    }

    pub fn size(&self) -> i64 {
        self.layout.size()
    }

    /// Maximum output index + 1. Computed by evaluating the whole domain,
    /// so this is a planning-time helper, not device code.
    pub fn cosize(&self) -> i64 {
        (0..self.size())
            .map(|i| self.call_1d(i))
            .max()
            .map_or(0, |m| m + 1)
    }
}

impl Display for SwizzledLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} o {}", self.swizzle, self.layout)
    }
}

/// ASCII table of a swizzled 2D layout's indices, like [`crate::print_2d`].
pub fn print_2d_swizzled(swizzled: &SwizzledLayout) -> String {
    let layout = &swizzled.layout;
    let (m_size, n_size) = if layout.shape.is_int() {
        (1, layout.shape.as_int())
    } else if layout.rank() == 1 {
        (1, layout.shape.size())
    } else {
        (layout.shape.get(0).size(), layout.shape.get(1).size())
    };

    let rows = m_size.min(16) as usize;
    let cols = n_size.min(16) as usize;

    let cells: Vec<Vec<String>> = (0..rows)
        .map(|m| {
            (0..cols)
                .map(|n| {
                    let idx = if layout.shape.is_int() || layout.rank() == 1 {
                        swizzled.call_1d(n as i64)
                    } else {
                        swizzled.call(&IntTuple::Tuple(alloc::vec![
                            IntTuple::Int(m as i64),
                            IntTuple::Int(n as i64),
                        ]))
                    };
                    idx.to_string()
                })
                .collect()
        })
        .collect();

    render_grid(&format!("{}", swizzled), &cells)
}

/// Compile-time swizzle with the same semantics as [`Swizzle`], usable in
/// `const` contexts and device code. Zero-sized; all math is `const fn`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StaticSwizzle<const B: u32, const M: u32, const S: i32>;

impl<const B: u32, const M: u32, const S: i32> StaticSwizzle<B, M, S> {
    /// Evaluated at monomorphization time; rejects overlapping or
    /// out-of-range fields.
    const VALID: () = {
        assert!(
            S.unsigned_abs() >= B,
            "swizzle requires |S| >= B so source and target fields do not overlap"
        );
        assert!(
            B + M + S.unsigned_abs() <= 63,
            "swizzle fields must fit below the top bit of the index"
        );
    };

    pub const fn new() -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::VALID;
        StaticSwizzle
    }

    pub const fn apply(idx: usize) -> usize {
        #[allow(clippy::let_unit_value)]
        let _ = Self::VALID;
        // Shift amounts are widened to usize: device backends require the
        // shift amount's width to match the value's.
        let bit_msk = (1usize << (B as usize)) - 1;
        let up = M as usize + if S > 0 { S as usize } else { 0 };
        let masked = idx & (bit_msk << up);
        if S >= 0 {
            idx ^ (masked >> (S as usize))
        } else {
            idx ^ (masked << ((-S) as usize))
        }
    }
}
