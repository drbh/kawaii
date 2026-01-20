//! CuTe Layout Algebra - an implementation of NVIDIA CuTe's hierarchical layout system.
//!
//! Based on: https://github.com/NVIDIA/cutlass

use std::fmt::{self, Display};

// IntTuple - Recursive integer or tuple type

#[derive(Debug, Clone, PartialEq)]
pub enum IntTuple {
    Int(i64),
    Tuple(Vec<IntTuple>),
}

impl IntTuple {
    /// Number of top-level elements. Int has rank 1, tuple has rank = len.
    pub fn rank(&self) -> usize {
        match self {
            IntTuple::Int(_) => 1,
            IntTuple::Tuple(v) => v.len(),
        }
    }

    /// Maximum nesting depth. Int has depth 0.
    pub fn depth(&self) -> usize {
        match self {
            IntTuple::Int(_) => 0,
            IntTuple::Tuple(v) if v.is_empty() => 1,
            IntTuple::Tuple(v) => 1 + v.iter().map(|x| x.depth()).max().unwrap_or(0),
        }
    }

    /// Product of all elements (flattened). This is the coordinate space size.
    pub fn size(&self) -> i64 {
        match self {
            IntTuple::Int(n) => *n,
            IntTuple::Tuple(v) if v.is_empty() => 1,
            IntTuple::Tuple(v) => v.iter().map(|x| x.size()).product(),
        }
    }

    /// Flatten to a single-level vector of integers.
    pub fn flatten(&self) -> Vec<i64> {
        match self {
            IntTuple::Int(n) => vec![*n],
            IntTuple::Tuple(v) => v.iter().flat_map(|x| x.flatten()).collect(),
        }
    }

    /// Get element at index. Panics if out of bounds or indexing an Int.
    pub fn get(&self, i: usize) -> &IntTuple {
        match self {
            IntTuple::Int(_) => panic!("cannot index into Int"),
            IntTuple::Tuple(v) => &v[i],
        }
    }

    /// Check if this is an Int (not a Tuple).
    pub fn is_int(&self) -> bool {
        matches!(self, IntTuple::Int(_))
    }

    /// Extract the integer value. Panics if Tuple.
    pub fn as_int(&self) -> i64 {
        match self {
            IntTuple::Int(n) => *n,
            IntTuple::Tuple(_) => panic!("expected Int, got Tuple"),
        }
    }

    /// Wrap this IntTuple in a single-element tuple.
    /// Turns `(4,2)` into `((4,2))` for vector layouts.
    pub fn wrap(self) -> IntTuple {
        IntTuple::Tuple(vec![self])
    }
}

impl Display for IntTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntTuple::Int(n) => write!(f, "{}", n),
            IntTuple::Tuple(v) => {
                write!(f, "(")?;
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", x)?;
                }
                write!(f, ")")
            }
        }
    }
}

// Convenience constructors
impl From<i64> for IntTuple {
    fn from(n: i64) -> Self {
        IntTuple::Int(n)
    }
}

impl<T: Into<IntTuple>, const N: usize> From<[T; N]> for IntTuple {
    fn from(arr: [T; N]) -> Self {
        IntTuple::Tuple(arr.into_iter().map(|x| x.into()).collect())
    }
}

/// Construct an IntTuple from values. Use: `int!(2, 3)` or `int!(2, int!(3, 4))`.
#[macro_export]
macro_rules! int {
    ($e:expr) => { IntTuple::from($e) };
    ($($e:expr),+ $(,)?) => { IntTuple::Tuple(vec![$( int!($e) ),+]) };
}

/// Alias for int! to match CuTe's make_coord syntax.
#[macro_export]
macro_rules! make_coord {
    ($e:expr) => { IntTuple::from($e) };
    ($($e:expr),+ $(,)?) => { IntTuple::Tuple(vec![$( make_coord!($e) ),+]) };
}

// Layout - A (shape, stride) pair mapping coordinates to indices

#[derive(Debug, Clone, PartialEq)]
pub struct Layout {
    pub shape: IntTuple,
    pub stride: IntTuple,
}

impl Layout {
    /// Create a new layout. If stride is None, uses compact column-major.
    pub fn new(shape: impl Into<IntTuple>, stride: Option<IntTuple>) -> Self {
        let shape = shape.into();
        let stride = stride.unwrap_or_else(|| compact_col_major(&shape));
        Layout { shape, stride }
    }

    /// Map a coordinate to a linear index.
    pub fn call(&self, coord: &IntTuple) -> i64 {
        crd2idx(coord, &self.shape, &self.stride)
    }

    /// Map a 1D index (using colexicographical order) to a linear index.
    pub fn call_1d(&self, idx: i64) -> i64 {
        crd2idx(&IntTuple::Int(idx), &self.shape, &self.stride)
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn size(&self) -> i64 {
        self.shape.size()
    }

    /// Maximum index + 1 (the range of the layout function).
    pub fn cosize(&self) -> i64 {
        cosize_impl(&self.shape, &self.stride)
    }

    /// Get sublayout for mode i.
    pub fn mode(&self, i: usize) -> Layout {
        Layout {
            shape: self.shape.get(i).clone(),
            stride: self.stride.get(i).clone(),
        }
    }

    /// Get nested sublayout by following a path of indices.
    /// layout_at(&[1, 0]) is equivalent to CuTe's layout<1,0>(a).
    pub fn layout_at(&self, indices: &[usize]) -> Layout {
        let mut result = self.clone();
        for &i in indices {
            result = result.mode(i);
        }
        result
    }

    /// Select specific modes by index.
    /// select(&[1, 3]) on (2,3,5,7):(1,2,6,30) returns (3,7):(2,30).
    pub fn select(&self, indices: &[usize]) -> Layout {
        let shapes: Vec<IntTuple> = indices.iter().map(|&i| self.shape.get(i).clone()).collect();
        let strides: Vec<IntTuple> = indices
            .iter()
            .map(|&i| self.stride.get(i).clone())
            .collect();
        Layout {
            shape: IntTuple::Tuple(shapes),
            stride: IntTuple::Tuple(strides),
        }
    }

    /// Take a range of modes [begin, end).
    /// take(1, 3) on (2,3,5,7):(1,2,6,30) returns (3,5):(2,6).
    pub fn take(&self, begin: usize, end: usize) -> Layout {
        let indices: Vec<usize> = (begin..end).collect();
        self.select(&indices)
    }

    /// Append another layout as a new mode.
    /// append(3:1, 4:3) => (3,4):(1,3)
    pub fn append(&self, other: &Layout) -> Layout {
        let shape = match &self.shape {
            IntTuple::Int(_) => IntTuple::Tuple(vec![self.shape.clone(), other.shape.clone()]),
            IntTuple::Tuple(v) => {
                let mut shapes = v.clone();
                shapes.push(other.shape.clone());
                IntTuple::Tuple(shapes)
            }
        };
        let stride = match &self.stride {
            IntTuple::Int(_) => IntTuple::Tuple(vec![self.stride.clone(), other.stride.clone()]),
            IntTuple::Tuple(v) => {
                let mut strides = v.clone();
                strides.push(other.stride.clone());
                IntTuple::Tuple(strides)
            }
        };
        Layout { shape, stride }
    }

    /// Prepend another layout as a new first mode.
    /// prepend(3:1, 4:3) => (4,3):(3,1)
    pub fn prepend(&self, other: &Layout) -> Layout {
        let shape = match &self.shape {
            IntTuple::Int(_) => IntTuple::Tuple(vec![other.shape.clone(), self.shape.clone()]),
            IntTuple::Tuple(v) => {
                let mut shapes = vec![other.shape.clone()];
                shapes.extend(v.clone());
                IntTuple::Tuple(shapes)
            }
        };
        let stride = match &self.stride {
            IntTuple::Int(_) => IntTuple::Tuple(vec![other.stride.clone(), self.stride.clone()]),
            IntTuple::Tuple(v) => {
                let mut strides = vec![other.stride.clone()];
                strides.extend(v.clone());
                IntTuple::Tuple(strides)
            }
        };
        Layout { shape, stride }
    }

    /// Replace mode at index i with another layout.
    /// replace(2, (3,4,4):(1,3,3), 4:3) replaces mode 2
    pub fn replace(&self, i: usize, other: &Layout) -> Layout {
        match (&self.shape, &self.stride) {
            (IntTuple::Tuple(shapes), IntTuple::Tuple(strides)) => {
                let mut new_shapes = shapes.clone();
                let mut new_strides = strides.clone();
                new_shapes[i] = other.shape.clone();
                new_strides[i] = other.stride.clone();
                Layout {
                    shape: IntTuple::Tuple(new_shapes),
                    stride: IntTuple::Tuple(new_strides),
                }
            }
            _ => panic!("replace requires tuple layout"),
        }
    }

    /// Group modes [begin, end) into a nested tuple.
    /// group(0, 2) on (2,3,5,7):(1,2,6,30) => ((2,3),5,7):((1,2),6,30)
    pub fn group(&self, begin: usize, end: usize) -> Layout {
        match (&self.shape, &self.stride) {
            (IntTuple::Tuple(shapes), IntTuple::Tuple(strides)) => {
                let grouped_shape = IntTuple::Tuple(shapes[begin..end].to_vec());
                let grouped_stride = IntTuple::Tuple(strides[begin..end].to_vec());

                let mut new_shapes: Vec<IntTuple> = shapes[..begin].to_vec();
                new_shapes.push(grouped_shape);
                new_shapes.extend(shapes[end..].to_vec());

                let mut new_strides: Vec<IntTuple> = strides[..begin].to_vec();
                new_strides.push(grouped_stride);
                new_strides.extend(strides[end..].to_vec());

                Layout {
                    shape: IntTuple::Tuple(new_shapes),
                    stride: IntTuple::Tuple(new_strides),
                }
            }
            _ => panic!("group requires tuple layout"),
        }
    }

    /// Flatten all nested tuples into a single-level layout.
    /// flatten on ((2,3),5,7):((1,2),6,30) => (2,3,5,7):(1,2,6,30)
    pub fn flatten(&self) -> Layout {
        Layout {
            shape: IntTuple::Tuple(
                self.shape
                    .flatten()
                    .into_iter()
                    .map(IntTuple::Int)
                    .collect(),
            ),
            stride: IntTuple::Tuple(
                self.stride
                    .flatten()
                    .into_iter()
                    .map(IntTuple::Int)
                    .collect(),
            ),
        }
    }
}

/// Concatenate layouts into a new layout.
/// make_layout(a, b) => (a.shape, b.shape):(a.stride, b.stride)
pub fn make_layout(layouts: &[&Layout]) -> Layout {
    let shapes: Vec<IntTuple> = layouts.iter().map(|l| l.shape.clone()).collect();
    let strides: Vec<IntTuple> = layouts.iter().map(|l| l.stride.clone()).collect();
    Layout {
        shape: IntTuple::Tuple(shapes),
        stride: IntTuple::Tuple(strides),
    }
}

impl Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.shape, self.stride)
    }
}

// Tile - A tuple of Layouts for by-mode composition

/// A Tile is a collection of layouts applied per-mode during composition.
#[derive(Debug, Clone, PartialEq)]
pub struct Tile(pub Vec<Layout>);

impl Tile {
    pub fn new(layouts: Vec<Layout>) -> Self {
        Tile(layouts)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, i: usize) -> &Layout {
        &self.0[i]
    }
}

impl std::ops::Index<usize> for Tile {
    type Output = Layout;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

impl Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<")?;
        for (i, l) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", l)?;
        }
        write!(f, ">")
    }
}

/// Create a Tile from layouts for by-mode composition.
#[macro_export]
macro_rules! tile {
    ($($layout:expr),+ $(,)?) => {
        Tile::new(vec![$($layout),+])
    };
}

/// Combine multiple layouts into one with each input as a mode.
pub fn combine_layouts(layouts: &[Layout]) -> Layout {
    if layouts.is_empty() {
        return Layout::new(1i64, Some(IntTuple::Int(0)));
    }
    if layouts.len() == 1 {
        return layouts[0].clone();
    }
    Layout {
        shape: IntTuple::Tuple(layouts.iter().map(|l| l.shape.clone()).collect()),
        stride: IntTuple::Tuple(layouts.iter().map(|l| l.stride.clone()).collect()),
    }
}

fn cosize_impl(shape: &IntTuple, stride: &IntTuple) -> i64 {
    match (shape, stride) {
        (IntTuple::Int(0), _) => 0,
        (IntTuple::Int(s), IntTuple::Int(d)) => (s - 1) * d + 1,
        (IntTuple::Tuple(shapes), IntTuple::Tuple(strides)) => {
            if shapes.is_empty() {
                return 1;
            }
            let max_idx: i64 = shapes
                .iter()
                .zip(strides.iter())
                .map(|(s, d)| cosize_impl(s, d) - 1)
                .sum();
            max_idx + 1
        }
        _ => panic!("shape and stride must have matching structure"),
    }
}

// Coordinate Mapping

/// Trait for types that can be used as crd2idx input (integers or coordinates).
pub trait Crd2IdxInput {
    fn to_int_tuple(&self) -> IntTuple;
}

impl Crd2IdxInput for i64 {
    fn to_int_tuple(&self) -> IntTuple {
        IntTuple::Int(*self)
    }
}

impl Crd2IdxInput for IntTuple {
    fn to_int_tuple(&self) -> IntTuple {
        self.clone()
    }
}

impl Crd2IdxInput for &IntTuple {
    fn to_int_tuple(&self) -> IntTuple {
        (*self).clone()
    }
}

/// Map a coordinate to a linear index using shape and stride.
/// Accepts both integer indices and coordinate tuples.
/// Examples: crd2idx(16, shape, stride), crd2idx(make_coord!(1,5), shape, stride)
pub fn crd2idx(coord: impl Crd2IdxInput, shape: &IntTuple, stride: &IntTuple) -> i64 {
    crd2idx_impl(&coord.to_int_tuple(), shape, stride)
}

fn crd2idx_impl(coord: &IntTuple, shape: &IntTuple, stride: &IntTuple) -> i64 {
    match (coord, shape, stride) {
        (IntTuple::Int(c), IntTuple::Int(_), IntTuple::Int(d)) => c * d,

        (IntTuple::Int(c), IntTuple::Tuple(shapes), IntTuple::Tuple(strides)) => {
            let mut result = 0i64;
            let mut remaining = *c;
            for (s, d) in shapes.iter().zip(strides.iter()) {
                let mode_size = s.size();
                let mode_coord = remaining % mode_size;
                remaining /= mode_size;
                result += crd2idx_impl(&IntTuple::Int(mode_coord), s, d);
            }
            result
        }

        (IntTuple::Tuple(coords), IntTuple::Tuple(shapes), IntTuple::Tuple(strides)) => coords
            .iter()
            .zip(shapes.iter())
            .zip(strides.iter())
            .map(|((c, s), d)| crd2idx_impl(c, s, d))
            .sum(),

        _ => panic!("mismatched structures in crd2idx"),
    }
}

/// Trait for types that can be used as idx2crd input (integers or coordinates).
pub trait Idx2CrdInput {
    fn to_1d_index(&self, shape: &IntTuple, stride: &IntTuple) -> i64;
}

impl Idx2CrdInput for i64 {
    fn to_1d_index(&self, _shape: &IntTuple, _stride: &IntTuple) -> i64 {
        *self
    }
}

impl Idx2CrdInput for IntTuple {
    fn to_1d_index(&self, shape: &IntTuple, stride: &IntTuple) -> i64 {
        crd2idx(self, shape, stride)
    }
}

/// Map an index or coordinate to the natural coordinate using colexicographical ordering.
/// Accepts both integer indices and coordinate tuples.
/// Examples: idx2crd(16, shape), idx2crd(make_coord!(1,5), shape)
pub fn idx2crd(input: impl Idx2CrdInput, shape: &IntTuple) -> IntTuple {
    let stride = compact_col_major(shape);
    let idx = input.to_1d_index(shape, &stride);
    idx2crd_with_stride(idx, shape, &stride)
}

/// Map an index back to a coordinate with explicit stride.
pub fn idx2crd_with_stride(idx: i64, shape: &IntTuple, stride: &IntTuple) -> IntTuple {
    match (shape, stride) {
        (IntTuple::Int(1), _) => IntTuple::Int(0),
        (IntTuple::Int(s), IntTuple::Int(d)) => IntTuple::Int((idx / d) % s),
        (IntTuple::Tuple(shapes), IntTuple::Tuple(strides)) => IntTuple::Tuple(
            shapes
                .iter()
                .zip(strides.iter())
                .map(|(s, d)| idx2crd_with_stride(idx, s, d))
                .collect(),
        ),
        _ => panic!("mismatched structures in idx2crd"),
    }
}

// Stride Generation

/// Generate compact column-major (colexicographical) strides.
/// First mode has stride `current`, subsequent modes have stride = prev * prev_shape.
pub fn compact_col_major(shape: &IntTuple) -> IntTuple {
    compact_col_major_inner(shape, 1)
}

fn compact_col_major_inner(shape: &IntTuple, current: i64) -> IntTuple {
    match shape {
        IntTuple::Int(_) => IntTuple::Int(current),
        IntTuple::Tuple(shapes) => {
            let mut strides = Vec::with_capacity(shapes.len());
            let mut stride = current;
            for s in shapes {
                strides.push(compact_col_major_inner(s, stride));
                stride *= s.size();
            }
            IntTuple::Tuple(strides)
        }
    }
}

/// Generate compact row-major strides.
/// Last mode has stride 1, earlier modes have stride = later * later_shape.
pub fn compact_row_major(shape: &IntTuple) -> IntTuple {
    let flat = shape.flatten();
    let mut strides: Vec<i64> = Vec::with_capacity(flat.len());
    let mut stride = 1i64;
    for s in flat.iter().rev() {
        strides.push(stride);
        stride *= s;
    }
    strides.reverse();
    reshape_flat(&strides, shape)
}

fn reshape_flat(flat: &[i64], shape: &IntTuple) -> IntTuple {
    let mut idx = 0;
    reshape_flat_inner(flat, shape, &mut idx)
}

fn reshape_flat_inner(flat: &[i64], shape: &IntTuple, idx: &mut usize) -> IntTuple {
    match shape {
        IntTuple::Int(_) => {
            let val = flat[*idx];
            *idx += 1;
            IntTuple::Int(val)
        }
        IntTuple::Tuple(shapes) => IntTuple::Tuple(
            shapes
                .iter()
                .map(|s| reshape_flat_inner(flat, s, idx))
                .collect(),
        ),
    }
}

// Layout Operations

/// Simplify a layout by merging compatible adjacent modes.
/// Eliminates size-1 modes and merges where shape[i] * stride[i] == stride[i+1].
pub fn coalesce(layout: &Layout) -> Layout {
    let flat_shapes = layout.shape.flatten();
    let flat_strides = layout.stride.flatten();

    if flat_shapes.is_empty() {
        return Layout::new(1i64, Some(IntTuple::Int(0)));
    }

    let mut result_shapes: Vec<i64> = Vec::new();
    let mut result_strides: Vec<i64> = Vec::new();

    // Process from right to left
    for i in (0..flat_shapes.len()).rev() {
        let curr_shape = flat_shapes[i];
        let curr_stride = flat_strides[i];

        if curr_shape == 1 {
            continue;
        }

        if let Some(&prev_stride) = result_strides.first() {
            if curr_shape * curr_stride == prev_stride {
                // Merge with previous
                result_shapes[0] *= curr_shape;
                result_strides[0] = curr_stride;
            } else {
                result_shapes.insert(0, curr_shape);
                result_strides.insert(0, curr_stride);
            }
        } else {
            result_shapes.push(curr_shape);
            result_strides.push(curr_stride);
        }
    }

    if result_shapes.is_empty() {
        return Layout::new(1i64, Some(IntTuple::Int(0)));
    }
    if result_shapes.len() == 1 {
        return Layout::new(result_shapes[0], Some(IntTuple::Int(result_strides[0])));
    }

    Layout {
        shape: IntTuple::Tuple(result_shapes.into_iter().map(IntTuple::Int).collect()),
        stride: IntTuple::Tuple(result_strides.into_iter().map(IntTuple::Int).collect()),
    }
}

/// Coalesce by mode using a profile.
/// The profile specifies how to apply coalesce: integers trigger coalesce, tuples recurse.
/// coalesce_by_mode((2,(1,6)):(1,(6,2)), (1,1)) => (2,6):(1,2)
pub fn coalesce_by_mode(layout: &Layout, profile: &IntTuple) -> Layout {
    match profile {
        IntTuple::Int(_) => coalesce(layout),
        IntTuple::Tuple(profile_modes) => {
            let shapes: Vec<IntTuple> = profile_modes
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let sublayout = layout.mode(i);
                    coalesce_by_mode(&sublayout, p).shape
                })
                .collect();
            let strides: Vec<IntTuple> = profile_modes
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let sublayout = layout.mode(i);
                    coalesce_by_mode(&sublayout, p).stride
                })
                .collect();
            Layout {
                shape: IntTuple::Tuple(shapes),
                stride: IntTuple::Tuple(strides),
            }
        }
    }
}

/// Replace stride-0 modes with size-1 modes.
pub fn filter_zeros(layout: &Layout) -> Layout {
    fn filter_z(shape: &IntTuple, stride: &IntTuple) -> (IntTuple, IntTuple) {
        match (shape, stride) {
            (IntTuple::Int(_), IntTuple::Int(0)) => (IntTuple::Int(1), IntTuple::Int(0)),
            (IntTuple::Int(s), IntTuple::Int(d)) => (IntTuple::Int(*s), IntTuple::Int(*d)),
            (IntTuple::Tuple(shapes), IntTuple::Tuple(strides)) => {
                let (new_shapes, new_strides): (Vec<_>, Vec<_>) = shapes
                    .iter()
                    .zip(strides.iter())
                    .map(|(s, d)| filter_z(s, d))
                    .unzip();
                (IntTuple::Tuple(new_shapes), IntTuple::Tuple(new_strides))
            }
            _ => panic!("mismatched structures"),
        }
    }

    let (new_shape, new_stride) = filter_z(&layout.shape, &layout.stride);
    Layout {
        shape: new_shape,
        stride: new_stride,
    }
}

/// Filter zeros and coalesce.
pub fn filter_layout(layout: &Layout) -> Layout {
    coalesce(&filter_zeros(layout))
}

fn ceil_div(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

fn signum(x: i64) -> i64 {
    if x > 0 {
        1
    } else if x < 0 {
        -1
    } else {
        0
    }
}

/// Find the complement layout - elements not covered by the input layout.
pub fn complement(layout: &Layout, cotarget: Option<i64>) -> Layout {
    let cotarget = cotarget.unwrap_or_else(|| layout.cosize());
    let filtered = filter_layout(layout);

    if filtered.size() == 0 {
        return Layout::new(cotarget, Some(compact_col_major(&IntTuple::Int(cotarget))));
    }

    let flat_shapes = filtered.shape.flatten();
    let flat_strides = filtered.stride.flatten();

    if flat_strides.iter().all(|&s| s == 0) {
        return Layout::new(cotarget, Some(compact_col_major(&IntTuple::Int(cotarget))));
    }

    // Sort modes by stride (ascending)
    let mut modes: Vec<_> = flat_shapes.into_iter().zip(flat_strides).collect();
    modes.sort_by_key(|&(_, s)| s.abs());

    let mut result_shapes = Vec::new();
    let mut result_strides = Vec::new();
    let mut prev_end = 1i64;

    for (shape, stride) in modes {
        if stride == 0 {
            continue;
        }

        let abs_stride = stride.abs();
        if abs_stride > prev_end {
            let gap_size = abs_stride / prev_end;
            if gap_size > 1 {
                result_shapes.push(gap_size);
                result_strides.push(prev_end);
            }
        }

        prev_end = abs_stride * shape;
    }

    if prev_end < cotarget {
        let remaining = ceil_div(cotarget, prev_end);
        if remaining > 1 {
            result_shapes.push(remaining);
            result_strides.push(prev_end);
        }
    }

    if result_shapes.is_empty() {
        return Layout::new(1i64, Some(IntTuple::Int(0)));
    }
    if result_shapes.len() == 1 {
        return Layout::new(result_shapes[0], Some(IntTuple::Int(result_strides[0])));
    }

    Layout {
        shape: IntTuple::Tuple(result_shapes.into_iter().map(IntTuple::Int).collect()),
        stride: IntTuple::Tuple(result_strides.into_iter().map(IntTuple::Int).collect()),
    }
}

/// Functional composition of layouts: result(c) = lhs(rhs(c)).
pub fn composition(lhs: &Layout, rhs: &Layout) -> Layout {
    let lhs_flat = coalesce(lhs);

    // Handle tuple RHS by right-distributivity
    if let IntTuple::Tuple(shapes) = &rhs.shape {
        if let IntTuple::Tuple(strides) = &rhs.stride {
            let results: Vec<_> = shapes
                .iter()
                .zip(strides.iter())
                .map(|(s, d)| {
                    let sub_rhs = Layout {
                        shape: s.clone(),
                        stride: d.clone(),
                    };
                    composition(&lhs_flat, &sub_rhs)
                })
                .collect();

            let (new_shapes, new_strides): (Vec<_>, Vec<_>) =
                results.into_iter().map(|l| (l.shape, l.stride)).unzip();

            return Layout {
                shape: IntTuple::Tuple(new_shapes),
                stride: IntTuple::Tuple(new_strides),
            };
        }
    }

    // RHS is integral
    let rhs_shape = rhs.shape.as_int();
    let rhs_stride = rhs.stride.as_int();

    // Special case: stride-0
    if rhs_stride == 0 {
        return Layout::new(rhs_shape, Some(IntTuple::Int(0)));
    }

    // Special case: LHS is integral
    if lhs_flat.shape.is_int() {
        let lhs_stride = lhs_flat.stride.as_int();
        return Layout::new(rhs_shape, Some(IntTuple::Int(rhs_stride * lhs_stride)));
    }

    // General case: LHS is tuple, RHS is integral
    let lhs_shapes = lhs_flat.shape.flatten();
    let lhs_strides = lhs_flat.stride.flatten();

    let mut result_shapes = Vec::new();
    let mut result_strides = Vec::new();
    let mut rest_shape = rhs_shape;
    let mut rest_stride = rhs_stride;

    for i in 0..lhs_shapes.len() - 1 {
        let curr_shape = lhs_shapes[i];
        let curr_stride = lhs_strides[i];

        let next_shape = ceil_div(curr_shape, rest_stride.abs());
        let mut next_stride = ceil_div(rest_stride.abs(), curr_shape) * signum(rest_stride);
        if next_stride == 0 {
            next_stride = 1;
        }

        if next_shape <= 1 || rest_shape <= 1 {
            rest_stride = next_stride;
        } else {
            let new_shape = next_shape.min(rest_shape);
            result_shapes.push(new_shape);
            result_strides.push(rest_stride * curr_stride);
            rest_shape = ceil_div(rest_shape, new_shape);
            rest_stride = next_stride;
        }
    }

    // Handle last mode
    let last_stride = *lhs_strides.last().unwrap();
    if rest_shape > 1 {
        result_shapes.push(rest_shape);
        result_strides.push(rest_stride * last_stride);
    }

    if result_shapes.is_empty() {
        return Layout::new(1i64, Some(IntTuple::Int(0)));
    }
    if result_shapes.len() == 1 {
        return Layout::new(result_shapes[0], Some(IntTuple::Int(result_strides[0])));
    }

    Layout {
        shape: IntTuple::Tuple(result_shapes.into_iter().map(IntTuple::Int).collect()),
        stride: IntTuple::Tuple(result_strides.into_iter().map(IntTuple::Int).collect()),
    }
}

/// Compose a layout with a Tile (by-mode composition).
/// Each tile element is composed with the corresponding mode of the layout.
pub fn composition_with_tile(lhs: &Layout, tile: &Tile) -> Layout {
    if lhs.shape.is_int() {
        panic!("Cannot compose rank-1 layout with multi-element Tile");
    }

    let lhs_rank = lhs.shape.rank();
    let mut result_shapes = Vec::with_capacity(lhs_rank);
    let mut result_strides = Vec::with_capacity(lhs_rank);

    for i in 0..lhs_rank {
        let mode_layout = lhs.mode(i);
        let composed = if i < tile.len() {
            composition(&mode_layout, &tile[i])
        } else {
            mode_layout
        };
        result_shapes.push(composed.shape);
        result_strides.push(composed.stride);
    }

    Layout {
        shape: IntTuple::Tuple(result_shapes),
        stride: IntTuple::Tuple(result_strides),
    }
}

/// Divide a layout into tiles and remainder.
/// Result has rank 2: mode 0 is the tile, mode 1 is iterations over tiles.
pub fn logical_divide(layout: &Layout, tiler: &Layout) -> Layout {
    let coal_layout = coalesce(layout);
    let cotarget = if coal_layout.size() > 0 {
        coal_layout.size()
    } else {
        tiler.shape.size()
    };

    let comp = complement(tiler, Some(cotarget));

    // Build combined tiler: (tiler, complement)
    let combined = Layout {
        shape: IntTuple::Tuple(vec![tiler.shape.clone(), comp.shape]),
        stride: IntTuple::Tuple(vec![tiler.stride.clone(), comp.stride]),
    };

    composition(layout, &combined)
}

/// Replicate a block layout according to a tiler.
/// Result = (block, complement(block) o tiler).
pub fn logical_product(block: &Layout, tiler: &Layout) -> Layout {
    let block_comp = complement(block, Some(block.shape.size() * tiler.cosize()));
    let rest = composition(&block_comp, tiler);

    Layout {
        shape: IntTuple::Tuple(vec![block.shape.clone(), rest.shape]),
        stride: IntTuple::Tuple(vec![block.stride.clone(), rest.stride]),
    }
}

/// Divide a layout with a Tile (by-mode logical divide).
pub fn logical_divide_with_tile(layout: &Layout, tile: &Tile) -> Layout {
    if layout.shape.is_int() {
        panic!("Cannot use multi-element Tile with rank-1 layout");
    }

    let layout_rank = layout.shape.rank();
    let mut result_shapes = Vec::with_capacity(layout_rank);
    let mut result_strides = Vec::with_capacity(layout_rank);

    for i in 0..layout_rank {
        let mode_layout = layout.mode(i);
        let divided = if i < tile.len() {
            logical_divide(&mode_layout, &tile[i])
        } else {
            Layout {
                shape: IntTuple::Tuple(vec![mode_layout.shape.clone()]),
                stride: IntTuple::Tuple(vec![mode_layout.stride.clone()]),
            }
        };
        result_shapes.push(divided.shape);
        result_strides.push(divided.stride);
    }

    Layout {
        shape: IntTuple::Tuple(result_shapes),
        stride: IntTuple::Tuple(result_strides),
    }
}

/// Zipped divide: rearranges logical_divide result to ((TileM,TileN), (RestM,RestN,...))
/// Gathers subtiles into mode-0 and rest into mode-1.
pub fn zipped_divide(layout: &Layout, tile: &Tile) -> Layout {
    let ld = logical_divide_with_tile(layout, tile);

    // Extract tile parts (mode 0 of each mode) and rest parts (mode 1 of each mode)
    let mut tile_shapes = Vec::new();
    let mut tile_strides = Vec::new();
    let mut rest_shapes = Vec::new();
    let mut rest_strides = Vec::new();

    for i in 0..ld.rank() {
        let mode = ld.mode(i);
        if mode.rank() >= 2 {
            tile_shapes.push(mode.shape.get(0).clone());
            tile_strides.push(mode.stride.get(0).clone());
            rest_shapes.push(mode.shape.get(1).clone());
            rest_strides.push(mode.stride.get(1).clone());
        } else {
            tile_shapes.push(mode.shape.clone());
            tile_strides.push(mode.stride.clone());
        }
    }

    Layout {
        shape: IntTuple::Tuple(vec![
            IntTuple::Tuple(tile_shapes),
            IntTuple::Tuple(rest_shapes),
        ]),
        stride: IntTuple::Tuple(vec![
            IntTuple::Tuple(tile_strides),
            IntTuple::Tuple(rest_strides),
        ]),
    }
}

/// Tiled divide: rearranges logical_divide result to ((TileM,TileN), RestM, RestN, ...)
/// Gathers subtiles into mode-0, rest parts become separate modes.
pub fn tiled_divide(layout: &Layout, tile: &Tile) -> Layout {
    let ld = logical_divide_with_tile(layout, tile);

    let mut tile_shapes = Vec::new();
    let mut tile_strides = Vec::new();
    let mut rest_shapes = Vec::new();
    let mut rest_strides = Vec::new();

    for i in 0..ld.rank() {
        let mode = ld.mode(i);
        if mode.rank() >= 2 {
            tile_shapes.push(mode.shape.get(0).clone());
            tile_strides.push(mode.stride.get(0).clone());
            rest_shapes.push(mode.shape.get(1).clone());
            rest_strides.push(mode.stride.get(1).clone());
        } else {
            tile_shapes.push(mode.shape.clone());
            tile_strides.push(mode.stride.clone());
        }
    }

    let mut result_shapes = vec![IntTuple::Tuple(tile_shapes)];
    result_shapes.extend(rest_shapes);
    let mut result_strides = vec![IntTuple::Tuple(tile_strides)];
    result_strides.extend(rest_strides);

    Layout {
        shape: IntTuple::Tuple(result_shapes),
        stride: IntTuple::Tuple(result_strides),
    }
}

/// Flat divide: rearranges logical_divide result to (TileM, TileN, RestM, RestN, ...)
/// All parts become separate modes.
pub fn flat_divide(layout: &Layout, tile: &Tile) -> Layout {
    let ld = logical_divide_with_tile(layout, tile);

    let mut tile_shapes = Vec::new();
    let mut tile_strides = Vec::new();
    let mut rest_shapes = Vec::new();
    let mut rest_strides = Vec::new();

    for i in 0..ld.rank() {
        let mode = ld.mode(i);
        if mode.rank() >= 2 {
            tile_shapes.push(mode.shape.get(0).clone());
            tile_strides.push(mode.stride.get(0).clone());
            rest_shapes.push(mode.shape.get(1).clone());
            rest_strides.push(mode.stride.get(1).clone());
        } else {
            tile_shapes.push(mode.shape.clone());
            tile_strides.push(mode.stride.clone());
        }
    }

    let mut result_shapes = tile_shapes;
    result_shapes.extend(rest_shapes);
    let mut result_strides = tile_strides;
    result_strides.extend(rest_strides);

    Layout {
        shape: IntTuple::Tuple(result_shapes),
        stride: IntTuple::Tuple(result_strides),
    }
}

// Visualization

/// Create an ASCII table visualization of a 2D layout.
pub fn print_2d(layout: &Layout) -> String {
    let mut lines = vec![layout.to_string()];

    let (m_size, n_size) = if layout.shape.is_int() {
        (1, layout.shape.as_int())
    } else if layout.rank() == 1 {
        (1, layout.shape.size())
    } else {
        (layout.shape.get(0).size(), layout.shape.get(1).size())
    };

    let max_m = m_size.min(16) as usize;
    let max_n = n_size.min(16) as usize;

    let max_idx = layout.cosize();
    let idx_width = format!("{}", max_idx).len().max(1);

    // Column headers: "      0   1   2   3"
    let mut header = "   ".to_string();
    for n in 0..max_n {
        header.push_str(&format!("{:>width$}", n, width = idx_width + 3));
    }
    lines.push(header);

    // Separator: "    +---+---+---+---+"
    let sep = "    +".to_string()
        + &(0..max_n)
            .map(|_| format!("{}+", "-".repeat(idx_width + 2)))
            .collect::<String>();
    lines.push(sep.clone());

    // Rows
    for m in 0..max_m {
        let mut row = format!("{:>2}  |", m);
        for n in 0..max_n {
            let idx = if layout.shape.is_int() || layout.rank() == 1 {
                layout.call_1d(n as i64)
            } else {
                layout.call(&int!(m as i64, n as i64))
            };
            row.push_str(&format!("{:>width$} |", idx, width = idx_width + 1));
        }
        lines.push(row);
        lines.push(sep.clone());
    }

    // Remove trailing newline by joining without final separator
    lines.pop();
    lines.push(sep);
    lines.join("\n")
}

/// Print a 1D layout as a sequence of indices.
pub fn print_1d(layout: &Layout) -> String {
    let total = layout.size();
    let indices: Vec<i64> = if total > 64 {
        (0..16).map(|i| layout.call_1d(i)).collect()
    } else {
        (0..total).map(|i| layout.call_1d(i)).collect()
    };

    if total > 64 {
        format!(
            "{}\n[{}, ... ] (size={})",
            layout,
            indices
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            total
        )
    } else {
        format!(
            "{}\n[{}]",
            layout,
            indices
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}
