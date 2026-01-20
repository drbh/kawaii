use std::vec;

use kawaii::{compact_row_major, crd2idx, idx2crd, int, make_layout, print_2d, IntTuple, Layout};

// using-a-layout
// http://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#using-a-layout
#[test]
fn test_layouts() {
    let s8 = Layout::new(8i64, None);
    assert_eq!(s8.to_string(), "8:1");

    let d8 = Layout::new(8i64, None);
    assert_eq!(d8.to_string(), "8:1");

    let s2xs4 = Layout::new(int!(2, 4), None);
    assert_eq!(s2xs4.to_string(), "(2,4):(1,2)");

    let s2xd4 = Layout::new(int!(2, 4), None);
    assert_eq!(s2xd4.to_string(), "(2,4):(1,2)");

    let s2xd4_a = Layout::new(int!(2, 4), Some(int!(12, 1)));
    assert_eq!(s2xd4_a.to_string(), "(2,4):(12,1)");

    let s2xd4_col = Layout::new(int!(2, 4), None);
    assert_eq!(s2xd4_col.to_string(), "(2,4):(1,2)");

    let s2xd4_row = Layout::new(int!(2, 4), Some(compact_row_major(&int!(2, 4))));
    assert_eq!(s2xd4_row.to_string(), "(2,4):(4,1)");

    let s2xh4 = Layout::new(int!(2, int!(2, 2)), Some(int!(4, int!(2, 1))));
    assert_eq!(s2xh4.to_string(), "(2,(2,2)):(4,(2,1))");

    let s2xh4_col = Layout::new(int!(2, int!(2, 2)), None);
    assert_eq!(s2xh4_col.to_string(), "(2,(2,2)):(1,(2,4))");
}

// Helper to make test out easier to read/compare
fn convert_to_2d_vec(layout: &Layout) -> Vec<Vec<i64>> {
    let (rows, cols) = match &layout.shape {
        IntTuple::Int(n) => (*n as usize, 1),
        IntTuple::Tuple(v) => (v[0].size() as usize, v[1].size() as usize),
    };

    let mut result = vec![vec![0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = layout.call(&int!(i as i64, j as i64));
        }
    }
    result
}

// using-a-layout
// http://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#using-a-layout
#[test]
fn test_2d_layouts() {
    let s2xs4 = Layout::new(int!(2, 4), None);
    let expected = vec![
        vec![0, 2, 4, 6], //
        vec![1, 3, 5, 7], //
    ];
    let result = convert_to_2d_vec(&s2xs4);
    assert_eq!(result, expected);

    let s2xd4_a = Layout::new(int!(2, 4), Some(int!(12, 1)));
    let expected = vec![
        vec![0, 1, 2, 3],     //
        vec![12, 13, 14, 15], //
    ];
    let result = convert_to_2d_vec(&s2xd4_a);
    assert_eq!(result, expected);

    let s2xh4_col: Layout = Layout::new(int!(2, int!(2, 2)), None);
    let expected = vec![
        vec![0, 2, 4, 6], //
        vec![1, 3, 5, 7], //
    ];
    let result = convert_to_2d_vec(&s2xh4_col);
    assert_eq!(result, expected);

    let s2xh4 = Layout::new(int!(2, int!(2, 2)), Some(int!(4, int!(2, 1))));
    let expected = vec![
        vec![0, 2, 1, 3], //
        vec![4, 6, 5, 7], //
    ];
    let result = convert_to_2d_vec(&s2xh4);
    assert_eq!(result, expected)
}

// Helper to make test out easier to read/compare
fn convert_to_1d_vec(layout: &Layout) -> Vec<i64> {
    let size = layout.shape.size() as usize;
    let mut result = vec![0; size];
    for i in 0..size {
        result[i] = layout.call_1d(i as i64);
    }
    result
}

#[test]
fn test_1d_layouts() {
    let s2xs4 = Layout::new(int!(2, 4), None);
    let indices = convert_to_1d_vec(&s2xs4);
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    let s2xd4_a = Layout::new(int!(2, 4), Some(int!(12, 1)));
    let indices = convert_to_1d_vec(&s2xd4_a);
    assert_eq!(indices, vec![0, 12, 1, 13, 2, 14, 3, 15]);

    let s2xh4_col = Layout::new(int!(2, int!(2, 2)), None);
    let indices = convert_to_1d_vec(&s2xh4_col);
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    let s2xh4 = Layout::new(int!(2, int!(2, 2)), Some(int!(4, int!(2, 1))));
    let indices = convert_to_1d_vec(&s2xh4);
    assert_eq!(indices, vec![0, 4, 2, 6, 1, 5, 3, 7]);
}

#[test]
fn test_print2d_layout() {
    let layout = Layout::new(int!(2, int!(2, 2)), Some(int!(4, int!(2, 1))));
    let output = print_2d(&layout);
    let expected = r#"(2,(2,2)):(4,(2,1))
      0   1   2   3
    +---+---+---+---+
 0  | 0 | 2 | 1 | 3 |
    +---+---+---+---+
 1  | 4 | 6 | 5 | 7 |
    +---+---+---+---+"#;
    assert_eq!(output, expected);
}

// vector-layouts
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#vector-layouts
#[test]
fn test_vector_layouts() {
    // 8:1
    let layout = Layout::new(int!(8), Some(int!(1)));
    assert_eq!(layout.to_string(), "8:1");
    let indices = convert_to_1d_vec(&layout);
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    // 8:2
    let layout = Layout::new(int!(8), Some(int!(2)));
    assert_eq!(layout.to_string(), "8:2");
    let indices = convert_to_1d_vec(&layout);
    assert_eq!(indices, vec![0, 2, 4, 6, 8, 10, 12, 14]);

    // ((4,2)):((2,1))
    let layout = Layout::new(int!(4, 2).wrap(), Some(int!(2, 1).wrap()));
    assert_eq!(layout.to_string(), "((4,2)):((2,1))");
    let indices = convert_to_1d_vec(&layout);
    assert_eq!(indices, vec![0, 2, 4, 6, 1, 3, 5, 7]);

    // ((4,2)):((1,4))
    let layout = Layout::new(int!(4, 2).wrap(), Some(int!(1, 4).wrap()));
    assert_eq!(layout.to_string(), "((4,2)):((1,4))");
    let indices = convert_to_1d_vec(&layout);
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

// matrix-examples
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#matrix-examples
#[test]
fn test_matrix_examples() {
    // (4,2):(1,4)
    let layout = Layout::new(int!(4, 2), Some(int!(1, 4)));
    let expected = vec![
        vec![0, 4], //
        vec![1, 5], //
        vec![2, 6], //
        vec![3, 7], //
    ];
    let result = convert_to_2d_vec(&layout);
    assert_eq!(result, expected);

    // (4,2):(2,1)
    let layout = Layout::new(int!(4, 2), Some(int!(2, 1)));
    let expected = vec![
        vec![0, 1], //
        vec![2, 3], //
        vec![4, 5], //
        vec![6, 7], //
    ];
    let result = convert_to_2d_vec(&layout);
    assert_eq!(result, expected);

    // ((2,2),2):((4,1),2)
    let layout = Layout::new(int!(int!(2, 2), 2), Some(int!(int!(4, 1), 2)));
    let expected = vec![
        vec![0, 2], //
        vec![4, 6], //
        vec![1, 3], //
        vec![5, 7], //
    ];
    let result = convert_to_2d_vec(&layout);
    assert_eq!(result, expected);
}

// coordinate-mapping
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#coordinate-mapping
#[test]
fn test_coordinate_mapping() {
    let shape = int!(3, int!(2, 3));
    let idx = 16;
    let crd = idx2crd(idx, &shape);
    assert_eq!(crd.to_string(), "(1,(1,2))");

    let coord = int!(1, 5);
    let crd = idx2crd(coord, &shape);
    assert_eq!(crd.to_string(), "(1,(1,2))");

    let coord = int!(1, int!(1, 2));
    let crd = idx2crd(coord, &shape);
    assert_eq!(crd.to_string(), "(1,(1,2))");
}

// index-mapping
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#index-mapping
#[test]
fn test_index_mapping() {
    let shape = int!(3, int!(2, 3));
    let stride = int!(3, int!(12, 1));

    let crd = crd2idx(16, &shape, &stride);
    assert_eq!(crd, 17);

    let coord = int!(1, 5);
    let crd = crd2idx(coord, &shape, &stride);
    assert_eq!(crd, 17);

    let coord = int!(1, int!(1, 2));
    let crd = crd2idx(coord, &shape, &stride);
    assert_eq!(crd, 17);
}

// layout-manipulation
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#layout-manipulation
#[test]
fn test_layout_sublayouts() {
    // Layout a = Layout<Shape<_4,Shape<_3,_6>>>{}; // (4,(3,6)):(1,(4,12))
    let a = Layout::new(int!(4, int!(3, 6)), None);
    assert_eq!(a.to_string(), "(4,(3,6)):(1,(4,12))");

    // layout<0>(a) => 4:1
    let a0 = a.mode(0);
    assert_eq!(a0.to_string(), "4:1");

    // layout<1>(a) => (3,6):(4,12)
    let a1 = a.mode(1);
    assert_eq!(a1.to_string(), "(3,6):(4,12)");

    // layout<1,0>(a) => 3:4
    let a10 = a.layout_at(&[1, 0]);
    assert_eq!(a10.to_string(), "3:4");

    // layout<1,1>(a) => 6:12
    let a11 = a.layout_at(&[1, 1]);
    assert_eq!(a11.to_string(), "6:12");

    // select tests with (2,3,5,7):(1,2,6,30)
    let b = Layout::new(int!(2, 3, 5, 7), None);
    assert_eq!(b.to_string(), "(2,3,5,7):(1,2,6,30)");

    // select<1,3>(b) => (3,7):(2,30)
    let b13 = b.select(&[1, 3]);
    assert_eq!(b13.to_string(), "(3,7):(2,30)");

    // select<0,1,3>(b) => (2,3,7):(1,2,30)
    let b013 = b.select(&[0, 1, 3]);
    assert_eq!(b013.to_string(), "(2,3,7):(1,2,30)");

    // select<2>(b) => (5):(6)
    let b2 = b.select(&[2]);
    assert_eq!(b2.to_string(), "(5):(6)");

    // take<1,3>(b) => (3,5):(2,6)
    let b_take_13 = b.take(1, 3);
    assert_eq!(b_take_13.to_string(), "(3,5):(2,6)");

    // take<1,4>(b) => (3,5,7):(2,6,30)
    let b_take_14 = b.take(1, 4);
    assert_eq!(b_take_14.to_string(), "(3,5,7):(2,6,30)");
}

// concatenation
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#concatenation
#[test]
fn test_layout_concatenation() {
    let a = Layout::new(int!(3), Some(int!(1))); // 3:1
    let b = Layout::new(int!(4), Some(int!(3))); // 4:3

    // make_layout(a, b) => (3,4):(1,3)
    let row = make_layout(&[&a, &b]);
    assert_eq!(row.to_string(), "(3,4):(1,3)");

    // make_layout(b, a) => (4,3):(3,1)
    let col = make_layout(&[&b, &a]);
    assert_eq!(col.to_string(), "(4,3):(3,1)");

    // make_layout(row, col) => ((3,4),(4,3)):((1,3),(3,1))
    let q = make_layout(&[&row, &col]);
    assert_eq!(q.to_string(), "((3,4),(4,3)):((1,3),(3,1))");

    // make_layout(a) => (3):(1)
    let aa = make_layout(&[&a]);
    assert_eq!(aa.to_string(), "(3):(1)");

    // make_layout(aa) => ((3)):((1))
    let aaa = make_layout(&[&aa]);
    assert_eq!(aaa.to_string(), "((3)):((1))");

    // append(a, b) => (3,4):(1,3)
    let ab = a.append(&b);
    assert_eq!(ab.to_string(), "(3,4):(1,3)");

    // prepend(a, b) => (4,3):(3,1)
    let ba = a.prepend(&b);
    assert_eq!(ba.to_string(), "(4,3):(3,1)");

    // append(ab, ab) => (3,4,(3,4)):(1,3,(1,3))
    let c = ab.append(&ab);
    assert_eq!(c.to_string(), "(3,4,(3,4)):(1,3,(1,3))");

    // replace<2>(c, b) => (3,4,4):(1,3,3)
    let d = c.replace(2, &b);
    assert_eq!(d.to_string(), "(3,4,4):(1,3,3)");
}

// grouping-and-flattening
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#grouping-and-flattening
#[test]
fn test_grouping_and_flattening() {
    // (2,3,5,7):(1,2,6,30)
    let a = Layout::new(int!(2, 3, 5, 7), None);
    assert_eq!(a.to_string(), "(2,3,5,7):(1,2,6,30)");

    // group<0,2>(a) => ((2,3),5,7):((1,2),6,30)
    let b = a.group(0, 2);
    assert_eq!(b.to_string(), "((2,3),5,7):((1,2),6,30)");

    // group<1,3>(b) => ((2,3),(5,7)):((1,2),(6,30))
    let c = b.group(1, 3);
    assert_eq!(c.to_string(), "((2,3),(5,7)):((1,2),(6,30))");

    // flatten(b) => (2,3,5,7):(1,2,6,30)
    let f = b.flatten();
    assert_eq!(f.to_string(), "(2,3,5,7):(1,2,6,30)");

    // flatten(c) => (2,3,5,7):(1,2,6,30)
    let e = c.flatten();
    assert_eq!(e.to_string(), "(2,3,5,7):(1,2,6,30)");
}
