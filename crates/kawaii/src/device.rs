/// A fixed-size 2D layout that is cheap to construct inside GPU kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Layout2D {
    shape: [usize; 2],
    stride: [usize; 2],
}

impl Layout2D {
    pub const fn new(shape: [usize; 2], stride: [usize; 2]) -> Self {
        Self { shape, stride }
    }

    pub const fn row_major(rows: usize, cols: usize) -> Self {
        Self::new([rows, cols], [cols, 1])
    }

    pub const fn col_major(rows: usize, cols: usize) -> Self {
        Self::new([rows, cols], [1, rows])
    }

    pub const fn rows(&self) -> usize {
        self.shape[0]
    }

    pub const fn cols(&self) -> usize {
        self.shape[1]
    }

    pub const fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub const fn stride(&self) -> [usize; 2] {
        self.stride
    }

    pub const fn contains(&self, row: usize, col: usize) -> bool {
        row < self.rows() && col < self.cols()
    }

    pub const fn index(&self, row: usize, col: usize) -> usize {
        row * self.stride[0] + col * self.stride[1]
    }

    pub const fn cosize(&self) -> usize {
        if self.rows() == 0 || self.cols() == 0 {
            0
        } else {
            self.index(self.rows() - 1, self.cols() - 1) + 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Layout2D;

    #[test]
    fn row_major_indices_match_expectations() {
        let layout = Layout2D::row_major(3, 4);
        assert_eq!(layout.index(2, 1), 9);
        assert_eq!(layout.cosize(), 12);
    }

    #[test]
    fn col_major_indices_match_expectations() {
        let layout = Layout2D::col_major(3, 4);
        assert_eq!(layout.index(2, 1), 5);
        assert_eq!(layout.cosize(), 12);
    }
}
