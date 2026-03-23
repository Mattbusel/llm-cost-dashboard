//! # Terminal Visualization
//!
//! Sparklines, histograms, bar charts, and time-series plots rendered as
//! printable ASCII / Unicode strings suitable for terminal display.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::visualization::{Sparkline, BarChart, Histogram};
//!
//! let values = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 3.0, 7.0];
//! let line = Sparkline::render(&values, 8);
//! println!("{line}");
//!
//! let labeled = Sparkline::with_label("cost", &values, 8);
//! println!("{labeled}");
//!
//! let chart = BarChart::render(
//!     &["gpt-4", "claude-3", "gemini"],
//!     &[3.50, 2.10, 1.80],
//!     40,
//! );
//! println!("{chart}");
//! ```

// ── Sparkline ────────────────────────────────────────────────────────────────

/// Renders a Unicode sparkline from a series of f64 values.
pub struct Sparkline;

/// The 8 Unicode block characters used for sparklines, from lowest to highest.
const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

impl Sparkline {
    /// Renders `values` as a sparkline of exactly `width` characters.
    ///
    /// If `values` has more elements than `width`, values are down-sampled by
    /// averaging consecutive buckets.  If `values` is empty an all-space
    /// string of `width` chars is returned.
    pub fn render(values: &[f64], width: usize) -> String {
        if values.is_empty() || width == 0 {
            return " ".repeat(width);
        }

        // Down-sample to `width` buckets if needed.
        let sampled = Self::downsample(values, width);

        let min = sampled.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = sampled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        sampled
            .iter()
            .map(|&v| {
                let idx = if range < f64::EPSILON {
                    3 // all equal → middle block
                } else {
                    let norm = (v - min) / range;
                    ((norm * 7.0).round() as usize).min(7)
                };
                BLOCKS[idx]
            })
            .collect()
    }

    /// Renders `values` as a labeled sparkline:
    /// `"label: [sparkline] min..max"`.
    pub fn with_label(label: &str, values: &[f64], width: usize) -> String {
        let spark = Self::render(values, width);
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if values.is_empty() {
            format!("{}: [] (empty)", label)
        } else {
            format!("{}: [{}] {:.2}..{:.2}", label, spark, min, max)
        }
    }

    /// Resamples `values` to exactly `n` samples by linear interpolation.
    ///
    /// If `values.len() > n` adjacent values are averaged (down-sample).
    /// If `values.len() < n` values are repeated/interpolated (up-sample).
    fn downsample(values: &[f64], n: usize) -> Vec<f64> {
        if values.len() == n {
            return values.to_vec();
        }
        if values.len() == 1 {
            return vec![values[0]; n];
        }
        // Map each output index to a fractional position in the source.
        (0..n)
            .map(|i| {
                let src_f = i as f64 * (values.len() - 1) as f64 / (n - 1).max(1) as f64;
                let lo = src_f.floor() as usize;
                let hi = (lo + 1).min(values.len() - 1);
                let t = src_f - lo as f64;
                values[lo] * (1.0 - t) + values[hi] * t
            })
            .collect()
    }
}

// ── Histogram ────────────────────────────────────────────────────────────────

/// Renders an ASCII histogram of a numeric dataset.
pub struct Histogram {
    /// Number of buckets to partition the data into.
    pub buckets: usize,
}

impl Histogram {
    /// Creates a `Histogram` with the given bucket count.
    pub fn new(buckets: usize) -> Self {
        Self {
            buckets: buckets.max(1),
        }
    }

    /// Partitions `values` into `n` equal-width buckets.
    ///
    /// Returns a `Vec` of `(bucket_min, bucket_max, count)` tuples.
    pub fn compute_buckets(values: &[f64], n: usize) -> Vec<(f64, f64, usize)> {
        if values.is_empty() || n == 0 {
            return vec![];
        }
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        let width = if range < f64::EPSILON { 1.0 } else { range / n as f64 };

        let mut buckets: Vec<(f64, f64, usize)> = (0..n)
            .map(|i| {
                let lo = min + i as f64 * width;
                let hi = lo + width;
                (lo, hi, 0)
            })
            .collect();

        for &v in values {
            let idx = if range < f64::EPSILON {
                0
            } else {
                ((v - min) / width).floor() as usize
            };
            let idx = idx.min(n - 1);
            buckets[idx].2 += 1;
        }
        buckets
    }

    /// Renders a text histogram of `values` with the given dimensions.
    ///
    /// Each bucket is drawn as a column of `#` characters scaled to `height`
    /// rows.  Bucket ranges and counts are printed below the chart.
    pub fn render(&self, values: &[f64], width: usize, height: usize) -> String {
        if values.is_empty() {
            return "(no data)".to_string();
        }
        let height = height.max(1);
        let width = width.max(self.buckets);

        let buckets = Self::compute_buckets(values, self.buckets);
        let max_count = buckets.iter().map(|b| b.2).max().unwrap_or(1).max(1);

        // Determine the character width of each bar column.
        let col_width = (width / self.buckets).max(1);

        let mut rows: Vec<String> = Vec::new();

        // Draw bars from top to bottom.
        for row in (0..height).rev() {
            let threshold = row as f64 / (height - 1).max(1) as f64 * max_count as f64;
            let mut line = String::new();
            for (_, _, count) in &buckets {
                let bar = if *count as f64 > threshold { '#' } else { ' ' };
                for _ in 0..col_width {
                    line.push(bar);
                }
            }
            rows.push(line);
        }

        // Footer: bucket ranges and counts.
        rows.push("-".repeat(col_width * self.buckets));
        for (lo, hi, count) in &buckets {
            let label = format!("[{:.1},{:.1}]:{}", lo, hi, count);
            let padded = format!("{:<width$}", label, width = col_width);
            rows.push(padded);
        }

        rows.join("\n")
    }
}

// ── BarChart ─────────────────────────────────────────────────────────────────

/// Renders horizontal and vertical bar charts.
pub struct BarChart;

/// Full block character used to fill bars.
const FULL_BLOCK: char = '█';
/// Light shade block for empty portion.
const LIGHT_SHADE: char = '░';

impl BarChart {
    /// Renders a horizontal bar chart.
    ///
    /// Each row is formatted as `"label | ████░░ value"`.  The bar width is
    /// scaled so the largest value fills `width` characters of bar space.
    pub fn render(labels: &[&str], values: &[f64], width: usize) -> String {
        if labels.is_empty() || values.is_empty() {
            return String::new();
        }
        let max_label = labels.iter().map(|l| l.len()).max().unwrap_or(0);
        let max_value = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bar_area = width.saturating_sub(max_label + 3 + 10).max(4);

        let mut lines: Vec<String> = Vec::new();
        for (label, value) in labels.iter().zip(values.iter()) {
            let filled = if max_value < f64::EPSILON {
                0
            } else {
                ((value / max_value) * bar_area as f64).round() as usize
            };
            let empty = bar_area.saturating_sub(filled);
            let bar: String = std::iter::repeat(FULL_BLOCK)
                .take(filled)
                .chain(std::iter::repeat(LIGHT_SHADE).take(empty))
                .collect();
            lines.push(format!(
                "{:>width$} | {} {:.2}",
                label,
                bar,
                value,
                width = max_label
            ));
        }
        lines.join("\n")
    }

    /// Renders a vertical bar chart with labels beneath each column.
    ///
    /// Each column is `height` rows tall; the tallest bar fills all rows.
    pub fn render_vertical(labels: &[&str], values: &[f64], height: usize) -> String {
        if labels.is_empty() || values.is_empty() {
            return String::new();
        }
        let height = height.max(1);
        let max_value = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let col_width = 3usize;

        // Compute bar heights.
        let bar_heights: Vec<usize> = values
            .iter()
            .map(|&v| {
                if max_value < f64::EPSILON {
                    0
                } else {
                    ((v / max_value) * height as f64).round() as usize
                }
            })
            .collect();

        let mut rows: Vec<String> = Vec::new();
        for row in (0..height).rev() {
            let mut line = String::new();
            for &bh in &bar_heights {
                if bh > row {
                    for _ in 0..col_width {
                        line.push(FULL_BLOCK);
                    }
                } else {
                    for _ in 0..col_width {
                        line.push(' ');
                    }
                }
            }
            rows.push(line);
        }

        // Label row.
        let label_row: String = labels
            .iter()
            .map(|l| format!("{:^width$}", &l[..l.len().min(col_width)], width = col_width))
            .collect();
        rows.push(label_row);

        rows.join("\n")
    }
}

// ── TimeSeriesPlot ────────────────────────────────────────────────────────────

/// Renders ASCII line plots of time series data.
pub struct TimeSeriesPlot;

impl TimeSeriesPlot {
    /// Renders a single time series as an ASCII plot.
    ///
    /// Points are marked with `·` and connected with `|`, `/`, or `\`.
    pub fn render(series: &[(u64, f64)], width: usize, height: usize) -> String {
        if series.is_empty() || width < 2 || height < 2 {
            return "(no data)".to_string();
        }

        let (grid, x_min, x_max, y_min, y_max) =
            Self::build_grid(std::slice::from_ref(&("", series.to_vec())), width, height, &['·']);

        let mut rows: Vec<String> = Vec::new();
        for (row_idx, row) in grid.iter().enumerate() {
            let y_val = y_max - (row_idx as f64 / (height - 1) as f64) * (y_max - y_min);
            let line: String = row.iter().collect();
            rows.push(format!("{:>8.2} |{}", y_val, line));
        }

        // X axis.
        let axis = format!("{:>8} +{}", "", "-".repeat(width));
        rows.push(axis);
        let x_min_str = Self::fmt_ts(x_min);
        let x_max_str = Self::fmt_ts(x_max);
        let pad = width.saturating_sub(x_min_str.len() + x_max_str.len());
        rows.push(format!("         {}{}{}", x_min_str, " ".repeat(pad), x_max_str));

        rows.join("\n")
    }

    /// Renders multiple time series on the same plot using different marker
    /// characters.
    pub fn render_multi(
        series: &[(&str, Vec<(u64, f64)>)],
        width: usize,
        height: usize,
    ) -> String {
        if series.is_empty() || width < 2 || height < 2 {
            return "(no data)".to_string();
        }

        let markers: Vec<char> = vec!['·', '×', '+', '*', 'o', '#', '@', '%'];
        let owned: Vec<(&str, Vec<(u64, f64)>)> = series.to_vec();
        let (grid, _, _, y_min, y_max) =
            Self::build_grid(&owned, width, height, &markers);

        let mut rows: Vec<String> = Vec::new();
        for (row_idx, row) in grid.iter().enumerate() {
            let y_val = y_max - (row_idx as f64 / (height - 1) as f64) * (y_max - y_min);
            let line: String = row.iter().collect();
            rows.push(format!("{:>8.2} |{}", y_val, line));
        }
        rows.push(format!("{:>8} +{}", "", "-".repeat(width)));

        // Legend.
        for (i, (name, _)) in series.iter().enumerate() {
            let marker = markers.get(i).copied().unwrap_or('·');
            rows.push(format!("  {} {}", marker, name));
        }

        rows.join("\n")
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    fn fmt_ts(ts: u64) -> String {
        format!("{}", ts)
    }

    /// Builds a `height x width` character grid with all series plotted.
    ///
    /// Returns `(grid, x_min, x_max, y_min, y_max)`.
    fn build_grid(
        series: &[(&str, Vec<(u64, f64)>)],
        width: usize,
        height: usize,
        markers: &[char],
    ) -> (Vec<Vec<char>>, u64, u64, f64, f64) {
        // Determine global bounds.
        let all_ts: Vec<u64> = series.iter().flat_map(|(_, pts)| pts.iter().map(|p| p.0)).collect();
        let all_vals: Vec<f64> = series.iter().flat_map(|(_, pts)| pts.iter().map(|p| p.1)).collect();

        if all_ts.is_empty() {
            let grid = vec![vec![' '; width]; height];
            return (grid, 0, 1, 0.0, 1.0);
        }

        let x_min = *all_ts.iter().min().unwrap_or(&0);
        let x_max = *all_ts.iter().max().unwrap_or(&1);
        let y_min = all_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = all_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let x_range = (x_max - x_min).max(1) as f64;
        let y_range = (y_max - y_min).abs().max(f64::EPSILON);

        let mut grid = vec![vec![' '; width]; height];

        for (s_idx, (_, points)) in series.iter().enumerate() {
            let marker = markers.get(s_idx).copied().unwrap_or('·');
            let mut prev_col: Option<usize> = None;
            let mut prev_row: Option<usize> = None;

            // Sort by timestamp.
            let mut pts = points.clone();
            pts.sort_by_key(|p| p.0);

            for (ts, val) in &pts {
                let col = (((ts - x_min) as f64 / x_range) * (width - 1) as f64).round() as usize;
                let col = col.min(width - 1);
                let row = ((y_max - val) / y_range * (height - 1) as f64).round() as usize;
                let row = row.min(height - 1);

                // Connect to previous point.
                if let (Some(pc), Some(pr)) = (prev_col, prev_row) {
                    Self::draw_line(&mut grid, pc, pr, col, row, height, width);
                }

                grid[row][col] = marker;
                prev_col = Some(col);
                prev_row = Some(row);
            }
        }

        (grid, x_min, x_max, y_min, y_max)
    }

    /// Draws a connecting line between two points on the grid using Bresenham's
    /// algorithm and `|`, `/`, `\` characters.
    fn draw_line(
        grid: &mut Vec<Vec<char>>,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        height: usize,
        width: usize,
    ) {
        // Simple step interpolation.
        let dx = x1 as i64 - x0 as i64;
        let dy = y1 as i64 - y0 as i64;
        let steps = dx.unsigned_abs().max(dy.unsigned_abs()) as usize;
        if steps == 0 {
            return;
        }

        for i in 1..steps {
            let x = (x0 as i64 + (dx * i as i64) / steps as i64) as usize;
            let y = (y0 as i64 + (dy * i as i64) / steps as i64) as usize;
            if x < width && y < height && grid[y][x] == ' ' {
                let ch = if dx == 0 {
                    '|'
                } else if dy == 0 {
                    '-'
                } else if (dy > 0) == (dx > 0) {
                    '\\'
                } else {
                    '/'
                };
                grid[y][x] = ch;
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparkline_length_equals_width() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        for width in [1, 5, 10, 20, 30] {
            let spark = Sparkline::render(&values, width);
            assert_eq!(
                spark.chars().count(),
                width,
                "sparkline width mismatch for width={width}"
            );
        }
    }

    #[test]
    fn sparkline_empty_input() {
        let spark = Sparkline::render(&[], 8);
        assert_eq!(spark.len(), 8);
    }

    #[test]
    fn sparkline_with_label_contains_label() {
        let values = vec![1.0, 2.0, 3.0];
        let out = Sparkline::with_label("cost", &values, 5);
        assert!(out.starts_with("cost:"));
        assert!(out.contains(".."));
    }

    #[test]
    fn sparkline_uniform_values() {
        // All equal values should not panic.
        let values = vec![5.0; 10];
        let spark = Sparkline::render(&values, 10);
        assert_eq!(spark.chars().count(), 10);
    }

    #[test]
    fn histogram_bucket_count() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let hist = Histogram::new(5);
        let buckets = Histogram::compute_buckets(&values, 5);
        assert_eq!(buckets.len(), 5, "should produce exactly 5 buckets");
        let total: usize = buckets.iter().map(|b| b.2).sum();
        assert_eq!(total, 100, "all values should be assigned to a bucket");
    }

    #[test]
    fn histogram_render_contains_separator() {
        let values: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let hist = Histogram::new(4);
        let out = hist.render(&values, 40, 5);
        assert!(out.contains('-'));
    }

    #[test]
    fn histogram_empty_values() {
        let hist = Histogram::new(5);
        let out = hist.render(&[], 40, 5);
        assert_eq!(out, "(no data)");
    }

    #[test]
    fn bar_chart_contains_labels() {
        let labels = &["alpha", "beta", "gamma"];
        let values = &[10.0, 5.0, 8.0];
        let out = BarChart::render(labels, values, 60);
        for label in labels {
            assert!(out.contains(label), "bar chart should contain label '{label}'");
        }
    }

    #[test]
    fn bar_chart_contains_bars() {
        let labels = &["a", "b"];
        let values = &[1.0, 2.0];
        let out = BarChart::render(labels, values, 40);
        assert!(out.contains(FULL_BLOCK));
    }

    #[test]
    fn bar_chart_vertical_renders() {
        let labels = &["x", "y", "z"];
        let values = &[3.0, 7.0, 5.0];
        let out = BarChart::render_vertical(labels, values, 8);
        assert!(!out.is_empty());
        // The tallest bar (y=7) should produce the most filled rows.
        assert!(out.contains(FULL_BLOCK));
    }

    #[test]
    fn time_series_plot_renders() {
        let series: Vec<(u64, f64)> = vec![
            (0, 1.0), (1, 2.0), (2, 1.5), (3, 3.0), (4, 2.5),
        ];
        let out = TimeSeriesPlot::render(&series, 20, 5);
        assert!(!out.is_empty());
        assert!(out.contains('|'));
    }

    #[test]
    fn time_series_multi_renders() {
        let s1: Vec<(u64, f64)> = vec![(0, 1.0), (1, 2.0), (2, 3.0)];
        let s2: Vec<(u64, f64)> = vec![(0, 3.0), (1, 2.0), (2, 1.0)];
        let series = vec![("series-a", s1), ("series-b", s2)];
        let out = TimeSeriesPlot::render_multi(&series, 20, 5);
        assert!(out.contains("series-a"));
        assert!(out.contains("series-b"));
    }
}
