//! # Log file tailing
//!
//! Follows a newline-delimited JSON log file the way `tail -f` does: lines
//! appended after a given byte offset are delivered over a channel as they
//! are written.  Partial lines are held back until their newline arrives,
//! and a file that shrinks (truncated or rotated in place) is re-read from
//! the start.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

/// Read every complete line in `path`, returning the lines and the byte
/// offset just past the last complete line.
///
/// A trailing line without a newline is not returned; its bytes are left for
/// [`follow`] to pick up once the writer finishes it.
///
/// # Errors
///
/// Returns the underlying I/O error if the file cannot be opened or read.
pub fn read_complete_lines(path: &Path) -> std::io::Result<(Vec<String>, u64)> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    let end = buf.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
    let lines = split_lines(&buf[..end]);
    Ok((lines, end as u64))
}

fn split_lines(bytes: &[u8]) -> Vec<String> {
    String::from_utf8_lossy(bytes)
        .lines()
        .map(|l| l.trim_end_matches('\r').to_string())
        .filter(|l| !l.trim().is_empty())
        .collect()
}

/// Start following `path` from byte `offset` on a background thread.
///
/// Every complete line appended to the file is sent on the returned channel.
/// The thread polls every `interval` and stops once the receiver is dropped.
pub fn follow(path: impl Into<PathBuf>, offset: u64, interval: Duration) -> Receiver<String> {
    let (tx, rx) = mpsc::channel();
    let path = path.into();
    std::thread::Builder::new()
        .name("llm-dash-tail".into())
        .spawn(move || follow_loop(&path, offset, interval, &tx))
        .map_err(|e| tracing::warn!(error = %e, "could not start log tail thread"))
        .ok();
    rx
}

fn follow_loop(path: &Path, mut offset: u64, interval: Duration, tx: &Sender<String>) {
    let mut pending: Vec<u8> = Vec::new();
    loop {
        match poll_once(path, &mut offset, &mut pending) {
            Ok(lines) => {
                for line in lines {
                    if tx.send(line).is_err() {
                        return; // receiver gone: dashboard exited
                    }
                }
            }
            Err(e) => tracing::debug!(error = %e, "log tail poll failed; retrying"),
        }
        std::thread::sleep(interval);
    }
}

/// One poll step: read any bytes past `offset`, return completed lines.
fn poll_once(path: &Path, offset: &mut u64, pending: &mut Vec<u8>) -> std::io::Result<Vec<String>> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    if len < *offset {
        // Truncated or replaced: start again from the top.
        *offset = 0;
        pending.clear();
    }
    if len == *offset {
        return Ok(Vec::new());
    }
    file.seek(SeekFrom::Start(*offset))?;
    let mut chunk = Vec::new();
    file.read_to_end(&mut chunk)?;
    *offset += chunk.len() as u64;
    pending.extend_from_slice(&chunk);
    let Some(last_nl) = pending.iter().rposition(|b| *b == b'\n') else {
        return Ok(Vec::new());
    };
    let complete: Vec<u8> = pending.drain(..=last_nl).collect();
    Ok(split_lines(&complete))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn reads_only_complete_lines() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "a\nb\npartial").unwrap();
        let (lines, off) = read_complete_lines(f.path()).unwrap();
        assert_eq!(lines, vec!["a", "b"]);
        assert_eq!(off, 4);
    }

    #[test]
    fn poll_picks_up_appends_and_partial_lines() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "one\n").unwrap();
        let (_, mut off) = read_complete_lines(f.path()).unwrap();
        let mut pending = Vec::new();
        assert!(poll_once(f.path(), &mut off, &mut pending).unwrap().is_empty());
        write!(f, "two\nthr").unwrap();
        f.flush().unwrap();
        assert_eq!(poll_once(f.path(), &mut off, &mut pending).unwrap(), vec!["two"]);
        write!(f, "ee\n").unwrap();
        f.flush().unwrap();
        assert_eq!(poll_once(f.path(), &mut off, &mut pending).unwrap(), vec!["three"]);
    }

    #[test]
    fn follow_delivers_appended_lines() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        let rx = follow(f.path(), 0, Duration::from_millis(20));
        writeln!(f, "{{\"x\":1}}").unwrap();
        f.flush().unwrap();
        let got = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(got, "{\"x\":1}");
    }
}
