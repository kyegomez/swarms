use std::fs::File;
use std::io::prelude::*;
use std::time::Instant;
use std::io::{BufReader, io};
use ranyon::prelude::{IntoParallelRefIterator, ParallelIterator};

fn read_file(path: &str) -> Vec<String> {
    /// Reads the contents of a file located at the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of strings representing the lines of the file if the file was successfully read,
    /// or an `io::Error` if there was an error reading the file.
    ///
    /// # Example
    ///
    /// ```
    /// use std::io;
    /// use std::fs::File;
    /// use std::io::BufReader;
    ///
    /// fn read_file(path: &str) -> io::Result<Vec<String>> {
    ///     let contents: io::Result<Vec<String>> = BufReader::new(File::open(path).expect("Could not open file"))
    ///         .lines()
    ///         .collect();
    ///     contents
    /// }
    /// ```
    let contents: io::Result<Vec<String>> = BufReader::new(File::open(path).expect("Could not open file"))
        .lines()
        .collect();
    return contents.expect("Could not read file");
}