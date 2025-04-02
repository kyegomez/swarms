use std::fmt::Write;

pub fn hex_string(vec: &Vec<u8>) -> String {
    let mut s = String::new();
    for byte in vec {
        write!(&mut s, "{:02x}", byte).expect("Unable to write");
    }
    s
}
