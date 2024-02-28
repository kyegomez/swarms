use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use std::fs;
use std::time::Instant;

// Define the new execute function
fn execute(script_path: &str, threads: usize) -> PyResult<()> {
    (0..threads).into_par_iter().for_each(|_| {
        Python::with_gil(|py| {
            let sys = py.import("sys").unwrap();
            let path: &PyList = match sys.getattr("path") {
                Ok(path) => match path.downcast() {
                    Ok(path) => path,
                    Err(e) => {
                        eprintln!("Failed to downcast path: {:?}", e);
                        return;
                    }
                },
                Err(e) => {
                    eprintln!("Failed to get path attribute: {:?}", e);
                    return;
                }
            };

            if let Err(e) = path.append("lib/python3.11/site-packages") {
                eprintln!("Failed to append path: {:?}", e);
            }

            let script = fs::read_to_string(script_path).unwrap();
            py.run(&script, None, None).unwrap();
        });
    });
    Ok(())
}

fn main() -> PyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let threads = 20;

    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_python_script>", args[0]);
        std::process::exit(1);
    }
    let script_path = &args[1];

    let start = Instant::now();

    // Call the execute function
    execute(script_path, threads)?;

    let duration = start.elapsed();
    match fs::write("/tmp/elapsed.time", format!("booting time: {:?}", duration)) {
        Ok(_) => println!("Successfully wrote elapsed time to /tmp/elapsed.time"),
        Err(e) => eprintln!("Failed to write elapsed time: {:?}", e),
    }

    Ok(())
}
