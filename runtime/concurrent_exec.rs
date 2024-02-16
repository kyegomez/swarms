use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::IntoPyDict;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;


#[pymodule]
fn rust_module(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(concurrent_exec, m)?)?;
    Ok(())
}

/// This function wraps Python code in Rust concurrency for ultra high performance.
///
/// # Arguments
///
/// * `py_codes` - A vector of string slices that holds the Python codes to be executed.
/// * `timeout` - An optional duration to specify a timeout for the Python code execution.
/// * `num_threads` - The number of threads to use for executing the Python code.
/// * `error_handler` - A function to handle errors during Python code execution.
/// * `log_function` - A function to log the execution of the Python code.
/// * `result_handler` - A function to handle the results of the Python code execution.
///
/// # Example
///
/// ```
/// let py_codes = vec!["print('Hello, World!')", "print('Hello, Rust!')"];
/// let timeout = Some(Duration::from_secs(5));
/// let num_threads = 4;
/// let error_handler = |e| eprintln!("Error: {}", e);
/// let log_function = |s| println!("Log: {}", s);
/// let result_handler = |r| println!("Result: {:?}", r);
/// execute_python_codes(py_codes, timeout, num_threads, error_handler, log_function, result_handler);
/// ```

#[pyfunction]
pub fn concurrent_exec<F, G, H>(
    py_codes: Vec<&str>,
    timeout: Option<Duration>,
    num_threads: usize,
    error_handler: F,
    log_function: G,
    result_handler: H,
) -> PyResult<Vec<PyResult<()>>>
where
    F: Fn(&str),
    G: Fn(&str),
    H: Fn(&PyResult<()>),
{
    let gil = Python::acquire_gil();
    let py = gil.python();
    let py_codes = Arc::new(Mutex::new(py_codes));
    let results = Arc::new(Mutex::new(Vec::new()));
    let pool = ThreadPool::new(num_threads);

    pool.install(|| {
        py_codes.par_iter().for_each(|code| {
            let locals = [("__name__", "__main__")].into_py_dict(py);
            let globals = [("__name__", "__main__")].into_py_dict(py);

            log_function(&format!("Executing Python code: {}", code));
            let result = py.run(code, Some(globals), Some(locals));

            match timeout {
                Some(t) => {
                    let now = Instant::now();
                    let timeout_thread = thread::spawn(move || {
                        while now.elapsed() < t {
                            if let Ok(_) = result {
                                break;
                            }
                        }
                        if now.elapsed() >= t {
                            error_handler(&format!("Python code execution timed out: {}", code));
                        }
                    });

                    timeout_thread.join().unwrap();
                }
                None => {}
            }

            results.lock().unwrap().push(result.clone(result));
            result_handler(&result);
        });
    });

    pool.join();
    Ok(results.lock().unwrap().clone())
}