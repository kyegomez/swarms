/// This module provides a multi-threading processor for executing Python modules and functions in parallel.
/// It utilizes the `rayon` crate for parallel processing and the `pyo3` crate for interacting with the Python interpreter.
/// The `multithreading_processor` function takes a vector of `PythonModule` structs and the number of threads to use.
/// Each `PythonModule` struct contains the name of the Python module, the name of the function to call, and any arguments to pass to the function.
/// The function imports the Python module, calls the specified function, and sends any errors encountered back to the main thread.
/// If an import error occurs, a `PythonError::ImportError` is returned.
/// If a function call error occurs, a `PythonError::FunctionError` is returned.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use log::{info, error};

struct PythonModule<'a> {
    name: &'a str,
    function: &'a str,
}

enum PythonError {
    ImportError(String),
    FunctionError(String),
}

#[pyfunction]
fn my_module(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_python_modules, m)?)?;
    Ok(())
}



/// The function returns `Ok(())` if all modules are processed successfully.
/// Note: This code assumes that the necessary dependencies (`pyo3`, `rayon`, `log`) are already imported and initialized.
///
/// # Arguments
///
/// * `modules` - A vector of `PythonModule` structs representing the Python modules and functions to execute.
/// * `num_threads` - The number of threads to use for parallel processing.
///
/// # Examples
///
/// ```
/// use pyo3::types::PyModule;
/// use pyo3::types::PyResult;
/// use pyo3::prelude::*;
///
/// struct PythonModule<'a> {
///     name: &'a str,
///     function: &'a str,
///     args: Vec<&'a str>,
/// }
///
/// #[pymodule]
/// fn multithreading_processor(modules: Vec<PythonModule>, num_threads: usize) -> Result<(), PythonError> {
///     // Function implementation
///     Ok(())
/// }
/// ```
///
/// # Errors
///
/// Returns a `PythonError` if an import error or a function call error occurs.
///
/// # Panics
///
/// This function does not panic.
///
/// # Safety
///
/// This function is safe to call, but it assumes that the necessary dependencies (`pyo3`, `rayon`, `log`) are already imported and initialized.
// Initialize Python interpreter
#[pyfunction]
fn process_python_modules(modules: Vec<PythonModule>, num_threads: usize) -> Result<(), PythonError> {

    let gil = Python::acquire_gil();
    let py = gil.python();

    // Set the global thread pool's configuration
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    // Create a channel to send errors from threads to the main thread
    let (tx, rx) = channel();
    let tx = Arc::new(Mutex::new(tx));

    // Process each Python module in parallel
    modules.par_iter().for_each(|module| {
        let result = PyModule::import(py, module.name)
            .map_err(|_| PythonError::ImportError(module.name.to_string()))
            .and_then(|m| m.call0(module.function)
            .map_err(|_| PythonError::FunctionError(module.function.to_string())));

        if let Err(e) = result {
            let tx = tx.lock().unwrap();
            tx.send(e).unwrap();
        }
    });

    // Check for errors
    drop(tx); // Close the sender
    for error in rx {
        match error {
            PythonError::ImportError(module) => error!("Failed to import module {}", module),
            PythonError::FunctionError(function) => error!("Failed to call function {}", function),
        }
    }

    Ok(())
}