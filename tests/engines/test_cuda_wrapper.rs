#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::IntoPyDict;

    #[test]
    fn test_execute_on_device() {
        let gil = Python::acquire_gil();
        let py = gil.python();

        // Define a Python module for testing
        let rust_cuda = PyModule::new(py, "rust_cuda").unwrap();
        rust_cuda.add_function(wrap_pyfunction!(execute_on_device, rust_cuda).unwrap()).unwrap();

        // Test the execute_on_device function
        let result: PyResult<f32> = rust_cuda.call1("execute_on_device", (0, 1.0f32, 2.0f32)).unwrap().extract().unwrap();
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_cuda() {
        // Test the execute_cuda function
        let result = execute_cuda(0, 1.0f32, 2.0f32);
        assert!(result.is_ok());
    }
}