#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::IntoPyDict;

    #[test]
    fn test_process_python_modules() {
        let gil = Python::acquire_gil();
        let py = gil.python();

        // Define a Python module for testing
        let code = r#"
        def test_function():
            return "Hello, World!"
        "#;
        let test_module = PyModule::new(py, "test_module").unwrap();
        test_module.add_function(wrap_pyfunction!(test_function, test_module).unwrap()).unwrap();
        test_module.add(py, "test_function", code).unwrap();

        // Define a PythonModule struct for testing
        let test_python_module = PythonModule {
            name: "test_module",
            function: "test_function",
        };

        // Test the process_python_modules function
        let result = process_python_modules(vec![test_python_module], 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_python_modules_import_error() {
        // Define a PythonModule struct with a non-existent module
        let test_python_module = PythonModule {
            name: "non_existent_module",
            function: "test_function",
        };

        // Test the process_python_modules function
        let result = process_python_modules(vec![test_python_module], 1);
        assert!(matches!(result, Err(PythonError::ImportError(_))));
    }

    #[test]
    fn test_process_python_modules_function_error() {
        let gil = Python::acquire_gil();
        let py = gil.python();

        // Define a Python module for testing
        let test_module = PyModule::new(py, "test_module").unwrap();

        // Define a PythonModule struct with a non-existent function
        let test_python_module = PythonModule {
            name: "test_module",
            function: "non_existent_function",
        };

        // Test the process_python_modules function
        let result = process_python_modules(vec![test_python_module], 1);
        assert!(matches!(result, Err(PythonError::FunctionError(_))));
    }
}