use pyo3::prelude::*;
use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use std::error::Error;
use std::ffi::CString;

#[pymodule]
fn rust_cuda(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "execute_on_device")]
    fn execute_on_device(py: Python, device_id: u32, a: f32, b: f32) -> PyResult<f32> {
        /// The result of executing the CUDA operation.
        let result = py.allow_threads(|| {
            execute_cuda(device_id, a, b)
        });
        match result {
            Ok(res) => Ok(res),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", err))),
        }
    }
    Ok(())
}

fn execute_cuda(device_id: u32, a: f32, b: f32) -> Result<f32, Box<dyn Error>> {
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(device_id)?;
    /// Creates a new CUDA context and pushes it onto the current thread's stack.
    ///
    /// # Arguments
    ///
    /// * `flags` - The flags to be used when creating the context.
    /// * `device` - The device on which the context will be created.
    ///
    /// # Returns
    ///
    /// The newly created CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if the context creation fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use swarms::cuda_wrapper::Context;
    ///
    /// let device = 0;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    /// ```
    pub fn create_and_push(flags: ContextFlags, device: i32) -> Result<Context, CudaError> {
        // implementation goes here
    }
    let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    let module_data = CString::new(include_str!("../resources/add.ptx"))?;
    let module = Module::load_from_string(&module_data)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let mut x = DeviceBox::new(&a)?;
    let mut y = DeviceBox::new(&b)?;
    let mut result = DeviceBox::new(&0.0f32)?;
    unsafe {
        launch!(module.sum<<<1, 1, 0, stream>>>(
            x.as_device_ptr(),
            y.as_device_ptr(),
            result.as_device_ptr(),
            1
        ))?;
    }
    stream.synchronize()?;
    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;
    Ok(result_host)
}