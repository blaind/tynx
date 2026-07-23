//! Thread-local eager gradient-mode controls.

use std::cell::Cell;

use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyAny, PyDict, PyTuple},
};

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub(crate) fn is_grad_enabled() -> bool {
    GRAD_ENABLED.get()
}

fn replace_grad_enabled(enabled: bool) -> bool {
    GRAD_ENABLED.replace(enabled)
}

/// Context manager returned by [`no_grad`].
#[pyclass(name = "_NoGrad", frozen, unsendable)]
pub(crate) struct PyNoGrad {
    previous: Cell<Option<bool>>,
}

#[pymethods]
impl PyNoGrad {
    fn __enter__(&self) -> PyResult<()> {
        if self.previous.get().is_some() {
            return Err(PyRuntimeError::new_err(
                "the same no_grad() context manager cannot be entered twice",
            ));
        }
        self.previous.set(Some(replace_grad_enabled(false)));
        Ok(())
    }

    fn __exit__(
        &self,
        _exception_type: &Bound<'_, PyAny>,
        _exception: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        self.restore();
        false
    }

    fn __call__(&self, py: Python<'_>, function: Py<PyAny>) -> PyResult<Py<PyAny>> {
        no_grad_wrapper(py, function)
    }
}

/// Callable wrapper produced when `no_grad()` is used as a decorator.
#[pyclass(name = "_NoGradFunction", dict, frozen, unsendable)]
pub(crate) struct PyNoGradFunction {
    function: Py<PyAny>,
}

#[pymethods]
impl PyNoGradFunction {
    fn __get__(
        &self,
        py: Python<'_>,
        instance: &Bound<'_, PyAny>,
        owner: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let function = self
            .function
            .bind(py)
            .call_method1("__get__", (instance, owner))?
            .unbind();
        no_grad_wrapper(py, function)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let previous = replace_grad_enabled(false);
        let result = self.function.bind(py).call(args, kwargs).map(Bound::unbind);
        replace_grad_enabled(previous);
        result
    }
}

fn no_grad_wrapper(py: Python<'_>, function: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let wrapper = Py::new(
        py,
        PyNoGradFunction {
            function: function.clone_ref(py),
        },
    )?;
    py.import("functools")?
        .getattr("update_wrapper")?
        .call1((wrapper.clone_ref(py), function))?;
    Ok(wrapper.into_any())
}

impl PyNoGrad {
    fn restore(&self) {
        if let Some(previous) = self.previous.take() {
            replace_grad_enabled(previous);
        }
    }
}

impl Drop for PyNoGrad {
    fn drop(&mut self) {
        self.restore();
    }
}

/// Disable eager graph construction inside a `with` block.
#[pyfunction]
pub(crate) fn no_grad() -> PyNoGrad {
    PyNoGrad {
        previous: Cell::new(None),
    }
}

/// Return whether eager operations currently construct autodiff graphs.
#[pyfunction(name = "is_grad_enabled")]
pub(crate) fn is_grad_enabled_py() -> bool {
    is_grad_enabled()
}
