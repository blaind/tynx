//! Thread-local eager gradient-mode controls.

use std::cell::Cell;

use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyAny};

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
