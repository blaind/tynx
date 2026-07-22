//! Deterministic named storage for parameters and tied aliases.

use std::collections::{BTreeSet, HashMap};

use tynx_core::{Result, TynxError};

use crate::{ParamId, ParameterSlot};

#[derive(Debug)]
struct StoreEntry {
    slot: ParameterSlot,
    canonical_name: String,
    aliases: BTreeSet<String>,
}

/// A deterministic collection of uniquely identified parameter slots.
///
/// Entries retain insertion order for runtime iteration. Registering the same slot under another
/// name records a tied-state alias instead of duplicating it; the lexicographically smallest path
/// becomes its canonical checkpoint name.
#[derive(Debug, Default)]
pub struct ParameterStore {
    entries: Vec<StoreEntry>,
    by_id: HashMap<ParamId, usize>,
    by_name: HashMap<String, ParamId>,
}

impl ParameterStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the number of unique slots.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether this store has no slots.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Register a slot at a stable hierarchical path.
    ///
    /// Re-registering the same identity under another name records an alias. Reusing a name for a
    /// different identity is rejected so persisted state is never ambiguous.
    pub fn insert(&mut self, name: impl Into<String>, slot: ParameterSlot) -> Result<ParamId> {
        let name = name.into();
        validate_name(&name)?;
        let id = slot.id();

        if let Some(existing_id) = self.by_name.get(&name).copied() {
            if existing_id == id {
                return Ok(id);
            }
            return Err(TynxError::TypeMismatch(format!(
                "parameter name '{name}' is already bound to slot {}",
                existing_id.get()
            )));
        }

        if let Some(index) = self.by_id.get(&id).copied() {
            let entry = &mut self.entries[index];
            if name < entry.canonical_name {
                entry.aliases.insert(entry.canonical_name.clone());
                entry.canonical_name = name.clone();
                entry.slot.set_name(name.clone());
            } else {
                entry.aliases.insert(name.clone());
            }
            self.by_name.insert(name, id);
            return Ok(id);
        }

        slot.set_name(name.clone());
        let index = self.entries.len();
        self.entries.push(StoreEntry {
            slot,
            canonical_name: name.clone(),
            aliases: BTreeSet::new(),
        });
        self.by_id.insert(id, index);
        self.by_name.insert(name, id);
        Ok(id)
    }

    /// Return a slot by process-local identity.
    pub fn get(&self, id: ParamId) -> Option<&ParameterSlot> {
        self.by_id.get(&id).map(|index| &self.entries[*index].slot)
    }

    /// Return a slot by canonical name or alias.
    pub fn get_by_name(&self, name: &str) -> Option<&ParameterSlot> {
        self.by_name.get(name).and_then(|id| self.get(*id))
    }

    /// Return the stable identity registered for a canonical name or alias.
    pub fn id_by_name(&self, name: &str) -> Option<ParamId> {
        self.by_name.get(name).copied()
    }

    /// Return the canonical checkpoint/state name for an identity.
    pub fn canonical_name(&self, id: ParamId) -> Option<&str> {
        self.by_id
            .get(&id)
            .map(|index| self.entries[*index].canonical_name.as_str())
    }

    /// Iterate tied aliases for an identity in lexical order.
    pub fn aliases(&self, id: ParamId) -> impl Iterator<Item = &str> {
        self.by_id
            .get(&id)
            .into_iter()
            .flat_map(|index| self.entries[*index].aliases.iter().map(String::as_str))
    }

    /// Iterate unique slots in deterministic insertion order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &ParameterSlot> {
        self.entries.iter().map(|entry| &entry.slot)
    }

    /// Iterate canonical names and unique slots in deterministic insertion order.
    pub fn named(&self) -> impl ExactSizeIterator<Item = (&str, &ParameterSlot)> {
        self.entries
            .iter()
            .map(|entry| (entry.canonical_name.as_str(), &entry.slot))
    }

    /// Iterate unique slots that are currently trainable.
    pub fn trainable(&self) -> impl Iterator<Item = &ParameterSlot> {
        self.iter().filter(|slot| slot.contract().trainable())
    }

    /// Clear accumulated gradients from every unique slot.
    pub fn zero_grad(&self) {
        for slot in self.iter() {
            slot.zero_grad();
        }
    }
}

fn validate_name(name: &str) -> Result<()> {
    if name.trim().is_empty() {
        return Err(TynxError::TypeMismatch(
            "parameter state name cannot be empty".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, TensorData};
    use tynx_core::DynTensor;

    use super::*;

    fn slot(value: f32, trainable: bool, device: &Device) -> ParameterSlot {
        let tensor = DynTensor::from_data(TensorData::new(vec![value], [1]), 1, device).unwrap();
        ParameterSlot::new(None, tensor, trainable).unwrap()
    }

    #[test]
    fn retains_unique_slots_in_insertion_order() {
        let device = Device::autodiff(Device::default());
        let first = slot(1.0, true, &device);
        let second = slot(2.0, true, &device);
        let mut store = ParameterStore::new();

        store.insert("layer2.weight", first.clone()).unwrap();
        store.insert("layer1.weight", second.clone()).unwrap();

        assert_eq!(store.len(), 2);
        assert_eq!(
            store.iter().map(ParameterSlot::id).collect::<Vec<_>>(),
            [first.id(), second.id()]
        );
        assert_eq!(
            store.named().map(|(name, _)| name).collect::<Vec<_>>(),
            ["layer2.weight", "layer1.weight"]
        );
    }

    #[test]
    fn tied_paths_select_the_lexical_canonical_name() {
        let device = Device::autodiff(Device::default());
        let tied = slot(1.0, true, &device);
        let mut store = ParameterStore::new();

        store.insert("z_head.weight", tied.clone()).unwrap();
        store.insert("a_shared.weight", tied.clone()).unwrap();
        store.insert("m_alias.weight", tied.clone()).unwrap();

        assert_eq!(store.len(), 1);
        assert_eq!(store.canonical_name(tied.id()), Some("a_shared.weight"));
        assert_eq!(tied.name().as_deref(), Some("a_shared.weight"));
        assert_eq!(
            store.aliases(tied.id()).collect::<Vec<_>>(),
            ["m_alias.weight", "z_head.weight"]
        );
        assert_eq!(store.get_by_name("z_head.weight").unwrap().id(), tied.id());
        assert_eq!(store.id_by_name("m_alias.weight"), Some(tied.id()));
    }

    #[test]
    fn registering_the_same_path_and_slot_is_idempotent() {
        let device = Device::autodiff(Device::default());
        let weight = slot(1.0, true, &device);
        let mut store = ParameterStore::new();

        store.insert("weight", weight.clone()).unwrap();
        store.insert("weight", weight.clone()).unwrap();

        assert_eq!(store.len(), 1);
        assert_eq!(store.aliases(weight.id()).count(), 0);
    }

    #[test]
    fn rejects_a_name_shared_by_distinct_slots() {
        let device = Device::autodiff(Device::default());
        let first = slot(1.0, true, &device);
        let second = slot(2.0, true, &device);
        let mut store = ParameterStore::new();

        store.insert("weight", first.clone()).unwrap();
        let error = store.insert("weight", second).unwrap_err();

        assert!(error.to_string().contains("already bound"));
        assert_eq!(store.len(), 1);
        assert_eq!(store.get_by_name("weight").unwrap().id(), first.id());
    }

    #[test]
    fn trainable_iteration_skips_frozen_slots_and_aliases() {
        let device = Device::autodiff(Device::default());
        let weight = slot(1.0, true, &device);
        let frozen = slot(2.0, false, &device);
        let mut store = ParameterStore::new();

        store.insert("weight", weight.clone()).unwrap();
        store.insert("tied_weight", weight.clone()).unwrap();
        store.insert("running_mean", frozen).unwrap();

        assert_eq!(
            store.trainable().map(ParameterSlot::id).collect::<Vec<_>>(),
            [weight.id()]
        );
    }

    #[test]
    fn rejects_empty_persisted_names() {
        let device = Device::autodiff(Device::default());
        let mut store = ParameterStore::new();

        let error = store.insert("  ", slot(1.0, true, &device)).unwrap_err();

        assert!(error.to_string().contains("cannot be empty"));
        assert!(store.is_empty());
    }
}
