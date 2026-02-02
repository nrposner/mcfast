//! MergeForest: A high-performance merge tree data structure for McFACTS
//!
//! Reads galaxy output files and constructs a forest of binary merge trees
//! representing black hole merger histories.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use glob::glob;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

/// Null UUID constant - represents "no parent"
const NULL_UUID: &str = "00000000-0000-0000-0000-000000000000";

/// A single node in the merge tree
#[derive(Debug, Clone)]
struct MergeNode {
    /// Indices of parent nodes (what merged to create this BH)
    /// None if this is a leaf node (initial BH, gen 1)
    parents: Option<(usize, usize)>,
    /// Index of child node (what this BH merged into)
    /// None if this is a root node (final product)
    child: Option<usize>,
}

/// Raw record from CSV files
#[derive(Debug, Clone)]
struct MergeRecord {
    unique_id: String,
    parent_1: Option<String>,
    parent_2: Option<String>,
}

/// A forest of binary merge trees representing black hole merger histories
#[pyclass(module = "mcfast")]
pub struct MergeForest {
    /// All nodes in the forest, indexed by their position
    nodes: Vec<MergeNode>,
    /// Map from UUID string to node index
    uuid_to_index: HashMap<String, usize>,
    /// Map from node index back to UUID string
    index_to_uuid: Vec<String>,
    /// Indices of root nodes (final products - no children)
    roots: Vec<usize>,
    /// Indices of leaf nodes (initial BHs - no parents)
    leaves: Vec<usize>,
    /// Indices of singleton nodes (both root and leaf - never merged)
    singletons: Vec<usize>,
}

impl MergeForest {
    /// Strip the `::type` suffix from column names
    /// e.g., "unique_id::uuid.UUID" -> "unique_id"
    fn strip_type_suffix(col_name: &str) -> &str {
        col_name.split("::").next().unwrap_or(col_name)
    }

    /// Check if a UUID string represents a null/missing parent
    fn is_null_uuid(uuid: &str) -> bool {
        uuid == NULL_UUID || uuid.is_empty()
    }

    /// Read all matching files from a directory and parse merge records
    fn read_files(directory: &str, pattern: &str) -> PyResult<Vec<MergeRecord>> {
        let full_pattern = format!("{}/{}", directory.trim_end_matches('/'), pattern);

        // Collect all matching file paths
        let file_paths: Vec<_> = glob(&full_pattern)
            .map_err(|e| PyValueError::new_err(format!("Invalid glob pattern: {}", e)))?
            .filter_map(|entry| entry.ok())
            .collect();

        if file_paths.is_empty() {
            return Err(PyValueError::new_err(format!(
                "No files found matching pattern: {}",
                full_pattern
            )));
        }

        // Read files in parallel using Rayon
        let results: Vec<PyResult<Vec<MergeRecord>>> = file_paths
            .par_iter()
            .map(|path| Self::read_single_file(path))
            .collect();

        // Flatten results, propagating any errors
        let mut all_records = Vec::new();
        for result in results {
            all_records.extend(result?);
        }

        Ok(all_records)
    }

    /// Read a single whitespace-separated file and extract merge records
    fn read_single_file(path: &Path) -> PyResult<Vec<MergeRecord>> {
        let content = fs::read_to_string(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read file {:?}: {}", path, e))
        })?;

        let mut lines = content.lines();

        // Parse header line
        let header_line = match lines.next() {
            Some(line) => line,
            None => return Ok(Vec::new()), // Empty file
        };

        let headers: Vec<&str> = header_line.split_whitespace().collect();

        // Find column indices for the columns we need
        let unique_id_idx = headers
            .iter()
            .position(|h| Self::strip_type_suffix(h) == "unique_id");
        let parent_1_idx = headers
            .iter()
            .position(|h| Self::strip_type_suffix(h) == "parent_unique_id_1");
        let parent_2_idx = headers
            .iter()
            .position(|h| Self::strip_type_suffix(h) == "parent_unique_id_2");

        // Skip files that don't have the required columns
        let unique_id_idx = match unique_id_idx {
            Some(idx) => idx,
            None => return Ok(Vec::new()),
        };

        // Parent columns are optional - files without them just don't contribute
        let (parent_1_idx, parent_2_idx) = match (parent_1_idx, parent_2_idx) {
            (Some(p1), Some(p2)) => (p1, p2),
            _ => return Ok(Vec::new()),
        };

        // Parse data lines
        let mut records = Vec::new();

        for line in lines {
            let fields: Vec<&str> = line.split_whitespace().collect();

            // Skip malformed lines
            let max_idx = unique_id_idx.max(parent_1_idx).max(parent_2_idx);
            if fields.len() <= max_idx {
                continue;
            }

            let unique_id = fields[unique_id_idx].to_string();
            let parent_1_raw = fields[parent_1_idx];
            let parent_2_raw = fields[parent_2_idx];

            let parent_1 = if Self::is_null_uuid(parent_1_raw) {
                None
            } else {
                Some(parent_1_raw.to_string())
            };

            let parent_2 = if Self::is_null_uuid(parent_2_raw) {
                None
            } else {
                Some(parent_2_raw.to_string())
            };

            records.push(MergeRecord {
                unique_id,
                parent_1,
                parent_2,
            });
        }

        Ok(records)
    }

    /// Build the forest from raw records
    fn build_from_records(records: Vec<MergeRecord>) -> PyResult<Self> {
        // Step 1: Deduplicate by unique_id (same BH appears in multiple snapshots)
        // Keep the first occurrence with actual parent info if available
        let mut seen: HashMap<String, MergeRecord> = HashMap::new();
        for record in records {
            seen.entry(record.unique_id.clone())
                .and_modify(|existing| {
                    // Prefer records that have parent information
                    if existing.parent_1.is_none() && record.parent_1.is_some() {
                        *existing = record.clone();
                    }
                })
                .or_insert(record);
        }

        // Step 2: Collect all unique UUIDs (including parents that might not have their own record)
        let mut all_uuids: Vec<String> = Vec::new();
        let mut uuid_set: HashSet<String> = HashSet::new();

        for record in seen.values() {
            if uuid_set.insert(record.unique_id.clone()) {
                all_uuids.push(record.unique_id.clone());
            }
            if let Some(ref p1) = record.parent_1 {
                if uuid_set.insert(p1.clone()) {
                    all_uuids.push(p1.clone());
                }
            }
            if let Some(ref p2) = record.parent_2 {
                if uuid_set.insert(p2.clone()) {
                    all_uuids.push(p2.clone());
                }
            }
        }

        // Step 3: Build uuid_to_index and index_to_uuid mappings
        let uuid_to_index: HashMap<String, usize> = all_uuids
            .iter()
            .enumerate()
            .map(|(i, uuid)| (uuid.clone(), i))
            .collect();
        let index_to_uuid = all_uuids;

        // Step 4: Create nodes with parent links
        let mut nodes: Vec<MergeNode> = (0..index_to_uuid.len())
            .map(|_| MergeNode {
                parents: None,
                child: None,
            })
            .collect();

        for record in seen.values() {
            let node_idx = uuid_to_index[&record.unique_id];

            match (&record.parent_1, &record.parent_2) {
                (Some(p1), Some(p2)) => {
                    let p1_idx = uuid_to_index[p1];
                    let p2_idx = uuid_to_index[p2];
                    nodes[node_idx].parents = Some((p1_idx, p2_idx));
                }
                (Some(p1), None) | (None, Some(p1)) => {
                    // Single parent case - shouldn't happen in binary mergers
                    // but handle gracefully by duplicating
                    let p1_idx = uuid_to_index[p1];
                    nodes[node_idx].parents = Some((p1_idx, p1_idx));
                }
                (None, None) => {
                    // Leaf node - no parents
                }
            }
        }

        // Step 5: Compute child links by inverting parent relationships
        for child_idx in 0..nodes.len() {
            if let Some((p1_idx, p2_idx)) = nodes[child_idx].parents {
                // Both parents point to this child
                nodes[p1_idx].child = Some(child_idx);
                if p1_idx != p2_idx {
                    nodes[p2_idx].child = Some(child_idx);
                }
            }
        }

        // Step 6: Identify roots, leaves, and singletons
        let mut roots = Vec::new();
        let mut leaves = Vec::new();
        let mut singletons = Vec::new();

        for (idx, node) in nodes.iter().enumerate() {
            let is_root = node.child.is_none();
            let is_leaf = node.parents.is_none();

            if is_root && is_leaf {
                singletons.push(idx);
            } else if is_root {
                roots.push(idx);
            } else if is_leaf {
                leaves.push(idx);
            }
            // Internal nodes are neither root nor leaf
        }

        Ok(MergeForest {
            nodes,
            uuid_to_index,
            index_to_uuid,
            roots,
            leaves,
            singletons,
        })
    }
}

#[pymethods]
impl MergeForest {
    /// Get all ancestors (full merge tree) for a given UUID
    ///
    /// Returns all UUIDs that contributed to creating this black hole,
    /// traversing the entire merge tree backwards.
    ///
    /// Args:
    ///     uuid: The UUID of the black hole to trace ancestry for
    ///
    /// Returns:
    ///     List of ancestor UUIDs (not including the input UUID)
    pub fn get_ancestors(&self, uuid: &str) -> PyResult<Vec<(String, String)>> {

        let start_idx = self
            .uuid_to_index
            .get(uuid)
            .ok_or_else(|| PyValueError::new_err(format!("UUID not found: {}", uuid)))?;

        let mut ancestors = Vec::new();
        let mut stack = Vec::new();
        let mut visited = HashSet::new();

        // Start with the parents of the given node
        if let Some((p1, p2)) = self.nodes[*start_idx].parents {
            stack.push(("A".to_string(), p1));
            if p1 != p2 {
                stack.push(("B".to_string(), p2));
            }
        }

        fn numbering_helper(input: &str) -> (String, String) { 
            let outputs = if input.ends_with('A') 
            || input.ends_with('B')  {
                ("1", "2")
            } else {
                ("A", "B")
            };

            (input.to_owned() + outputs.0, input.to_owned() + outputs.1)
        }

        ancestors.push(('P'.to_string(), uuid.to_string()));

        // DFS traversal of all ancestors with cycle detection
        while let Some((val, idx)) = stack.pop() {
            if !visited.insert(idx) {
                continue; // Skip already visited nodes
            }

            ancestors.push((val.to_owned(), self.index_to_uuid[idx].clone()));

            if let Some((p1, p2)) = self.nodes[idx].parents {

                let outputs = numbering_helper(&val);

                stack.push((outputs.0, p1));
                if p1 != p2 {
                    stack.push((outputs.1, p2));
                }
            }
        }

        Ok(ancestors)
    }

    /// Get the descendant (child) of a black hole
    ///
    /// Returns the UUID of the black hole that this one merged into,
    /// or None if this is a final product (root node).
    ///
    /// Args:
    ///     uuid: The UUID of the black hole
    ///
    /// Returns:
    ///     UUID of the descendant, or None if this is a root
    pub fn get_descendant(&self, uuid: &str) -> PyResult<Option<String>> {
        let idx = self
            .uuid_to_index
            .get(uuid)
            .ok_or_else(|| PyValueError::new_err(format!("UUID not found: {}", uuid)))?;

        Ok(self.nodes[*idx]
            .child
            .map(|c| self.index_to_uuid[c].clone()))
    }

    /// Get the direct parents of a black hole
    ///
    /// Returns the UUIDs of the two black holes that merged to create this one,
    /// or None if this is a leaf node (initial black hole).
    ///
    /// Args:
    ///     uuid: The UUID of the black hole
    ///
    /// Returns:
    ///     Tuple of (parent1_uuid, parent2_uuid) or None
    pub fn get_parents(&self, uuid: &str) -> PyResult<Option<(String, String)>> {
        let idx = self
            .uuid_to_index
            .get(uuid)
            .ok_or_else(|| PyValueError::new_err(format!("UUID not found: {}", uuid)))?;

        Ok(self.nodes[*idx].parents.map(|(p1, p2)| {
            (
                self.index_to_uuid[p1].clone(),
                self.index_to_uuid[p2].clone(),
            )
        }))
    }

    /// Get the full lineage to the final product
    ///
    /// Returns a list of UUIDs tracing from this black hole
    /// to its final merger product (root).
    ///
    /// Args:
    ///     uuid: The UUID to trace forward from
    ///
    /// Returns:
    ///     List of UUIDs from this node to the root (inclusive)
    pub fn get_lineage_to_root(&self, uuid: &str) -> PyResult<Vec<String>> {
        let mut idx = *self
            .uuid_to_index
            .get(uuid)
            .ok_or_else(|| PyValueError::new_err(format!("UUID not found: {}", uuid)))?;

        let mut lineage = vec![self.index_to_uuid[idx].clone()];

        while let Some(child_idx) = self.nodes[idx].child {
            lineage.push(self.index_to_uuid[child_idx].clone());
            idx = child_idx;
        }

        Ok(lineage)
    }

    /// Get the generation (depth from leaves) of a black hole
    ///
    /// Generation 0 = initial BH (leaf), Gen 1 = product of two Gen 0 BHs, etc.
    ///
    /// Args:
    ///     uuid: The UUID to query
    ///
    /// Returns:
    ///     Generation number (0 for leaves)
    pub fn get_generation(&self, uuid: &str) -> PyResult<usize> {
        let idx = *self
            .uuid_to_index
            .get(uuid)
            .ok_or_else(|| PyValueError::new_err(format!("UUID not found: {}", uuid)))?;

        // Iterative approach with memoization
        fn compute_gen(
            nodes: &[MergeNode],
            idx: usize,
            cache: &mut HashMap<usize, usize>,
        ) -> usize {
            if let Some(&cached) = cache.get(&idx) {
                return cached;
            }

            let gen = match nodes[idx].parents {
                None => 0, // Leaf node
                Some((p1, p2)) => {
                    let g1 = compute_gen(nodes, p1, cache);
                    let g2 = compute_gen(nodes, p2, cache);
                    g1.max(g2)
                }
            };

            cache.insert(idx, gen);
            gen
        }

        let mut cache = HashMap::new();
        Ok(compute_gen(&self.nodes, idx, &mut cache))
    }

    /// Get all root nodes (final merger products with no children)
    ///
    /// Returns:
    ///     List of UUIDs for all root nodes
    pub fn roots(&self) -> Vec<String> {
        self.roots
            .iter()
            .map(|&idx| self.index_to_uuid[idx].clone())
            .collect()
    }

    /// Get all leaf nodes (initial black holes with no parents)
    ///
    /// Returns:
    ///     List of UUIDs for all leaf nodes
    pub fn leaves(&self) -> Vec<String> {
        self.leaves
            .iter()
            .map(|&idx| self.index_to_uuid[idx].clone())
            .collect()
    }

    /// Get all singleton nodes (black holes that never merged)
    ///
    /// These are nodes that are both roots AND leaves - they existed
    /// but never participated in any merger.
    ///
    /// Returns:
    ///     List of UUIDs for all singleton nodes
    pub fn singletons(&self) -> Vec<String> {
        self.singletons
            .iter()
            .map(|&idx| self.index_to_uuid[idx].clone())
            .collect()
    }

    /// Check if a UUID exists in the forest
    ///
    /// Args:
    ///     uuid: The UUID to check
    ///
    /// Returns:
    ///     True if the UUID exists, False otherwise
    pub fn contains(&self, uuid: &str) -> bool {
        self.uuid_to_index.contains_key(uuid)
    }

    /// Get the total number of nodes in the forest
    ///
    /// Returns:
    ///     Total node count
    pub fn __len__(&self) -> usize {
        self.nodes.len()
    }

    /// String representation for debugging
    pub fn __repr__(&self) -> String {
        format!(
            "MergeForest(nodes={}, roots={}, leaves={}, singletons={})",
            self.nodes.len(),
            self.roots.len(),
            self.leaves.len(),
            self.singletons.len()
        )
    }

    /// Pickle support: returns empty args for __new__ (state handled by __getstate__)
    pub fn __getnewargs_ex__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        use pyo3::types::{PyDict, PyTuple};
        // Return empty args and empty kwargs - actual data comes from __getstate__
        Ok((
            PyTuple::empty(py).into_any(),
            PyDict::new(py).into_any(),
        ))
    }

    #[allow(clippy::complexity)]
    /// Pickle serialization support - returns state as a Python dict
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::types::PyDict;

        let state = PyDict::new(py);

        // Serialize index_to_uuid (uuid_to_index can be reconstructed)
        state.set_item("index_to_uuid", &self.index_to_uuid)?;

        // Serialize nodes as list of ((p1, p2) or None, child or None)
        let nodes_data: Vec<(Option<(usize, usize)>, Option<usize>)> = self
            .nodes
            .iter()
            .map(|n| (n.parents, n.child))
            .collect();
        state.set_item("nodes", nodes_data)?;

        // Serialize category lists
        state.set_item("roots", &self.roots)?;
        state.set_item("leaves", &self.leaves)?;
        state.set_item("singletons", &self.singletons)?;

        Ok(state.into_any())
    }

    /// Pickle deserialization support - reconstructs from Python dict
    #[allow(clippy::complexity)]
    pub fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        use pyo3::types::PyDict;

        let state: &Bound<'_, PyDict> = state.cast()?;

        // Restore index_to_uuid
        let index_to_uuid: Vec<String> = state
            .get_item("index_to_uuid")?
            .ok_or_else(|| PyValueError::new_err("Missing 'index_to_uuid' in pickle state"))?
            .extract()?;

        // Rebuild uuid_to_index from index_to_uuid
        let uuid_to_index: HashMap<String, usize> = index_to_uuid
            .iter()
            .enumerate()
            .map(|(i, uuid)| (uuid.clone(), i))
            .collect();

        // Restore nodes
        let nodes_data: Vec<(Option<(usize, usize)>, Option<usize>)> = state
            .get_item("nodes")?
            .ok_or_else(|| PyValueError::new_err("Missing 'nodes' in pickle state"))?
            .extract()?;

        let nodes: Vec<MergeNode> = nodes_data
            .into_iter()
            .map(|(parents, child)| MergeNode { parents, child })
            .collect();

        // Restore category lists
        let roots: Vec<usize> = state
            .get_item("roots")?
            .ok_or_else(|| PyValueError::new_err("Missing 'roots' in pickle state"))?
            .extract()?;

        let leaves: Vec<usize> = state
            .get_item("leaves")?
            .ok_or_else(|| PyValueError::new_err("Missing 'leaves' in pickle state"))?
            .extract()?;

        let singletons: Vec<usize> = state
            .get_item("singletons")?
            .ok_or_else(|| PyValueError::new_err("Missing 'singletons' in pickle state"))?
            .extract()?;

        // Update self
        self.nodes = nodes;
        self.uuid_to_index = uuid_to_index;
        self.index_to_uuid = index_to_uuid;
        self.roots = roots;
        self.leaves = leaves;
        self.singletons = singletons;

        Ok(())
    }

    /// Required for pickle: creates an uninitialized instance for __setstate__
    #[new]
    #[pyo3(signature = (directory=None, pattern=None))]
    pub fn py_new(directory: Option<&str>, pattern: Option<&str>) -> PyResult<Self> {
        match (directory, pattern) {
            (Some(dir), Some(pat)) => {
                let records = Self::read_files(dir, pat)?;
                if records.is_empty() {
                    return Err(PyValueError::new_err(
                        "No merge records found in the specified files",
                    ));
                }
                Self::build_from_records(records)
            }
            (None, None) => {
                // Return empty instance for pickle's __setstate__
                Ok(MergeForest {
                    nodes: Vec::new(),
                    uuid_to_index: HashMap::new(),
                    index_to_uuid: Vec::new(),
                    roots: Vec::new(),
                    leaves: Vec::new(),
                    singletons: Vec::new(),
                })
            }
            _ => Err(PyValueError::new_err(
                "Both 'directory' and 'pattern' must be provided, or neither (for pickle)",
            )),
        }
    }
}
