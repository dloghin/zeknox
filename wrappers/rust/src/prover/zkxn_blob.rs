// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Parse the native `gpu_prove` ZKXN binary blob and build a Plonky2 [`ProofWithPublicInputs`].
//!
//! Layout matches `native/prover/prover.cu` (`kProofMagic`, `kProofVersion`): header, three Merkle
//! caps (flat `u64` limbs), then a flat opening list (`2` base-field limbs per extension sample).
//!
//! A [`plonky2::fri::proof::FriProof`] cannot be recovered from the blob (FRI is not serialized
//! there). Callers must supply an [`FriProof`] generated with the **same** Fiat–Shamir transcript
//! as the caps and openings in the blob—typically by running the Plonky2 CPU prover, or in the
//! future by extending the native prover.
//!
//! [`verified_proof_matching_zkxn_blob`] compares the blob to a CPU-generated proof and returns
//! that proof when every field agrees, so [`plonky2::plonk::verifier::verify`] succeeds without
//! trusting the blob alone for FRI.

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::fri::proof::FriProof;
use plonky2::hash::hash_types::HashOut;
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::plonk::config::{GenericConfig, Hasher};
use plonky2::plonk::proof::{OpeningSet, Proof, ProofWithPublicInputs};

use super::{ProverConfig, GPU_PROOF_MAGIC};

const ZKXN_PROOF_VERSION: u32 = 1;
const ZKXN_SECTIONS: u32 = 1;
const HASH_OUT_LIMBS: usize = 4;

/// Errors from parsing or assembling a ZKXN GPU proof blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZkxnBlobError {
    TooShort {
        need: usize,
        got: usize,
    },
    BadMagic {
        got: u32,
    },
    BadVersion {
        got: u32,
    },
    BadSections {
        got: u32,
    },
    CapLimbCount {
        which: &'static str,
        limbs: u64,
    },
    OpeningsLimbCount {
        expected: usize,
        got: usize,
    },
    OpeningsCountOverflow,
    BlobTrailingBytes {
        extra: usize,
    },
    /// Native `gpu_prove` packs Z / partial products / quotients differently than Plonky2’s
    /// [`OpeningSet`] (see `validate_proof_with_pis_shape`). Until the GPU pipeline matches, you
    /// cannot turn the blob openings alone into a Plonky2-verifiable proof.
    BlobIncompatiblePlonky2 {
        gpu_opening_u64_limbs: usize,
        plonky2_opening_u64_limbs: usize,
    },
    CpuBlobMismatch {
        detail: &'static str,
    },
}

impl core::fmt::Display for ZkxnBlobError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ZkxnBlobError::TooShort { need, got } => {
                write!(f, "ZKXN blob too short: need {} bytes, got {}", need, got)
            }
            ZkxnBlobError::BadMagic { got } => {
                write!(f, "ZKXN blob bad magic: expected {:#x}, got {:#x}", GPU_PROOF_MAGIC, got)
            }
            ZkxnBlobError::BadVersion { got } => {
                write!(f, "ZKXN blob bad version: expected {}, got {}", ZKXN_PROOF_VERSION, got)
            }
            ZkxnBlobError::BadSections { got } => {
                write!(f, "ZKXN blob bad sections: expected {}, got {}", ZKXN_SECTIONS, got)
            }
            ZkxnBlobError::CapLimbCount { which, limbs } => write!(
                f,
                "ZKXN blob {} cap limb count {} is not a multiple of {}",
                which, limbs, HASH_OUT_LIMBS
            ),
            ZkxnBlobError::OpeningsLimbCount { expected, got } => write!(
                f,
                "ZKXN blob openings: expected {} u64 limbs, got {}",
                expected, got
            ),
            ZkxnBlobError::OpeningsCountOverflow => {
                write!(f, "ZKXN blob openings count does not fit in usize")
            }
            ZkxnBlobError::BlobIncompatiblePlonky2 {
                gpu_opening_u64_limbs,
                plonky2_opening_u64_limbs,
            } => write!(
                f,
                "ZKXN blob opening layout is not Plonky2-shaped: GPU uses {} u64 limbs, Plonky2 common data expects {}",
                gpu_opening_u64_limbs, plonky2_opening_u64_limbs
            ),
            ZkxnBlobError::BlobTrailingBytes { extra } => {
                write!(f, "ZKXN blob has {} trailing bytes after openings", extra)
            }
            ZkxnBlobError::CpuBlobMismatch { detail } => {
                write!(f, "ZKXN blob does not match CPU Plonky2 proof: {}", detail)
            }
        }
    }
}

impl std::error::Error for ZkxnBlobError {}

fn read_u32(blob: &[u8], off: &mut usize) -> Result<u32, ZkxnBlobError> {
    if *off + 4 > blob.len() {
        return Err(ZkxnBlobError::TooShort {
            need: *off + 4,
            got: blob.len(),
        });
    }
    let v = u32::from_le_bytes(blob[*off..*off + 4].try_into().unwrap());
    *off += 4;
    Ok(v)
}

fn read_u64(blob: &[u8], off: &mut usize) -> Result<u64, ZkxnBlobError> {
    if *off + 8 > blob.len() {
        return Err(ZkxnBlobError::TooShort {
            need: *off + 8,
            got: blob.len(),
        });
    }
    let v = u64::from_le_bytes(blob[*off..*off + 8].try_into().unwrap());
    *off += 8;
    Ok(v)
}

fn read_merkle_cap<H: Hasher<GoldilocksField, Hash = HashOut<GoldilocksField>>>(
    blob: &[u8],
    off: &mut usize,
    cap_name: &'static str,
) -> Result<MerkleCap<GoldilocksField, H>, ZkxnBlobError> {
    let n_limbs_u64 = read_u64(blob, off)?;
    let n_limbs = usize::try_from(n_limbs_u64).map_err(|_| ZkxnBlobError::CapLimbCount {
        which: cap_name,
        limbs: n_limbs_u64,
    })?;
    if n_limbs % HASH_OUT_LIMBS != 0 {
        return Err(ZkxnBlobError::CapLimbCount {
            which: cap_name,
            limbs: n_limbs_u64,
        });
    }
    let n_hashes = n_limbs / HASH_OUT_LIMBS;
    let mut v = Vec::with_capacity(n_hashes);
    for _ in 0..n_hashes {
        let e0 = read_u64(blob, off)?;
        let e1 = read_u64(blob, off)?;
        let e2 = read_u64(blob, off)?;
        let e3 = read_u64(blob, off)?;
        v.push(HashOut {
            elements: [
                GoldilocksField::from_canonical_u64(e0),
                GoldilocksField::from_canonical_u64(e1),
                GoldilocksField::from_canonical_u64(e2),
                GoldilocksField::from_canonical_u64(e3),
            ],
        });
    }
    Ok(MerkleCap(v))
}

/// Number of `u64` limbs in the openings section implied by Plonky2 [`CommonCircuitData`] / shape
/// validation (two base-field elements per quadratic extension sample).
pub fn zkxn_opening_limbs_for_plonky2_common(
    common: &CommonCircuitData<GoldilocksField, 2>,
) -> usize {
    let n_constants = common.num_constants;
    let n_sigmas = common.config.num_routed_wires;
    let n_wires = common.config.num_wires;
    let n_ch = common.config.num_challenges;
    let n_zs = n_ch;
    let n_zs_next = n_ch;
    let n_partial = common.partial_products_range().count();
    let n_quotient = common.config.num_challenges * common.quotient_degree_factor;
    let n_ext = n_constants + n_sigmas + n_wires + n_zs + n_zs_next + n_partial + n_quotient;
    2 * n_ext
}

/// Opening flat size produced by native `gpu_prove` (`native/prover/prover.cu`): one Z column,
/// `num_chunks` partial-product columns, and `max(1, quotient_degree_factor)` quotient columns in
/// the opening set (not the same as Plonky2’s multi-challenge layout).
pub fn zkxn_opening_limbs_for_gpu_prover_config(config: &ProverConfig) -> usize {
    let n_c = config.num_constants as usize;
    let n_s = config.num_routed_wires as usize;
    let n_w = config.num_wires as usize;
    let qfac = config.quotient_degree_factor as usize;
    let num_chunks = if qfac == 0 {
        0
    } else {
        (n_s + qfac - 1) / qfac
    };
    let n_z = 1usize;
    let n_zn = 1usize;
    let n_q = (config.num_challenges as usize) * (config.quotient_degree_factor.max(1) as usize);
    let n_ext = n_c + n_s + n_w + n_z + n_zn + num_chunks + n_q;
    2 * n_ext
}

fn opening_set_from_flat_plonky2(
    common: &CommonCircuitData<GoldilocksField, 2>,
    flat: &[u64],
) -> Result<OpeningSet<GoldilocksField, 2>, ZkxnBlobError> {
    let need = zkxn_opening_limbs_for_plonky2_common(common);
    if flat.len() != need {
        return Err(ZkxnBlobError::OpeningsLimbCount {
            expected: need,
            got: flat.len(),
        });
    }

    type EF = <GoldilocksField as Extendable<2>>::Extension;

    let mut idx = 0usize;
    let mut next_ext = || -> EF {
        let lo = flat[idx];
        let hi = flat[idx + 1];
        idx += 2;
        EF::from_basefield_array([
            GoldilocksField::from_canonical_u64(lo),
            GoldilocksField::from_canonical_u64(hi),
        ])
    };

    let mut take_n = |n: usize| -> Vec<EF> { (0..n).map(|_| next_ext()).collect() };

    let constants = take_n(common.num_constants);
    let plonk_sigmas = take_n(common.config.num_routed_wires);
    let wires = take_n(common.config.num_wires);
    let plonk_zs = take_n(common.config.num_challenges);
    let plonk_zs_next = take_n(common.config.num_challenges);
    let partial_products = take_n(common.partial_products_range().count());
    let quotient_polys = take_n(common.config.num_challenges * common.quotient_degree_factor);

    debug_assert_eq!(idx, flat.len());

    Ok(OpeningSet {
        constants,
        plonk_sigmas,
        wires,
        plonk_zs,
        plonk_zs_next,
        partial_products,
        quotient_polys,
    })
}

/// Parsed ZKXN blob: Merkle caps and raw opening limbs (native `gpu_prove` layout).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedZkxnBlob<H: Hasher<GoldilocksField, Hash = HashOut<GoldilocksField>>> {
    pub wires_cap: MerkleCap<GoldilocksField, H>,
    pub plonk_zs_partial_products_cap: MerkleCap<GoldilocksField, H>,
    pub quotient_polys_cap: MerkleCap<GoldilocksField, H>,
    pub openings_flat_u64: Vec<u64>,
}

impl<H: Hasher<GoldilocksField, Hash = HashOut<GoldilocksField>>> ParsedZkxnBlob<H> {
    /// Decode openings using `common`’s polynomial counts (must match the circuit used for `gpu_prove`).
    pub fn opening_set(
        &self,
        common: &CommonCircuitData<GoldilocksField, 2>,
    ) -> Result<OpeningSet<GoldilocksField, 2>, ZkxnBlobError> {
        opening_set_from_flat_plonky2(common, &self.openings_flat_u64)
    }
}

/// Parse a `gpu_prove` output buffer into caps and opening limbs.
pub fn parse_zkxn_gpu_blob<H: Hasher<GoldilocksField, Hash = HashOut<GoldilocksField>>>(
    blob: &[u8],
) -> Result<ParsedZkxnBlob<H>, ZkxnBlobError> {
    let mut off = 0usize;
    let magic = read_u32(blob, &mut off)?;
    if magic != GPU_PROOF_MAGIC {
        return Err(ZkxnBlobError::BadMagic { got: magic });
    }
    let version = read_u32(blob, &mut off)?;
    if version != ZKXN_PROOF_VERSION {
        return Err(ZkxnBlobError::BadVersion { got: version });
    }
    let sections = read_u32(blob, &mut off)?;
    if sections != ZKXN_SECTIONS {
        return Err(ZkxnBlobError::BadSections { got: sections });
    }

    let wires_cap = read_merkle_cap::<H>(blob, &mut off, "wires")?;
    let plonk_zs_partial_products_cap =
        read_merkle_cap::<H>(blob, &mut off, "zs_partial_products")?;
    let quotient_polys_cap = read_merkle_cap::<H>(blob, &mut off, "quotient")?;

    let n_open_limbs = read_u64(blob, &mut off)?;
    let n_open = usize::try_from(n_open_limbs).map_err(|_| ZkxnBlobError::OpeningsCountOverflow)?;
    let open_bytes = n_open
        .checked_mul(8)
        .ok_or(ZkxnBlobError::OpeningsCountOverflow)?;
    let need = off
        .checked_add(open_bytes)
        .ok_or(ZkxnBlobError::OpeningsCountOverflow)?;
    if need > blob.len() {
        return Err(ZkxnBlobError::TooShort {
            need,
            got: blob.len(),
        });
    }
    let mut openings_flat_u64 = Vec::with_capacity(n_open);
    for _ in 0..n_open {
        openings_flat_u64.push(read_u64(blob, &mut off)?);
    }

    if off != blob.len() {
        return Err(ZkxnBlobError::BlobTrailingBytes {
            extra: blob.len() - off,
        });
    }

    Ok(ParsedZkxnBlob {
        wires_cap,
        plonk_zs_partial_products_cap,
        quotient_polys_cap,
        openings_flat_u64,
    })
}

/// Assemble [`ProofWithPublicInputs`] from a blob and a CPU-generated [`FriProof`].
pub fn proof_with_public_inputs_from_zkxn_blob<C: GenericConfig<2, F = GoldilocksField>>(
    common: &CommonCircuitData<GoldilocksField, 2>,
    config: &ProverConfig,
    blob: &[u8],
    opening_proof: FriProof<GoldilocksField, C::Hasher, 2>,
    public_inputs: Vec<GoldilocksField>,
) -> Result<ProofWithPublicInputs<GoldilocksField, C, 2>, ZkxnBlobError>
where
    C::Hasher: Hasher<GoldilocksField, Hash = HashOut<GoldilocksField>>,
{
    let gpu_limbs = zkxn_opening_limbs_for_gpu_prover_config(config);
    let p2_limbs = zkxn_opening_limbs_for_plonky2_common(common);
    if gpu_limbs != p2_limbs {
        return Err(ZkxnBlobError::BlobIncompatiblePlonky2 {
            gpu_opening_u64_limbs: gpu_limbs,
            plonky2_opening_u64_limbs: p2_limbs,
        });
    }
    let parsed = parse_zkxn_gpu_blob::<C::Hasher>(blob)?;
    if parsed.openings_flat_u64.len() != p2_limbs {
        return Err(ZkxnBlobError::OpeningsLimbCount {
            expected: p2_limbs,
            got: parsed.openings_flat_u64.len(),
        });
    }
    let openings = parsed.opening_set(common)?;
    let proof = Proof {
        wires_cap: parsed.wires_cap,
        plonk_zs_partial_products_cap: parsed.plonk_zs_partial_products_cap,
        quotient_polys_cap: parsed.quotient_polys_cap,
        openings,
        opening_proof,
    };
    Ok(ProofWithPublicInputs {
        proof,
        public_inputs,
    })
}

fn proof_caps_and_openings_match<C: GenericConfig<2, F = GoldilocksField>>(
    blob_proof: &Proof<GoldilocksField, C, 2>,
    cpu: &Proof<GoldilocksField, C, 2>,
) -> Result<(), ZkxnBlobError> {
    if blob_proof.wires_cap != cpu.wires_cap {
        return Err(ZkxnBlobError::CpuBlobMismatch {
            detail: "wires_cap differs (GPU vs CPU)",
        });
    }
    if blob_proof.plonk_zs_partial_products_cap != cpu.plonk_zs_partial_products_cap {
        return Err(ZkxnBlobError::CpuBlobMismatch {
            detail: "plonk_zs_partial_products_cap differs (GPU vs CPU)",
        });
    }
    if blob_proof.quotient_polys_cap != cpu.quotient_polys_cap {
        return Err(ZkxnBlobError::CpuBlobMismatch {
            detail: "quotient_polys_cap differs: native GPU prover still commits zero quotients; \
                     wire this in `native/prover/prover.cu` to match Plonky2",
        });
    }
    if blob_proof.openings != cpu.openings {
        return Err(ZkxnBlobError::CpuBlobMismatch {
            detail: "openings differ (transcript / evaluation mismatch)",
        });
    }
    Ok(())
}

/// Run the CPU prover, parse `blob`, and return the CPU proof when caps and openings match the
/// blob. Then [`plonky2::plonk::verifier::verify`] accepts the returned value.
///
/// When the native orchestrator matches Plonky2 (quotient + transcript), this succeeds and proves
/// the GPU blob encodes the same Plonk+FRI witness as the CPU prover.
pub fn verified_proof_matching_zkxn_blob<C: GenericConfig<2, F = GoldilocksField>>(
    prover_data: &plonky2::plonk::circuit_data::ProverOnlyCircuitData<GoldilocksField, C, 2>,
    common: &CommonCircuitData<GoldilocksField, 2>,
    partition: plonky2::iop::witness::PartitionWitness<GoldilocksField>,
    blob: &[u8],
    timing: &mut plonky2::util::timing::TimingTree,
) -> Result<ProofWithPublicInputs<GoldilocksField, C, 2>, ZkxnBlobError>
where
    C::Hasher: Hasher<GoldilocksField, Hash = HashOut<GoldilocksField>>,
    C::InnerHasher: Hasher<GoldilocksField>,
{
    let cpu = plonky2::plonk::prover::prove_with_partition_witness(
        prover_data,
        common,
        partition,
        timing,
    )
    .map_err(|_| ZkxnBlobError::CpuBlobMismatch {
        detail: "CPU prove failed",
    })?;

    let cfg = prover_config_from_common(common);
    let from_blob = proof_with_public_inputs_from_zkxn_blob::<C>(
        common,
        &cfg,
        blob,
        cpu.proof.opening_proof.clone(),
        cpu.public_inputs.clone(),
    )?;

    proof_caps_and_openings_match::<C>(&from_blob.proof, &cpu.proof)?;
    Ok(cpu)
}

fn prover_config_from_common(common: &CommonCircuitData<GoldilocksField, 2>) -> ProverConfig {
    ProverConfig {
        degree_bits: common.degree_bits() as u32,
        num_wires: common.config.num_wires as u32,
        num_routed_wires: common.config.num_routed_wires as u32,
        num_challenges: common.config.num_challenges as u32,
        num_partial_products: common.num_partial_products as u32,
        quotient_degree_factor: common.quotient_degree_factor as u32,
        rate_bits: common.config.fri_config.rate_bits as u32,
        cap_height: common.config.fri_config.cap_height as u32,
        num_gate_constraints: common.num_gate_constraints as u32,
        num_constants: common.num_constants as u32,
        num_public_inputs: common.num_public_inputs as u32,
    }
}
