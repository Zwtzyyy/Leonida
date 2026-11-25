//! Compute programs the title hands to the console device's method 0x218
//! (`DeserializeComputePipelineStateX` in the console's own messages), turned
//! into a shader container of the kind the rest of the translator reads.
//!
//! The method's description is the root signature, then (pointer, size)
//! pairs: kind 0 at +0x08, kind 1 at +0x18, kind 6 at +0x28, kind 7 at +0x38
//! (the kind numbers of the console's `FSPX` payload; the layout was measured
//! by the Palmetto project, GPL-3.0, docs/fspx.md, and checked here on the
//! 42 programs Grand Theft Auto VI creates this way).
//!
//! ```text
//! kind 0  7 u32: ?, ?, 0, threads x | y << 16, threads z | bit 20 wave32,
//!         COMPUTE_PGM_RSRC1, COMPUTE_PGM_RSRC2
//! kind 6  the RDNA2 code; 36 bytes of VOP3 no-ops (opcode 0x180, three
//!         12-byte words carrying the shader hash) come first
//! kind 7  u32 0x403, size, ?, ?, ?, counts, then 32-byte records:
//!         0x41 type, a, b, c, d, -1, 0
//!           type 0x401: root parameter: b = root type (3 table),
//!                       d bits 0-15 = its first user SGPR
//!           type 0x3kk: one descriptor in that table: kk = resource kind
//!                       (0 SRV, 1 sampler, 2 CBV, 3 UAV), a = dword offset
//!                       in the table, b = shader register, d bits 0-7 =
//!                       descriptor dwords
//!         0x42 (used resources, the same ones again), 0x43 the shader hash
//! ```
//!
//! The container holds one `DXIL` part whose `XDAT` index has one native
//! blob (`XBCH`, zlib, `FSSM`) with a compute record, the format
//! [`crate::native_programs`] reads.

use crate::ShaderError::{self, Malformed, Unsupported};
use crate::dxbc::{FourCC, build};

const ABSENT: u64 = 0xDEAD_BEEF_DEAD_BEEF;
const RECORD_SIZE: usize = 0x54;
const PROLOGUE: usize = 36;

fn word(b: &[u8], i: usize) -> Option<u32> {
    b.get(4 * i..4 * i + 4).map(|w| u32::from_le_bytes(w.try_into().unwrap_or([0; 4])))
}

/// One table descriptor of kind 7: (resource kind, table dword, register, dwords).
type Entry = (u8, u32, u32, u8);

/// The first user SGPR of the table and its descriptors.
fn records(k7: &[u8]) -> Result<(u8, Vec<Entry>), ShaderError> {
    if word(k7, 0) != Some(0x403) {
        return Err(Malformed("console compute: kind 7 lacks its 0x403 header"));
    }
    let mut table_sgpr = None;
    let mut entries = Vec::new();
    let mut i = 6;
    while let Some(tag) = word(k7, i) {
        if tag != 0x41 && tag != 0x42 {
            break;
        }
        let w = |k| word(k7, i + k).ok_or(Malformed("console compute: kind 7 record cut short"));
        if tag == 0x41 {
            let kind = w(1)?;
            match kind >> 8 {
                // Root type 3 (`d` 0xb0004) and 0 (`d` 0x30004) both arrive as
                // a table address in the user SGPR: the 42 GTA VI programs of
                // either start `s_mov_b32 s8, s4; s_mov_b32 s9, 0` and load
                // their descriptors from s[8:9].
                4 => {
                    if !matches!(w(3)?, 0 | 3) {
                        return Err(Unsupported { offset: 0, what: "a console compute root parameter that is not a table" });
                    }
                    if table_sgpr.replace((w(5)? & 0xFFFF) as u8).is_some() {
                        return Err(Unsupported { offset: 0, what: "a console compute program with two root tables" });
                    }
                }
                3 if kind & 0xFF <= 3 => entries.push(((kind & 0xFF) as u8, w(2)?, w(3)?, (w(5)? & 0xFF) as u8)),
                // shortcut: a type-2 record (one GTA VI program: b 0x1f6, d
                // 0x10504; its code reads only its table) is left out; a
                // program that read what it describes fails resource tracking.
                0 if kind == 2 => {}
                _ => return Err(Unsupported { offset: 0, what: "a console compute kind-7 record of an unknown type" }),
            }
        }
        i += 8;
    }
    // Without a root parameter record (three GTA VI programs) the table is
    // at user SGPR 4, where the code of every such program reads it; a
    // program that did not would fail resource tracking.
    let sgpr = match table_sgpr {
        Some(s) => s,
        None if !entries.is_empty() => 4,
        None => return Err(Malformed("console compute: no root table record")),
    };
    Ok((sgpr, entries))
}

/// A shader container holding the program of a method-0x218 description.
pub fn console_compute_container(k0: &[u8], k6: &[u8], k7: &[u8]) -> Result<Vec<u8>, ShaderError> {
    let k = |i| word(k0, i).ok_or(Malformed("console compute: kind 0 shorter than 7 words"));
    let (xy, z, rsrc1, rsrc2) = (k(3)?, k(4)?, k(5)?, k(6)?);
    let wave32 = z & 1 << 20 != 0;
    let threads = [xy & 0xFFFF, xy >> 16, z & 0xFFFF];
    if threads.iter().any(|&t| t == 0) || threads.iter().product::<u32>() > 1024 {
        return Err(Malformed("console compute: implausible thread-group size"));
    }
    // The hash-carrying no-ops (VOP3 opcode 0x180) are left out.
    let skip = (0..3).all(|n| word(k6, 3 * n).is_some_and(|w| w >> 16 == 0xD580));
    let code = if skip { &k6[PROLOGUE.min(k6.len())..] } else { k6 };
    if code.is_empty() || code.len() % 4 != 0 {
        return Err(Malformed("console compute: no code"));
    }
    let (user, entries) = records(k7)?;
    // RSRC1: VGPRs in granules of 8 (wave32) or 4 (wave64), SGPRs in 8s.
    let vgprs = ((rsrc1 & 0x3F) + 1) * if wave32 { 8 } else { 4 };
    let sgprs = (((rsrc1 >> 6 & 0xF) + 1) * 8).min(106);
    let lds = (rsrc2 >> 15 & 0x1FF) * 512;

    // FSSM payload: slot table, record, stage data, code, layout and used tables.
    let record_at = 0x80usize;
    let stage_end = record_at + RECORD_SIZE + 0x34;
    let code_at = stage_end.next_multiple_of(256);
    let layout_at = code_at + code.len();
    let used_at = layout_at + 20 * entries.len();
    let mut pl = vec![0u8; used_at + 20 * entries.len()];
    let put32 = |pl: &mut Vec<u8>, at: usize, v: u32| pl[at..at + 4].copy_from_slice(&v.to_le_bytes());
    let put64 = |pl: &mut Vec<u8>, at: usize, v: u64| pl[at..at + 8].copy_from_slice(&v.to_le_bytes());
    put64(&mut pl, 0, 6);
    for at in (0x08..0x80).step_by(8) {
        put64(&mut pl, at, ABSENT);
    }
    put64(&mut pl, 0x60, record_at as u64);
    let r = record_at;
    put64(&mut pl, r, code_at as u64);
    put64(&mut pl, r + 0x08, code.len() as u64);
    put32(&mut pl, r + 0x14, if wave32 { 1 << 5 } else { 0 });
    put64(&mut pl, r + 0x28, layout_at as u64);
    put64(&mut pl, r + 0x30, used_at as u64);
    put32(&mut pl, r + 0x38, entries.len() as u32);
    put32(&mut pl, r + 0x3c, entries.len() as u32);
    put32(&mut pl, r + 0x4c, lds);
    put32(&mut pl, r + 0x50, vgprs.min(256) | sgprs << 20);
    let e = r + RECORD_SIZE;
    put32(&mut pl, e + 0x24, rsrc2);
    for (n, t) in threads.into_iter().enumerate() {
        put32(&mut pl, e + 0x28 + 4 * n, t);
    }
    pl[code_at..code_at + code.len()].copy_from_slice(code);
    for (n, &(kind, dword, register, dwords)) in entries.iter().enumerate() {
        // layout: root type 3 (table), table dword, 1 SGPR, register, user data, slot 0
        let l = layout_at + 20 * n;
        put32(&mut pl, l, 3);
        put32(&mut pl, l + 4, dword);
        put32(&mut pl, l + 8, 1);
        put32(&mut pl, l + 12, register);
        pl[l + 16] = user;
        // used: kind, root type 3, table dword, register, stride 0, dwords, user data, slot 0
        let u = used_at + 20 * n;
        pl[u] = kind;
        pl[u + 1] = 3;
        put32(&mut pl, u + 4, dword);
        put32(&mut pl, u + 8, register);
        pl[u + 16] = dwords;
        pl[u + 17] = user;
    }
    let mut fssm = Vec::with_capacity(16 + pl.len());
    fssm.extend_from_slice(b"FSSM");
    fssm.extend_from_slice(&0x101u32.to_le_bytes());
    fssm.extend_from_slice(&(pl.len() as u64).to_le_bytes());
    fssm.extend_from_slice(&pl);
    let packed = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 1);
    let mut blob = Vec::with_capacity(0x18 + packed.len());
    blob.extend_from_slice(b"XBCH");
    blob.extend_from_slice(&3u32.to_le_bytes());
    blob.extend_from_slice(&(packed.len() as u64).to_le_bytes());
    blob.extend_from_slice(&(fssm.len() as u64).to_le_bytes());
    blob.extend_from_slice(&packed);
    // DXIL part: 0x18 bytes of program header (unused), XDAT with one entry, the blob.
    let blob_at = 0x18 + 12 + 12;
    let mut part = vec![0u8; blob_at];
    part[0x18..0x1c].copy_from_slice(b"XDAT");
    part[0x1c..0x20].copy_from_slice(&1u32.to_le_bytes());
    part[0x20..0x24].copy_from_slice(&0x24u32.to_le_bytes());
    part[0x24..0x28].copy_from_slice(&0x0du32.to_le_bytes());
    part[0x28..0x2c].copy_from_slice(&(blob_at as u32).to_le_bytes());
    part[0x2c..0x30].copy_from_slice(&(blob.len() as u32).to_le_bytes());
    part.extend_from_slice(&blob);
    Ok(build(&[(FourCC::DXIL, &part)]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{HwStage, ResourceType, RootType, StageIo};

    fn words(w: &[u32]) -> Vec<u8> {
        w.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn a_console_compute_description_reads_back_as_its_native_program() {
        // 8x8 wave64 group, 5 user SGPRs (table at s4), tgid x and y
        let k0 = words(&[0x4005_cf00, 0x6c09_0102, 0, 0x0008_0008, 1, 0x402c_0343, 0x98a]);
        let mut k6 = Vec::new();
        for _ in 0..3 {
            k6.extend(words(&[0xd580_0000, 0, 0]));
        }
        let code = words(&[0xbfa0_0003, 0xbf81_0000]);
        k6.extend(&code);
        let mut k7 = words(&[0x403, 0x144, 0x90, 0, 0, 0x0005_0440]);
        k7.extend(words(&[0x41, 0x401, 0, 3, 0, 0x000b_0004, u32::MAX, 0]));
        k7.extend(words(&[0x41, 0x300, 0, 0x10b, 4, 0x408, u32::MAX, 0]));
        k7.extend(words(&[0x41, 0x303, 8, 0x2c, 4, 0x408, u32::MAX, 0]));
        k7.extend(words(&[0x41, 0x302, 24, 0x1b, 0, 0x404, u32::MAX, 0]));
        k7.extend(words(&[0x42, 0, 0x0002_0001, 0x10b, 1, 0, u32::MAX, 0]));
        k7.extend(words(&[0x43, 1, 2, 3, 4, 0, 0, 0]));
        let container = console_compute_container(&k0, &k6, &k7).unwrap();
        let c = crate::dxbc::Container::parse(&container).unwrap();
        let p = crate::native_programs(c.part(FourCC::DXIL).unwrap().data).unwrap();
        assert_eq!(p.len(), 1);
        let p = &p[0];
        assert_eq!(p.stage, HwStage::Cs);
        assert!(p.wave64);
        assert_eq!(p.code, code);
        assert_eq!(p.vgprs, 16);
        let StageIo::Cs(io) = &p.io else { panic!("not compute") };
        assert_eq!(io.threads, [8, 8, 1]);
        assert_eq!(io.user_sgprs(), 5);
        assert_eq!(p.layout.len(), 3);
        assert!(p.layout.iter().all(|l| l.root == RootType::Table && l.user_data == 4 && l.slot == 0));
        assert_eq!(p.layout[1].table_offset_dwords, 8);
        assert_eq!(p.used[1].kind, ResourceType::Uav);
        assert_eq!(p.used[2].kind, ResourceType::Cbv);
        assert_eq!((p.used[2].table_offset_dwords, p.used[2].lower_bound, p.used[2].dwords), (24, 0x1b, 4));
    }

    #[test]
    fn a_root_parameter_that_is_not_a_table_is_refused() {
        let k0 = words(&[0, 0, 0, 0x0008_0008, 1, 0, 0x98a]);
        let k6 = words(&[0xbf81_0000]);
        let mut k7 = words(&[0x403, 0, 0, 0, 0, 0]);
        k7.extend(words(&[0x41, 0x401, 0, 1, 0, 4, u32::MAX, 0]));
        k7.extend(words(&[0x41, 0x300, 0, 0x10b, 4, 0x408, u32::MAX, 0]));
        assert!(console_compute_container(&k0, &k6, &k7).is_err());
    }
}
