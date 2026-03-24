//! Big-O complexity verification tests for R0N Ingress.
//!
//! Uses the `big-o-test` crate to empirically measure and enforce
//! time and space complexity bounds on all key functions.
//!
//! Run with: `cargo test --test big_o_complexity --release -- --test-threads=1`
//! Space complexity is only verified when running with `--test-threads=1` because
//! the global allocator hook captures allocations from all threads.
//!
//! These tests require release mode (`--release`) — the `big-o-test` crate
//! panics on underflow when debug-build measurements are too small/noisy.

// Skip the entire file in debug builds.
#![cfg(not(debug_assertions))]
#![allow(unsafe_code, clippy::cast_possible_truncation, clippy::cast_lossless)]

use big_o_test::*;
use std::fmt::Write as _;

const LOOP_MUL: u32 = 16;

/// Operations using raw/pool memory or allocating zero bytes bypass the
/// standard allocator, so space complexity cannot be measured (0b/0b → undefined).
const SPACE_UNMEASURABLE: BigOAlgorithmComplexity = BigOAlgorithmComplexity::WorseThanExponential;

/// Returns `expected` when running single-threaded (space is measurable),
/// or `WorseThanExponential` (skip) when multi-threaded.
fn space(expected: BigOAlgorithmComplexity) -> BigOAlgorithmComplexity {
    use std::sync::OnceLock;
    static SINGLE: OnceLock<bool> = OnceLock::new();
    let single = *SINGLE.get_or_init(|| {
        // Check --test-threads=N in CLI args (overrides env var)
        let args: Vec<String> = std::env::args().collect();
        for (i, arg) in args.iter().enumerate() {
            if arg == "--test-threads" {
                if let Some(v) = args.get(i + 1) {
                    return v == "1";
                }
            }
            if let Some(v) = arg.strip_prefix("--test-threads=") {
                return v == "1";
            }
        }
        // Fallback to env var; if unset, default is multi-threaded
        std::env::var("RUST_TEST_THREADS")
            .map(|v| v == "1")
            .unwrap_or(false)
    });
    if single {
        expected
    } else {
        BigOAlgorithmComplexity::WorseThanExponential
    }
}

// ═══════════════════════════════════════════════════════════════════
// perf::zero_copy — SharedBuffer
// ═══════════════════════════════════════════════════════════════════

mod shared_buffer {
    use super::*;
    use r0n_ingress::perf::SharedBuffer;

    #[test]
    fn from_slice_scales_linearly() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let data1 = parking_lot::RwLock::new(Vec::<u8>::new());
        let data2 = parking_lot::RwLock::new(Vec::<u8>::new());
        test_algorithm(
            "SharedBuffer::from_slice",
            10,
            || {
                *data1.write() = vec![0xABu8; n1 as usize];
                *data2.write() = vec![0xABu8; n2 as usize];
            },
            n1,
            || {
                let d = data1.read();
                let buf = SharedBuffer::from_slice(&d);
                buf.len() as u32
            },
            n2,
            || {
                let d = data2.read();
                let buf = SharedBuffer::from_slice(&d);
                buf.len() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }

    #[test]
    fn slice_is_constant() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let buf1 = parking_lot::RwLock::new(SharedBuffer::new(vec![]));
        let buf2 = parking_lot::RwLock::new(SharedBuffer::new(vec![]));
        test_algorithm(
            "SharedBuffer::slice",
            10,
            || {
                *buf1.write() = SharedBuffer::new(vec![0u8; n1 as usize]);
                *buf2.write() = SharedBuffer::new(vec![0u8; n2 as usize]);
            },
            n1,
            || {
                let b = buf1.read();
                let s = b.slice(0..b.len() / 2);
                s.len() as u32
            },
            n2,
            || {
                let b = buf2.read();
                let s = b.slice(0..b.len() / 2);
                s.len() as u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
        );
    }

    #[test]
    fn split_at_is_constant() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let buf1 = parking_lot::RwLock::new(SharedBuffer::new(vec![]));
        let buf2 = parking_lot::RwLock::new(SharedBuffer::new(vec![]));
        test_algorithm(
            "SharedBuffer::split_at",
            10,
            || {
                *buf1.write() = SharedBuffer::new(vec![0u8; n1 as usize]);
                *buf2.write() = SharedBuffer::new(vec![0u8; n2 as usize]);
            },
            n1,
            || {
                let b = buf1.read();
                let (l, r) = b.split_at(b.len() / 2);
                (l.len() + r.len()) as u32
            },
            n2,
            || {
                let b = buf2.read();
                let (l, r) = b.split_at(b.len() / 2);
                (l.len() + r.len()) as u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
        );
    }

    #[test]
    fn make_unique_scales_linearly() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let src1 = parking_lot::RwLock::new(SharedBuffer::new(vec![]));
        let src2 = parking_lot::RwLock::new(SharedBuffer::new(vec![]));
        let clone1 = parking_lot::RwLock::new(SharedBuffer::new(vec![]));
        let clone2 = parking_lot::RwLock::new(SharedBuffer::new(vec![]));
        test_algorithm(
            "SharedBuffer::make_unique",
            10,
            || {
                let b1 = SharedBuffer::new(vec![0u8; n1 as usize]);
                *src1.write() = b1.clone();
                *clone1.write() = b1;
                let b2 = SharedBuffer::new(vec![0u8; n2 as usize]);
                *src2.write() = b2.clone();
                *clone2.write() = b2;
            },
            n1,
            || {
                let mut c = clone1.write();
                c.make_unique();
                c.len() as u32
            },
            n2,
            || {
                let mut c = clone2.write();
                c.make_unique();
                c.len() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// perf::zero_copy — BufferChain
// ═══════════════════════════════════════════════════════════════════

mod buffer_chain {
    use super::*;
    use r0n_ingress::perf::{BufferChain, SharedBuffer};

    #[test]
    fn push_is_amortized_constant() {
        let n1: u32 = 50_000 * LOOP_MUL;
        let n2: u32 = 100_000 * LOOP_MUL;
        let chain1 = parking_lot::RwLock::new(BufferChain::new());
        let chain2 = parking_lot::RwLock::new(BufferChain::new());
        test_algorithm(
            "BufferChain::push (amortized)",
            10,
            || {
                chain1.write().clear();
                chain2.write().clear();
            },
            n1,
            || {
                let mut c = chain1.write();
                c.push(SharedBuffer::new(vec![0u8; 8]));
                c.buffer_count() as u32
            },
            n2,
            || {
                let mut c = chain2.write();
                c.push(SharedBuffer::new(vec![0u8; 8]));
                c.buffer_count() as u32
            },
            BigOAlgorithmComplexity::O1,
            space(BigOAlgorithmComplexity::O1),
        );
    }

    #[test]
    fn flatten_scales_linearly() {
        let n1: u32 = 10_000 * LOOP_MUL;
        let n2: u32 = 20_000 * LOOP_MUL;
        let chain1 = parking_lot::RwLock::new(BufferChain::new());
        let chain2 = parking_lot::RwLock::new(BufferChain::new());
        test_algorithm(
            "BufferChain::flatten",
            10,
            || {
                let mut c1 = chain1.write();
                c1.clear();
                for i in 0..n1 {
                    c1.push(SharedBuffer::new(vec![i as u8; 16]));
                }
                let mut c2 = chain2.write();
                c2.clear();
                for i in 0..n2 {
                    c2.push(SharedBuffer::new(vec![i as u8; 16]));
                }
            },
            n1,
            || {
                let c = chain1.read();
                let flat = c.flatten();
                flat.len() as u32
            },
            n2,
            || {
                let c = chain2.read();
                let flat = c.flatten();
                flat.len() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }

    #[test]
    fn get_scales_linearly() {
        let n1: u32 = 5_000 * LOOP_MUL;
        let n2: u32 = 10_000 * LOOP_MUL;
        let chain1 = parking_lot::RwLock::new(BufferChain::new());
        let chain2 = parking_lot::RwLock::new(BufferChain::new());
        test_algorithm(
            "BufferChain::get (worst case)",
            10,
            || {
                let mut c1 = chain1.write();
                c1.clear();
                for i in 0..n1 {
                    c1.push(SharedBuffer::new(vec![i as u8; 1]));
                }
                let mut c2 = chain2.write();
                c2.clear();
                for i in 0..n2 {
                    c2.push(SharedBuffer::new(vec![i as u8; 1]));
                }
            },
            n1,
            || {
                let c = chain1.read();
                c.get(c.len().saturating_sub(1)).unwrap_or(0) as u32
            },
            n2,
            || {
                let c = chain2.read();
                c.get(c.len().saturating_sub(1)).unwrap_or(0) as u32
            },
            BigOAlgorithmComplexity::ON,
            SPACE_UNMEASURABLE,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// perf::zero_copy — ReadBuffer
// ═══════════════════════════════════════════════════════════════════

mod read_buffer {
    use super::*;
    use r0n_ingress::perf::ReadBuffer;

    #[test]
    fn advance_read_is_constant() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let buf1 = parking_lot::RwLock::new(ReadBuffer::new(0));
        let buf2 = parking_lot::RwLock::new(ReadBuffer::new(0));
        test_algorithm(
            "ReadBuffer::advance_read",
            10,
            || {
                let mut b1 = buf1.write();
                *b1 = ReadBuffer::new(n1 as usize);
                b1.advance_write(n1 as usize);
                let mut b2 = buf2.write();
                *b2 = ReadBuffer::new(n2 as usize);
                b2.advance_write(n2 as usize);
            },
            n1,
            || {
                let mut b = buf1.write();
                b.advance_read(1);
                b.readable_len() as u32
            },
            n2,
            || {
                let mut b = buf2.write();
                b.advance_read(1);
                b.readable_len() as u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
        );
    }

    #[test]
    fn compact_scales_linearly() {
        // Keep sizes within L2 cache to avoid cache-thrashing under parallel load.
        let n1: u32 = 5_000 * LOOP_MUL;
        let n2: u32 = 10_000 * LOOP_MUL;
        let buf1 = parking_lot::RwLock::new(ReadBuffer::new(0));
        let buf2 = parking_lot::RwLock::new(ReadBuffer::new(0));
        test_algorithm(
            "ReadBuffer::compact",
            10,
            || {
                *buf1.write() = ReadBuffer::new(n1 as usize);
                *buf2.write() = ReadBuffer::new(n2 as usize);
            },
            n1,
            || {
                let mut b = buf1.write();
                let mut total = 0u32;
                for _ in 0..64 {
                    b.clear();
                    b.advance_write(n1 as usize);
                    b.advance_read(n1 as usize / 2);
                    b.compact();
                    total = total.wrapping_add(b.readable_len() as u32);
                }
                total
            },
            n2,
            || {
                let mut b = buf2.write();
                let mut total = 0u32;
                for _ in 0..64 {
                    b.clear();
                    b.advance_write(n2 as usize);
                    b.advance_read(n2 as usize / 2);
                    b.compact();
                    total = total.wrapping_add(b.readable_len() as u32);
                }
                total
            },
            BigOAlgorithmComplexity::ON,
            SPACE_UNMEASURABLE,
        );
    }

    #[test]
    fn ensure_writable_may_grow_linearly() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let buf1 = parking_lot::RwLock::new(ReadBuffer::new(0));
        let buf2 = parking_lot::RwLock::new(ReadBuffer::new(0));
        test_algorithm(
            "ReadBuffer::ensure_writable (grow)",
            10,
            || {
                *buf1.write() = ReadBuffer::new(16);
                *buf2.write() = ReadBuffer::new(16);
            },
            n1,
            || {
                let mut b = buf1.write();
                b.ensure_writable(n1 as usize);
                b.capacity() as u32
            },
            n2,
            || {
                let mut b = buf2.write();
                b.ensure_writable(n2 as usize);
                b.capacity() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// perf::zero_copy — WriteBuffer
// ═══════════════════════════════════════════════════════════════════

mod write_buffer {
    use super::*;
    use r0n_ingress::perf::WriteBuffer;

    #[test]
    fn append_scales_linearly() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let buf1 = parking_lot::RwLock::new(WriteBuffer::new(0));
        let buf2 = parking_lot::RwLock::new(WriteBuffer::new(0));
        test_algorithm(
            "WriteBuffer::append",
            10,
            || {
                *buf1.write() = WriteBuffer::new(n1 as usize);
                *buf2.write() = WriteBuffer::new(n2 as usize);
            },
            n1,
            || {
                let mut b = buf1.write();
                let data = vec![0u8; n1 as usize];
                b.append(&data);
                b.pending_len() as u32
            },
            n2,
            || {
                let mut b = buf2.write();
                let data = vec![0u8; n2 as usize];
                b.append(&data);
                b.pending_len() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }

    #[test]
    fn advance_is_constant() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let buf1 = parking_lot::RwLock::new(WriteBuffer::new(0));
        let buf2 = parking_lot::RwLock::new(WriteBuffer::new(0));
        test_algorithm(
            "WriteBuffer::advance",
            10,
            || {
                let mut b1 = buf1.write();
                *b1 = WriteBuffer::new(n1 as usize);
                b1.append(&vec![0u8; n1 as usize]);
                let mut b2 = buf2.write();
                *b2 = WriteBuffer::new(n2 as usize);
                b2.append(&vec![0u8; n2 as usize]);
            },
            n1,
            || {
                let mut b = buf1.write();
                b.advance(1);
                b.pending_len() as u32
            },
            n2,
            || {
                let mut b = buf2.write();
                b.advance(1);
                b.pending_len() as u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// perf::zero_copy — ByteCursor
// ═══════════════════════════════════════════════════════════════════

mod byte_cursor {
    use super::*;
    use r0n_ingress::perf::ByteCursor;

    #[test]
    fn read_until_scales_linearly() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let data1 = parking_lot::RwLock::new(Vec::<u8>::new());
        let data2 = parking_lot::RwLock::new(Vec::<u8>::new());
        test_algorithm(
            "ByteCursor::read_until",
            10,
            || {
                let mut d1 = data1.write();
                *d1 = vec![0x41u8; n1 as usize];
                d1[n1 as usize - 1] = b'\n';
                let mut d2 = data2.write();
                *d2 = vec![0x41u8; n2 as usize];
                d2[n2 as usize - 1] = b'\n';
            },
            n1,
            || {
                let d = data1.read();
                let mut cursor = ByteCursor::new(&d);
                cursor.read_until(b'\n').map_or(0u32, |s| s.len() as u32)
            },
            n2,
            || {
                let d = data2.read();
                let mut cursor = ByteCursor::new(&d);
                cursor.read_until(b'\n').map_or(0u32, |s| s.len() as u32)
            },
            BigOAlgorithmComplexity::ON,
            SPACE_UNMEASURABLE,
        );
    }

    #[test]
    fn read_slice_is_constant() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let data1 = parking_lot::RwLock::new(Vec::<u8>::new());
        let data2 = parking_lot::RwLock::new(Vec::<u8>::new());
        test_algorithm(
            "ByteCursor::read_slice",
            10,
            || {
                *data1.write() = vec![0x41u8; n1 as usize];
                *data2.write() = vec![0x41u8; n2 as usize];
            },
            n1,
            || {
                let d = data1.read();
                let mut cursor = ByteCursor::new(&d);
                cursor.read_slice(16).map_or(0u32, |s| s.len() as u32)
            },
            n2,
            || {
                let d = data2.read();
                let mut cursor = ByteCursor::new(&d);
                cursor.read_slice(16).map_or(0u32, |s| s.len() as u32)
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// perf::zero_copy — IoVec
// ═══════════════════════════════════════════════════════════════════

mod io_vec {
    use super::*;
    use r0n_ingress::perf::IoVec;

    #[test]
    fn push_is_amortized_constant() {
        let n1: u32 = 50_000 * LOOP_MUL;
        let n2: u32 = 100_000 * LOOP_MUL;
        let iovec1 = parking_lot::RwLock::new(IoVec::new());
        let iovec2 = parking_lot::RwLock::new(IoVec::new());
        test_algorithm(
            "IoVec::push (amortized)",
            10,
            || {
                iovec1.write().clear();
                iovec2.write().clear();
            },
            n1,
            || {
                let mut v = iovec1.write();
                v.push(vec![0u8; 16]);
                v.slice_count() as u32
            },
            n2,
            || {
                let mut v = iovec2.write();
                v.push(vec![0u8; 16]);
                v.slice_count() as u32
            },
            BigOAlgorithmComplexity::O1,
            space(BigOAlgorithmComplexity::O1),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// perf::memory — MemoryPool
// ═══════════════════════════════════════════════════════════════════

mod memory_pool {
    use super::*;
    use r0n_ingress::perf::MemoryPool;

    fn make_vec() -> Vec<u8> {
        Vec::with_capacity(64)
    }

    #[test]
    fn get_put_is_constant() {
        let iters: u32 = 50_000 * LOOP_MUL;
        let pool = parking_lot::RwLock::new(MemoryPool::new(1000, make_vec));
        test_crud_algorithms(
            "MemoryPool get/put",
            10,
            |_n| {
                let p = pool.read();
                p.clear();
                p.preallocate(500);
                p.len() as u32
            },
            |_n| {
                let p = pool.read();
                let v = p.get();
                p.put(v);
                p.len() as u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
            |_n| {
                let p = pool.read();
                p.len() as u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
            |_n| {
                let p = pool.read();
                p.len() as u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
            |_n| {
                let p = pool.read();
                let v = p.get();
                v.capacity() as u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
            0,
            iters,
            iters,
            iters,
            iters,
            1,
            1,
            1,
            1,
        );
    }

    #[test]
    fn preallocate_scales_linearly() {
        let n1: u32 = 5_000 * LOOP_MUL;
        let n2: u32 = 10_000 * LOOP_MUL;
        let pool1 = parking_lot::RwLock::new(MemoryPool::new(100_000, make_vec));
        let pool2 = parking_lot::RwLock::new(MemoryPool::new(100_000, make_vec));
        test_algorithm(
            "MemoryPool::preallocate",
            10,
            || {
                pool1.read().clear();
                pool2.read().clear();
            },
            n1,
            || {
                let p = pool1.read();
                p.preallocate(n1 as usize);
                p.len() as u32
            },
            n2,
            || {
                let p = pool2.read();
                p.preallocate(n2 as usize);
                p.len() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// perf::memory — Arena
// ═══════════════════════════════════════════════════════════════════

mod arena_alloc {
    use super::*;
    use r0n_ingress::perf::Arena;

    #[test]
    fn alloc_is_amortized_constant() {
        let n1: u32 = 50_000 * LOOP_MUL;
        let n2: u32 = 100_000 * LOOP_MUL;
        let arena1 = parking_lot::RwLock::new(Arena::new());
        let arena2 = parking_lot::RwLock::new(Arena::new());
        test_algorithm(
            "Arena::alloc (amortized)",
            10,
            || {
                // SAFETY: reset invalidates all prior allocations;
                // we hold no references across this boundary.
                unsafe {
                    arena1.read().reset();
                    arena2.read().reset();
                }
            },
            n1,
            || {
                let a = arena1.read();
                let _ptr = a.alloc(64);
                a.total_allocated() as u32
            },
            n2,
            || {
                let a = arena2.read();
                let _ptr = a.alloc(64);
                a.total_allocated() as u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
        );
    }

    #[test]
    fn alloc_slice_scales_linearly() {
        let n1: u32 = 50_000 * LOOP_MUL;
        let n2: u32 = 100_000 * LOOP_MUL;
        let src1 = vec![0u8; n1 as usize];
        let src2 = vec![0u8; n2 as usize];
        let arena1 = parking_lot::RwLock::new(Arena::with_chunk_size(n1 as usize * 2));
        let arena2 = parking_lot::RwLock::new(Arena::with_chunk_size(n2 as usize * 2));
        test_algorithm(
            "Arena::alloc_slice",
            10,
            || unsafe {
                arena1.read().reset();
                arena2.read().reset();
            },
            n1,
            || {
                let a = arena1.read();
                let s = a.alloc_slice(&src1);
                s.len() as u32
            },
            n2,
            || {
                let a = arena2.read();
                let s = a.alloc_slice(&src2);
                s.len() as u32
            },
            BigOAlgorithmComplexity::ON,
            SPACE_UNMEASURABLE,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// perf::memory — Slab
// ═══════════════════════════════════════════════════════════════════

mod slab_alloc {
    use super::*;
    use r0n_ingress::perf::Slab;

    fn make_item() -> Vec<u8> {
        Vec::with_capacity(128)
    }

    #[test]
    fn alloc_free_is_constant() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let slab1 = parking_lot::RwLock::new(Slab::new(5000, make_item));
        let slab2 = parking_lot::RwLock::new(Slab::new(5000, make_item));
        test_algorithm(
            "Slab alloc+free",
            10,
            || {},
            n1,
            || {
                let s = slab1.read();
                let obj = s.alloc();
                s.free(obj);
                0u32
            },
            n2,
            || {
                let s = slab2.read();
                let obj = s.alloc();
                s.free(obj);
                0u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// perf::benchmark — LatencyHistogram
// ═══════════════════════════════════════════════════════════════════

mod latency_histogram {
    use super::*;
    use r0n_ingress::perf::LatencyHistogram;
    use std::time::Duration;

    #[test]
    fn record_is_constant() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let hist1 = parking_lot::RwLock::new(LatencyHistogram::new(10, 10_000_000));
        let hist2 = parking_lot::RwLock::new(LatencyHistogram::new(10, 10_000_000));
        test_algorithm(
            "LatencyHistogram::record",
            10,
            || {
                hist1.read().reset();
                hist2.read().reset();
            },
            n1,
            || {
                let h = hist1.read();
                for i in 0..n1 {
                    h.record(Duration::from_micros(u64::from(i % 10000)));
                }
                h.count() as u32
            },
            n2,
            || {
                let h = hist2.read();
                for i in 0..n2 {
                    h.record(Duration::from_micros(u64::from(i % 10000)));
                }
                h.count() as u32
            },
            BigOAlgorithmComplexity::ON,
            SPACE_UNMEASURABLE,
        );
    }

    #[test]
    fn stats_scales_with_buckets() {
        let n1: u32 = 100_000;
        let n2: u32 = 200_000;
        let hist1 = parking_lot::RwLock::new(LatencyHistogram::new(1, 0));
        let hist2 = parking_lot::RwLock::new(LatencyHistogram::new(1, 0));
        test_algorithm(
            "LatencyHistogram::stats",
            10,
            || {
                *hist1.write() = LatencyHistogram::new(1, u64::from(n1));
                hist1.read().record(Duration::from_micros(1));
                *hist2.write() = LatencyHistogram::new(1, u64::from(n2));
                hist2.read().record(Duration::from_micros(1));
            },
            n1,
            || {
                let h = hist1.read();
                let s = h.stats().unwrap();
                s.count as u32
            },
            n2,
            || {
                let h = hist2.read();
                let s = h.stats().unwrap();
                s.count as u32
            },
            BigOAlgorithmComplexity::ON,
            SPACE_UNMEASURABLE,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// perf::benchmark — ThroughputMetrics
// ═══════════════════════════════════════════════════════════════════

mod throughput_metrics {
    use super::*;
    use r0n_ingress::perf::ThroughputMetrics;

    #[test]
    fn record_op_is_constant() {
        let n1: u32 = 200_000 * LOOP_MUL;
        let n2: u32 = 400_000 * LOOP_MUL;
        let m1 = parking_lot::RwLock::new(ThroughputMetrics::new());
        let m2 = parking_lot::RwLock::new(ThroughputMetrics::new());
        test_algorithm(
            "ThroughputMetrics::record_op",
            10,
            || {
                m1.read().reset();
                m2.read().reset();
            },
            n1,
            || {
                let m = m1.read();
                m.record_op();
                m.total_ops() as u32
            },
            n2,
            || {
                let m = m2.read();
                m.record_op();
                m.total_ops() as u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// ipc::message — Serialization
// ═══════════════════════════════════════════════════════════════════

mod ipc_message {
    use super::*;
    use r0n_ingress::ipc::{ControlCommand, ControlMessage, ControlResponse};

    #[test]
    fn message_serialize_with_payload_scales_linearly() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let msg1 = parking_lot::RwLock::new(ControlMessage::with_timestamp(
            1,
            ControlCommand::Init { config: Vec::new() },
            0,
        ));
        let msg2 = parking_lot::RwLock::new(ControlMessage::with_timestamp(
            2,
            ControlCommand::Init { config: Vec::new() },
            0,
        ));
        test_algorithm(
            "ControlMessage::to_bytes (payload sized)",
            10,
            || {
                *msg1.write() = ControlMessage::with_timestamp(
                    1,
                    ControlCommand::Init {
                        config: vec![0u8; n1 as usize],
                    },
                    0,
                );
                *msg2.write() = ControlMessage::with_timestamp(
                    2,
                    ControlCommand::Init {
                        config: vec![0u8; n2 as usize],
                    },
                    0,
                );
            },
            n1,
            || {
                let m = msg1.read();
                let bytes = m.to_bytes().unwrap();
                bytes.len() as u32
            },
            n2,
            || {
                let m = msg2.read();
                let bytes = m.to_bytes().unwrap();
                bytes.len() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }

    #[test]
    fn message_deserialize_with_payload_scales_linearly() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        let bytes1 = parking_lot::RwLock::new(Vec::<u8>::new());
        let bytes2 = parking_lot::RwLock::new(Vec::<u8>::new());
        test_algorithm(
            "ControlMessage::from_bytes (payload sized)",
            10,
            || {
                let m1 = ControlMessage::with_timestamp(
                    1,
                    ControlCommand::Init {
                        config: vec![0u8; n1 as usize],
                    },
                    0,
                );
                *bytes1.write() = m1.to_bytes().unwrap();
                let m2 = ControlMessage::with_timestamp(
                    2,
                    ControlCommand::Init {
                        config: vec![0u8; n2 as usize],
                    },
                    0,
                );
                *bytes2.write() = m2.to_bytes().unwrap();
            },
            n1,
            || {
                let b = bytes1.read();
                let m = ControlMessage::from_bytes(&b).unwrap();
                m.id as u32
            },
            n2,
            || {
                let b = bytes2.read();
                let m = ControlMessage::from_bytes(&b).unwrap();
                m.id as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }

    #[test]
    fn response_ok_is_constant() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        test_algorithm(
            "ControlResponse::ok",
            10,
            || {},
            n1,
            || {
                let r = ControlResponse::ok(42);
                r.request_id as u32
            },
            n2,
            || {
                let r = ControlResponse::ok(42);
                r.request_id as u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// module::config — ModuleConfig
// ═══════════════════════════════════════════════════════════════════

mod module_config {
    use super::*;
    use r0n_ingress::module::ModuleConfig;

    #[test]
    fn set_get_is_constant() {
        let n1: u32 = 50_000 * LOOP_MUL;
        let n2: u32 = 100_000 * LOOP_MUL;
        let cfg1 = parking_lot::RwLock::new(ModuleConfig::new());
        let cfg2 = parking_lot::RwLock::new(ModuleConfig::new());
        test_algorithm(
            "ModuleConfig set+get total",
            10,
            || {
                *cfg1.write() = ModuleConfig::new();
                *cfg2.write() = ModuleConfig::new();
            },
            n1,
            || {
                let mut c = cfg1.write();
                for i in 0..n1 {
                    c.set_string(format!("k{i:08}"), format!("v{i}"));
                }
                c.len() as u32
            },
            n2,
            || {
                let mut c = cfg2.write();
                for i in 0..n2 {
                    c.set_string(format!("k{i:08}"), format!("v{i}"));
                }
                c.len() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// module::contract — MetricsPayload
// ═══════════════════════════════════════════════════════════════════

mod metrics_payload {
    use super::*;
    use r0n_ingress::module::MetricsPayload;

    #[test]
    fn counter_gauge_insert_is_constant() {
        let n1: u32 = 50_000 * LOOP_MUL;
        let n2: u32 = 100_000 * LOOP_MUL;
        let payload1 = parking_lot::RwLock::new(MetricsPayload::new());
        let payload2 = parking_lot::RwLock::new(MetricsPayload::new());
        test_algorithm(
            "MetricsPayload counter insert",
            10,
            || {
                *payload1.write() = MetricsPayload::new();
                *payload2.write() = MetricsPayload::new();
            },
            n1,
            || {
                let mut p = payload1.write();
                for i in 0..n1 {
                    p.counter(format!("metric_{i}"), u64::from(i));
                }
                p.counters.len() as u32
            },
            n2,
            || {
                let mut p = payload2.write();
                for i in 0..n2 {
                    p.counter(format!("metric_{i}"), u64::from(i));
                }
                p.counters.len() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }

    #[test]
    fn to_prometheus_scales_linearly() {
        let n1: u32 = 5_000 * LOOP_MUL;
        let n2: u32 = 10_000 * LOOP_MUL;
        let payload1 = parking_lot::RwLock::new(MetricsPayload::new());
        let payload2 = parking_lot::RwLock::new(MetricsPayload::new());
        test_algorithm(
            "MetricsPayload::to_prometheus",
            10,
            || {
                let mut p1 = payload1.write();
                *p1 = MetricsPayload::new();
                for i in 0..n1 {
                    p1.counter(format!("c_{i}"), u64::from(i));
                    p1.gauge(format!("g_{i}"), f64::from(i));
                }
                let mut p2 = payload2.write();
                *p2 = MetricsPayload::new();
                for i in 0..n2 {
                    p2.counter(format!("c_{i}"), u64::from(i));
                    p2.gauge(format!("g_{i}"), f64::from(i));
                }
            },
            n1,
            || {
                let p = payload1.read();
                let s = p.to_prometheus("test");
                s.len() as u32
            },
            n2,
            || {
                let p = payload2.read();
                let s = p.to_prometheus("test");
                s.len() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// config::validation — ValidationResult
// ═══════════════════════════════════════════════════════════════════

mod validation_result {
    use super::*;
    use r0n_ingress::config::{ValidationError, ValidationResult};

    #[test]
    fn add_error_is_amortized_constant() {
        let n1: u32 = 50_000 * LOOP_MUL;
        let n2: u32 = 100_000 * LOOP_MUL;
        let result1 = parking_lot::RwLock::new(ValidationResult::new());
        let result2 = parking_lot::RwLock::new(ValidationResult::new());
        test_algorithm(
            "ValidationResult::add_error (amortized)",
            10,
            || {
                *result1.write() = ValidationResult::new();
                *result2.write() = ValidationResult::new();
            },
            n1,
            || {
                let mut r = result1.write();
                r.add_error(ValidationError::error("field", "msg"));
                r.errors().len() as u32
            },
            n2,
            || {
                let mut r = result2.write();
                r.add_error(ValidationError::error("field", "msg"));
                r.errors().len() as u32
            },
            BigOAlgorithmComplexity::O1,
            space(BigOAlgorithmComplexity::O1),
        );
    }

    #[test]
    fn merge_scales_linearly() {
        let n1: u32 = 10_000 * LOOP_MUL;
        let n2: u32 = 20_000 * LOOP_MUL;
        let result1 = parking_lot::RwLock::new(ValidationResult::new());
        let result2 = parking_lot::RwLock::new(ValidationResult::new());
        test_algorithm(
            "ValidationResult::merge",
            10,
            || {
                let mut base = ValidationResult::new();
                base.add_error(ValidationError::error("base", "base"));
                *result1.write() = base;
                let mut base2 = ValidationResult::new();
                base2.add_error(ValidationError::error("base", "base"));
                *result2.write() = base2;
            },
            n1,
            || {
                let mut other = ValidationResult::new();
                for i in 0..n1 {
                    other.add_error(ValidationError::error(format!("f_{i}"), format!("m_{i}")));
                }
                let mut r = result1.write();
                r.merge(other);
                r.errors().len() as u32
            },
            n2,
            || {
                let mut other = ValidationResult::new();
                for i in 0..n2 {
                    other.add_error(ValidationError::error(format!("f_{i}"), format!("m_{i}")));
                }
                let mut r = result2.write();
                r.merge(other);
                r.errors().len() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// config::schema — SchemaField
// ═══════════════════════════════════════════════════════════════════

mod config_schema {
    use super::*;
    use r0n_ingress::config::{ConfigSchema, SchemaField};

    #[test]
    fn with_property_is_amortized_constant() {
        let n1: u32 = 10_000 * LOOP_MUL;
        let n2: u32 = 20_000 * LOOP_MUL;
        test_algorithm(
            "SchemaField::with_property (amortized)",
            10,
            || {},
            n1,
            || {
                let mut obj = SchemaField::object("root");
                for i in 0..n1 {
                    obj = obj.with_property(SchemaField::string(&format!("field_{i}")));
                }
                obj.properties.map_or(0u32, |p| p.len() as u32)
            },
            n2,
            || {
                let mut obj = SchemaField::object("root");
                for i in 0..n2 {
                    obj = obj.with_property(SchemaField::string(&format!("field_{i}")));
                }
                obj.properties.map_or(0u32, |p| p.len() as u32)
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }

    #[test]
    fn to_json_schema_scales_with_fields() {
        let n1: u32 = 1_000 * LOOP_MUL;
        let n2: u32 = 2_000 * LOOP_MUL;
        let schema1 = parking_lot::RwLock::new(ConfigSchema::new("test", "1.0"));
        let schema2 = parking_lot::RwLock::new(ConfigSchema::new("test", "1.0"));
        test_algorithm(
            "ConfigSchema::to_json_schema (field count)",
            10,
            || {
                let mut s1 = ConfigSchema::new("test", "1.0");
                for i in 0..n1 {
                    s1 = s1.with_field(SchemaField::string(&format!("field_{i}")));
                }
                *schema1.write() = s1;
                let mut s2 = ConfigSchema::new("test", "1.0");
                for i in 0..n2 {
                    s2 = s2.with_field(SchemaField::string(&format!("field_{i}")));
                }
                *schema2.write() = s2;
            },
            n1,
            || {
                let s = schema1.read();
                let json = s.to_json_schema();
                format!("{json}").len() as u32
            },
            n2,
            || {
                let s = schema2.read();
                let json = s.to_json_schema();
                format!("{json}").len() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// config::loader — ConfigLoader
// ═══════════════════════════════════════════════════════════════════

mod config_loader {
    use super::*;
    use r0n_ingress::config::ConfigLoader;

    #[test]
    fn load_str_scales_linearly() {
        let n1: u32 = 1_000 * LOOP_MUL;
        let n2: u32 = 2_000 * LOOP_MUL;
        let toml1 = parking_lot::RwLock::new(String::new());
        let toml2 = parking_lot::RwLock::new(String::new());
        test_algorithm(
            "ConfigLoader::load_str (module count)",
            10,
            || {
                let mut s1 = String::from(
                    "[gateway]\nname = \"test\"\nbind_address = \"0.0.0.0\"\ncontrol_port = 9090\n\n[logging]\nlevel = \"info\"\n\n",
                );
                for i in 0..n1 {
                    let _ = write!(s1, "[[modules]]\nname = \"mod-{i}\"\ntype = \"tcp-router\"\nenabled = true\n\n");
                }
                *toml1.write() = s1;
                let mut s2 = String::from(
                    "[gateway]\nname = \"test\"\nbind_address = \"0.0.0.0\"\ncontrol_port = 9090\n\n[logging]\nlevel = \"info\"\n\n",
                );
                for i in 0..n2 {
                    let _ = write!(s2, "[[modules]]\nname = \"mod-{i}\"\ntype = \"tcp-router\"\nenabled = true\n\n");
                }
                *toml2.write() = s2;
            },
            n1,
            || {
                let t = toml1.read();
                let cfg = ConfigLoader::new().load_str(&t).unwrap();
                cfg.modules.len() as u32
            },
            n2,
            || {
                let t = toml2.read();
                let cfg = ConfigLoader::new().load_str(&t).unwrap();
                cfg.modules.len() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// module::manifest — ModuleManifest
// ═══════════════════════════════════════════════════════════════════

mod module_manifest {
    use super::*;
    use r0n_ingress::module::{Capability, Dependency, ModuleManifest};

    #[test]
    fn has_capability_is_constant() {
        let n1: u32 = 10_000 * LOOP_MUL;
        let n2: u32 = 20_000 * LOOP_MUL;
        let manifest1 =
            parking_lot::RwLock::new(ModuleManifest::builder("test").version(1, 0, 0).build());
        let manifest2 =
            parking_lot::RwLock::new(ModuleManifest::builder("test").version(1, 0, 0).build());
        test_algorithm(
            "ModuleManifest::has_capability",
            10,
            || {
                let mut b1 = ModuleManifest::builder("test").version(1, 0, 0);
                for i in 0..n1 {
                    b1 = b1.capability(Capability::Custom(format!("cap_{i}")));
                }
                *manifest1.write() = b1.build();
                let mut b2 = ModuleManifest::builder("test").version(1, 0, 0);
                for i in 0..n2 {
                    b2 = b2.capability(Capability::Custom(format!("cap_{i}")));
                }
                *manifest2.write() = b2.build();
            },
            n1,
            || {
                let m = manifest1.read();
                m.has_capability(&Capability::Custom("nonexistent".into())) as u32
            },
            n2,
            || {
                let m = manifest2.read();
                m.has_capability(&Capability::Custom("nonexistent".into())) as u32
            },
            BigOAlgorithmComplexity::O1,
            space(BigOAlgorithmComplexity::O1),
        );
    }

    #[test]
    fn required_dependencies_scales_linearly() {
        let n1: u32 = 10_000 * LOOP_MUL;
        let n2: u32 = 20_000 * LOOP_MUL;
        let manifest1 =
            parking_lot::RwLock::new(ModuleManifest::builder("test").version(1, 0, 0).build());
        let manifest2 =
            parking_lot::RwLock::new(ModuleManifest::builder("test").version(1, 0, 0).build());
        test_algorithm(
            "ModuleManifest::required_dependencies",
            10,
            || {
                let mut b1 = ModuleManifest::builder("test").version(1, 0, 0);
                for i in 0..n1 {
                    b1 = b1.dependency(Dependency::required(format!("dep_{i}")));
                }
                *manifest1.write() = b1.build();
                let mut b2 = ModuleManifest::builder("test").version(1, 0, 0);
                for i in 0..n2 {
                    b2 = b2.dependency(Dependency::required(format!("dep_{i}")));
                }
                *manifest2.write() = b2.build();
            },
            n1,
            || {
                let m = manifest1.read();
                m.required_dependencies().len() as u32
            },
            n2,
            || {
                let m = manifest2.read();
                m.required_dependencies().len() as u32
            },
            BigOAlgorithmComplexity::ON,
            space(BigOAlgorithmComplexity::ON),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// perf::connection_pool — PoolConfig
// ═══════════════════════════════════════════════════════════════════

mod pool_config {
    use super::*;
    use r0n_ingress::perf::PoolConfig;

    #[test]
    fn validate_is_constant() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        test_algorithm(
            "PoolConfig::validate",
            10,
            || {},
            n1,
            || {
                let cfg = PoolConfig::new().min_size(1).max_size(100);
                cfg.validate().is_ok() as u32
            },
            n2,
            || {
                let cfg = PoolConfig::new().min_size(1).max_size(100);
                cfg.validate().is_ok() as u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// module::contract — ContractVersion
// ═══════════════════════════════════════════════════════════════════

mod contract_version {
    use super::*;
    use r0n_ingress::module::ContractVersion;

    #[test]
    fn is_compatible_with_is_constant() {
        let n1: u32 = 100_000 * LOOP_MUL;
        let n2: u32 = 200_000 * LOOP_MUL;
        test_algorithm(
            "ContractVersion::is_compatible_with",
            10,
            || {},
            n1,
            || {
                let v1 = ContractVersion::new(1, 2, 3);
                let v2 = ContractVersion::new(1, 1, 0);
                v1.is_compatible_with(&v2) as u32
            },
            n2,
            || {
                let v1 = ContractVersion::new(1, 2, 3);
                let v2 = ContractVersion::new(1, 1, 0);
                v1.is_compatible_with(&v2) as u32
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// module::status — ModuleStatus
// ═══════════════════════════════════════════════════════════════════

mod module_status {
    use super::*;
    use r0n_ingress::module::ModuleStatus;

    #[test]
    fn status_checks_are_constant() {
        let n1: u32 = 200_000 * LOOP_MUL;
        let n2: u32 = 400_000 * LOOP_MUL;
        test_algorithm(
            "ModuleStatus::is_healthy + is_operational",
            10,
            || {},
            n1,
            || {
                let s = ModuleStatus::Running;
                (s.is_healthy() as u32) + (s.is_operational() as u32)
            },
            n2,
            || {
                let s = ModuleStatus::Running;
                (s.is_healthy() as u32) + (s.is_operational() as u32)
            },
            BigOAlgorithmComplexity::O1,
            SPACE_UNMEASURABLE,
        );
    }
}
