#!/usr/bin/env python3
"""Apply one named ablation to bench.cu. Each edit is deliberately small and
explicit so the instruction stream stays comparable to the baseline."""
import sys, re, os

HERE = os.environ.get("MEAS_K_DIR", ".")

name = sys.argv[1]
src = open(os.path.join(HERE, "bench.cu")).read()

A_IDX = "ty * TM + i + gemm_kshift(SWIZ, p)"
B_IDX = "gemm_col_slot<BN>(tx, j) + gemm_kshift(SWIZ, p)"
A_LOAD = "const float4 t = *reinterpret_cast<const float4*>(&ac[p][" + A_IDX + "]);"
B_LOAD = ("const float4 t =\n                    *reinterpret_cast<const float4*>"
          "(&bc[p][" + B_IDX + "]);")

def need(sub):
    if sub not in src:
        sys.exit(f"ablate: pattern not found for {name!r}:\n{sub}")

if name == "baseline":
    pass

elif name == "nobarrier":
    # Drop only the in-loop barrier (the one inside `if (more)`), keeping the
    # prologue barrier. Result is numerically wrong; we are timing the stream.
    need("            __syncthreads();\n            cur = nxt;")
    src = src.replace("            __syncthreads();\n            cur = nxt;",
                      "            cur = nxt;")

elif name == "noglobal":
    # Remove the four prefetch LDGs; the shared stores then write stale
    # registers. Isolates global-load latency/L2 pressure.
    for ln in ["            if (a_ok) av = *reinterpret_cast<const float4*>(ab + kt + BK);",
               "            if (a_ok2) av2 = *reinterpret_cast<const float4*>(ab2 + kt + BK);",
               "            if (b_ok) bv = *reinterpret_cast<const float4*>(bb + kt + BK);",
               "            if (b_ok2) bv2 = *reinterpret_cast<const float4*>(bb2 + kt + BK);"]:
        need(ln)
        src = src.replace(ln + "\n", "")

elif name == "bcast_shared":
    # Same LDS instruction count, but every lane of a warp reads the same
    # address, so each LDS.128 costs the broadcast minimum instead of 2 (A) /
    # 4 (B) wavefronts. Isolates shared-memory *traffic* from instruction count.
    need(A_LOAD); need(B_LOAD)
    src = src.replace(A_LOAD, A_LOAD.replace("[" + A_IDX + "]", "[i]"))
    src = src.replace(B_LOAD, B_LOAD.replace("[" + B_IDX + "]", "[j]"))

elif name == "no_shared_st":
    # Remove the 16 in-loop shared stores (keeps the prologue staging).
    m = re.search(r"(            const int nxt = cur \^ 1;\n)((?:            (?:as|bs)\[nxt\]\[[^\n]*\n)+)", src)
    if not m:
        sys.exit("ablate: in-loop STS block not found")
    src = src[:m.start(2)] + src[m.end(2):]

elif name == "ffma_only":
    # No shared reads at all: operands come from registers derived from kt, so
    # nothing folds. Gives the pure FFMA-issue ceiling for this loop shape.
    need(A_LOAD); need(B_LOAD)
    src = src.replace(A_LOAD,
        "const float4 t = make_float4(__int_as_float(kt + i), __int_as_float(kt + i + 1),"
        " __int_as_float(kt + i + 2), __int_as_float(kt + i + 3));")
    src = src.replace(B_LOAD,
        "const float4 t = make_float4(__int_as_float(kt + j + 5), __int_as_float(kt + j + 6),"
        " __int_as_float(kt + j + 7), __int_as_float(kt + j + 8));")
else:
    sys.exit(f"unknown ablation {name!r}")

open(os.path.join(HERE, f"bench_{name}.cu"), "w").write(src)
print(f"wrote bench_{name}.cu")
