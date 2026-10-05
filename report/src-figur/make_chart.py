import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
with open("/home/claude/proj/project/project/results_openmp.csv", newline="") as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) for k, v in r.items()})

N      = [r["matrix_size"]            for r in rows]
s_omp1 = [r["speedup_1"]             for r in rows]
s_omp2 = [r["speedup_2"]             for r in rows]
s_omp4 = [r["speedup_4"]             for r in rows]
s_ompm = [r["speedup_max"]           for r in rows]
s_ocl  = [r["speedup_opencl_compute"] for r in rows]

plt.rcParams.update({"font.size": 11, "font.family": "serif",
                     "axes.grid": True, "grid.alpha": 0.35,
                     "grid.linestyle": "--"})
fig, ax = plt.subplots(figsize=(6.4, 4.2))

ax.plot(N, s_omp1, "o-",  color="#9e9e9e", label="OpenMP (1 thread)",  lw=1.6, ms=5)
ax.plot(N, s_omp2, "s-",  color="#1f77b4", label="OpenMP (2 thread)",  lw=1.8, ms=5)
ax.plot(N, s_omp4, "^-",  color="#2ca02c", label="OpenMP (4 thread)",  lw=2.0, ms=6)
ax.plot(N, s_ompm, "v--", color="#17becf", label="OpenMP (maks thread)", lw=1.8, ms=5)
ax.plot(N, s_ocl,  "D-",  color="#d62728", label="OpenCL (GPU, komputasi)", lw=2.0, ms=5)

ax.axhline(1.0, color="black", lw=1.0, alpha=0.7)
ax.text(60, 1.04, "ambang impas (speedup = 1)", fontsize=8.5, color="black")

ax.set_xscale("log")
ax.set_xticks(N)
ax.set_xticklabels([str(int(n)) for n in N])
ax.minorticks_off()
ax.set_xlabel("Ukuran matriks MNA  $n$  (skala log)")
ax.set_ylabel(r"Speedup terhadap sekuensial  ($t_\mathrm{seq}/t$)")
ax.set_title("Speedup tiga backend terhadap baseline sekuensial", fontsize=11.5)
ax.legend(loc="upper left", fontsize=8.8, framealpha=0.95)
ax.set_ylim(0, 2.9)

fig.tight_layout()
fig.savefig("fig/speedup.pdf", bbox_inches="tight")
print("saved fig/speedup.pdf")
