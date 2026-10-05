# Diagram Mermaid — Solver MNA Paralel

Empat diagram di bawah dapat dirender langsung di GitHub, VS Code (ekstensi
Mermaid), Notion, Obsidian, atau di <https://mermaid.live>.

---

## 1. DFD Level 0 (Diagram Konteks)

```mermaid
flowchart LR
    User["Pengguna"]
    SPICE["LTspice<br/>(perangkat referensi)"]
    Sys((("0<br/>Sistem Solver<br/>MNA Paralel")))

    User -->|"deskripsi rangkaian, N, pilihan backend"| Sys
    Sys -->|"tegangan node, arus cabang, speedup"| User
    Sys -->|"netlist *.net"| SPICE
    SPICE -->|"tegangan referensi (.op)"| Sys
```

---

## 2. DFD Level 1

```mermaid
flowchart LR
    User["Pengguna"]
    SPICE["LTspice"]

    P1(("1<br/>Bangun<br/>Rangkaian"))
    P2(("2<br/>Susun<br/>Matriks MNA"))
    P3(("3<br/>Selesaikan<br/>Sistem Linear"))
    P4(("4<br/>Verifikasi &<br/>Laporkan"))

    D1[("D1 — Kernel GPU *.cl")]
    D2[("D2 — results_openmp.csv")]
    D3[("D3 — Netlist *.net")]

    User -->|"parameter N, jenis rangkaian"| P1
    P1 -->|"objek Circuit (komponen)"| P2
    P2 -->|"matriks A, vektor b"| P3
    P2 -.->|"ekspor netlist"| D3
    D3 -.-> SPICE
    D1 -.->|"sumber kernel"| P3
    P3 -->|"solusi x, waktu solve"| P4
    P4 -->|"x terverifikasi, speedup"| User
    P4 -.->|"baris hasil"| D2
    SPICE -.->|"V referensi"| P4
```

---

## 3. Flowchart — Algoritma Solver LU

Blok hijau adalah **wilayah paralel** yang diimplementasikan berbeda pada tiap
backend (sekuensial / OpenMP / OpenCL).

```mermaid
flowchart TD
    Start(["Mulai"]) --> Build["Bangun Circuit & susun MNA<br/>menghasilkan A, b berukuran n"]
    Build --> Init["k = 0"]
    Init --> Loop{"k &lt; n - 1 ?"}
    Loop -->|tidak| Subst["Substitusi maju (Ly = b)<br/>lalu mundur (Ux = y) menghasilkan x"]
    Loop -->|ya| Pivot["Cari pivot: baris dgn magnitudo<br/>A[i][k] terbesar, i &ge; k<br/>(hanya jalur CPU; GPU melewati)"]
    Pivot --> Sing{"magnitudo pivot<br/>&lt; 1e-14 ?"}
    Sing -->|ya| Err["Matriks singular —<br/>kembalikan galat"]
    Err --> End(["Selesai"])
    Sing -->|tidak| Swap["Tukar baris k dan pivot_row (CPU)"]
    Swap --> Elim["ELIMINASI BARIS  i = k+1 .. n-1<br/>factor = A[i][k] / A[k][k]<br/>A[i][j] -= factor * A[k][j]<br/>—— WILAYAH PARALEL ——<br/>Sekuensial: loop biasa<br/>OpenMP: #pragma omp parallel for schedule(dynamic)<br/>OpenCL: kernel compute_factors (1D) + lu_eliminate (2D)"]
    Elim --> Inc["k = k + 1"]
    Inc --> Loop
    Subst --> Resid["Cek residu: norm tak-hingga dari (A x - b)<br/>verified = (residu &lt; 1e-6)"]
    Resid --> Out["Keluarkan x, time_ms, verified"]
    Out --> End

    style Elim fill:#e7f6ec,stroke:#2ca02c,stroke-width:2px
    style Start fill:#dfeaf6
    style End fill:#dfeaf6
    style Err fill:#fbe4e4
```

---

## 4. (Bonus) Arsitektur Sistem

```mermaid
flowchart TD
    subgraph IN["Lapisan Masukan"]
        builders["Pembangkit Rangkaian<br/>makeLadder() · makeVoltageDivider()<br/>makeSimpleLoop() · generateMesh()"]
        netlist["Berkas netlist *.net<br/>(format LTspice)"]
    end
    subgraph MODEL["Lapisan Model"]
        circuit["Circuit<br/>resistor · sumber tegangan · sumber arus"]
        mna["MNA::build()<br/>stamping G &amp; cabang sumber tegangan<br/>menghasilkan A x = b"]
    end
    subgraph SOLVE["Lapisan Komputasi (Solver)"]
        seq["SequentialSolver<br/>LU + partial pivoting (1 inti CPU)"]
        omp["OpenMPSolver<br/>LU + pivoting; eliminasi paralel (multi-inti)"]
        ocl["OpenCLSolver<br/>LU tanpa pivoting (GPU); 2 kernel per kolom-k"]
        kern["Kernel GPU<br/>lu_factor.cl · lu_eliminate.cl"]
    end
    subgraph OUT["Lapisan Keluaran &amp; Verifikasi"]
        result["SolverResult<br/>x (tegangan node + arus cabang)<br/>time_ms · verified"]
        resid["Cek residu<br/>norm tak-hingga (A x - b) &lt; 1e-6"]
        csv["results_openmp.csv<br/>(sweep speedup)"]
    end

    builders --> circuit
    netlist -.->|ekspor/impor| circuit
    circuit --> mna
    mna --> seq
    mna --> omp
    mna --> ocl
    kern -.-> ocl
    seq --> result
    omp --> result
    ocl --> result
    result --> resid
    result -.-> csv
```
