How to compile RouteRL report (2 pages)
======================================
Local:
  cd report
  pdflatex report.tex
  pdflatex report.tex

Overleaf:
  1. New Project -> Upload
  2. Upload report.tex AND the figures/ folder (four PNG files)
  3. Compiler: pdfLaTeX, Recompile twice

Required files:
  report.tex
  figures/ablation_curves.png
  figures/load_sweep.png
  figures/routing_heatmap.png
  figures/policy_comparison.png

Output: report.pdf (committed as routeRL_report.pdf)
