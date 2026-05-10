# Sprint 1 — Foundations & Data Pipeline (Weeks 1–2)

**Scrum Master**: P4 — Rishikesh Aluguvelli  
**Sprint dates**: _fill in_

---

### What was completed

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Create GitHub repo structure | P4 | ⬜ | Folders, branch rules, issue templates |
| Download & verify Dresden + VISION | P4 | ⬜ | Download scripts written |
| Build device-stratified splits | P4 | ⬜ | splits.json — READ ONLY after this |
| Implement wavelet denoising filter | P1 | ⬜ | Lukas et al. Appendix A |
| Implement noise residual extractor | P1 | ⬜ | W = I - F(I) per channel |
| Implement camera fingerprint K_d | P1 | ⬜ | MLE estimator |
| Setup MLflow / W&B tracking | P4 | ⬜ | |
| Write pytest unit tests for PRNU | P1 | ⬜ | |
| Establish environment.yml | P4 | ⬜ | Pinned dependencies |

### What's next (Sprint 2)

- [ ] Implement NCC classical detector — P1
- [ ] Run Experiment A1 — P3
- [ ] Build patch extraction module — P2
- [ ] Build ResNet-18 CNN — P2
- [ ] Train CNN baseline — P2
- [ ] Evaluation harness (FAR/FRR) — P3

### Blockers & risks

| Blocker | Impact | Mitigation | Owner |
|---------|--------|------------|-------|
| Dataset access delays | Can't start training | Use synthetic data for dev | P4 |
| GPU access | Slow CNN training | Kaggle Notebooks as backup | P2 |

---

*Journal submitted: _fill in_*
