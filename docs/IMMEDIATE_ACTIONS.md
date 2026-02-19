# 🚀 Immediate Actions — Phase 3 Kickoff

**Created:** February 19, 2026  
**Priority:** HIGH  
**Timeline:** Start this week

---

## 🎯 What We're Building

Upgrade `mlcli suggest` from rules to a **Meta-ML Recommendation Engine** that learns from real ML projects.

**BUT FIRST:** We need to fix 4 critical infrastructure gaps that will block us later.

---

## ⚠️ Critical Gaps That Must Be Fixed FIRST

### 1. **Preprocessing Pipeline Persistence** (BLOCKER)
**Problem:** We train models on transformed data, but predictions use raw data.  
**Impact:** Predictions are WRONG in production.  
**Fix:** Save the fitted scikit-learn pipeline in `preprocess`, load it in `predict`.  
**Time:** 1 day  
**Priority:** P0 — DO THIS FIRST

### 2. **Schema Validation** (BLOCKER) 
**Problem:** JSON artifacts can silently break (wrong keys, types).  
**Impact:** Meta-ML model will train on garbage data.  
**Fix:** Create Pydantic schemas for all artifacts.  
**Time:** 2 days  
**Priority:** P0

### 3. **Artifact Versioning** (BLOCKER)
**Problem:** No way to track which data/config produced which model.  
**Impact:** Can't debug issues or ensure reproducibility.  
**Fix:** Build an `ArtifactTracker` that logs all generated files.  
**Time:** 2 days  
**Priority:** P0

### 4. **Telemetry Collection** (BLOCKER for learning loop)
**Problem:** No way to know if suggestions were helpful.  
**Impact:** Can't improve the model with real data.  
**Fix:** Add a `TelemetryCollector` that logs events locally.  
**Time:** 2 days  
**Priority:** P0

---

## 📋 Week 1 Sprint Plan

### Day 1-2: Infrastructure Foundation
**Goal:** Fix preprocessing pipeline persistence + start schemas

**Tasks:**
1. **Modify `mlcli/core/data.py`:**
   - Add `DataProcessor.save_pipeline()` method
   - Add `DataProcessor.load_pipeline()` classmethod
   - Save to `data/processed/preprocessing_pipeline.pkl`

2. **Modify `mlcli/commands/preprocess_cmd.py`:**
   - Call `processor.save_pipeline()` after preprocessing

3. **Modify `mlcli/commands/predict_cmd.py`:**
   - Load pipeline: `processor = DataProcessor.load_pipeline(...)`
   - Apply: `X_transformed = processor.transform_new_data(input_df)`

4. **Test it:**
   ```bash
   cd test_workspace
   mlcli preprocess --input data/raw/data.csv --target target
   mlcli predict --input data/raw/new_data.csv  # Should use saved pipeline
   ```

**Deliverable:** Predictions now use correct preprocessing.

### Day 3-4: Schema Validation
**Goal:** Pydantic schemas for all artifacts

**Tasks:**
1. **Create `mlcli/core/schemas/` package:**
   - `base.py` — `VersionedArtifact` base class
   - `data_profile.py` — `DataProfileSchema`
   - `training_summary.py` — `TrainingSummarySchema`
   - `evaluation_report.py` — `EvaluationReportSchema`

2. **Update artifact write locations:**
   - `preprocess_cmd.py` — validate before saving
   - `train_cmd.py` — validate before saving
   - `evaluate_cmd.py` — validate before saving

3. **Update artifact read locations:**
   - `suggest_cmd.py` — load with validation

**Deliverable:** All artifacts are type-safe with automatic validation.

### Day 5: Artifact Tracking
**Goal:** Track all generated artifacts with metadata

**Tasks:**
1. **Create `mlcli/core/versioning/artifact_tracker.py`:**
   - `ArtifactTracker` class with registry in `.mlcli/artifact_registry.json`
   - Methods: `register_artifact()`, `get_lineage()`, `list_artifacts()`

2. **Integrate into commands:**
   - Each command registers artifacts it creates
   - Include checksums, timestamps, parent artifacts

**Deliverable:** Full audit trail of all artifacts.

---

## 📋 Week 2 Sprint Plan

### Day 6-7: Telemetry System
**Goal:** Privacy-first telemetry collection

**Tasks:**
1. **Create `mlcli/core/telemetry/` package:**
   - `collector.py` — `TelemetryCollector` class
   - `privacy.py` — PII filtering utilities

2. **Integrate into suggest command:**
   - Log when suggestions are shown
   - Log which suggestions user acted on (track via session ID)

**Deliverable:** Telemetry flows locally, ready for meta-ML training data.

### Day 8-10: Knowledge Base Generation
**Goal:** Create synthetic training data for meta-ML model

**Tasks:**
1. **Create `mlcli/meta_ml/knowledge_base.py`:**
   - `KnowledgeBaseGenerator` class
   - Expert-labeled scenarios (10k+)

2. **Generate and validate:**
   - Run: `python scripts/generate_knowledge_base.py`
   - Manually review 100 samples
   - Ensure label distribution makes sense

**Deliverable:** `data/meta_ml/synthetic_kb.csv` with 10k training examples.

---

## 📋 Week 3-4 Sprint Plan

### Days 11-17: Meta-ML Engine
**Goal:** Train and integrate the recommendation model

**Tasks:**
1. **Create `mlcli/meta_ml/features.py`:**
   - Feature extraction from artifacts
   - 48-dimensional feature vector

2. **Create `mlcli/meta_ml/engine.py`:**
   - `MetaMLEngine` with XGBoost multi-label classifier
   - Confidence calibration (isotonic regression)

3. **Train model:**
   - Run: `python scripts/train_meta_ml.py`
   - Evaluate on holdout set
   - Save to `data/meta_ml/suggestion_model_v1.pkl`

4. **Integrate into suggest command:**
   - Load model, extract features, predict
   - Display with confidence bars
   - Graceful fallback to rules

**Deliverable:** AI-powered suggestions with confidence scores.

### Days 18-20: Testing & Documentation
**Goal:** Production-ready quality

**Tasks:**
1. Write unit tests for all new modules (target: 80% coverage)
2. Update README with Phase 3 features
3. Create user guide for interpreting suggestions
4. Beta test with 5 real users

**Deliverable:** Production-ready Phase 3.

---

## 🎯 Success Criteria (Must Pass Before Shipping)

- [ ] Predictions use correct preprocessing pipeline (verify with test)
- [ ] All artifacts pass Pydantic validation
- [ ] Artifact tracker logs show complete lineage
- [ ] Telemetry captures 10+ event types without PII
- [ ] Knowledge base: 10k+ samples, >90% expert review accuracy
- [ ] Meta-ML model: <0.15 Hamming loss on test set
- [ ] Suggest command shows confidence scores with visual bars
- [ ] Graceful fallback to rules if ML model unavailable
- [ ] Test coverage >75% for new modules
- [ ] 3 beta users confirm suggestions are helpful

---

## 🚦 Go/No-Go Decision Points

### End of Week 1
**Check:** Is preprocessing pipeline persistence working?  
**If NO:** Block all other work, fix this first.

### End of Week 2
**Check:** Are schemas catching validation errors?  
**If NO:** Don't proceed to meta-ML, fix schemas first.

### End of Week 3
**Check:** Does meta-ML model beat rule-based on 20 test cases?  
**If NO:** Improve feature engineering or add more synthetic data.

---

## 💡 Quick Wins You Can Ship Early

### Week 1 Quick Win: Better Preprocessing
**Ship:** Pipeline persistence + improved predict command  
**Value:** Predictions are now correct  
**Effort:** 2 days

### Week 2 Quick Win: Validated Artifacts
**Ship:** Pydantic schemas + better error messages  
**Value:** Fewer silent failures, easier debugging  
**Effort:** 3 days

### Week 3 Quick Win: Basic Meta-ML
**Ship:** Meta-ML suggestions (even if accuracy is 60%)  
**Value:** Shows the vision, collects telemetry  
**Effort:** 5 days

---

## 📞 Who Should Work on What?

### Senior ML Engineer
- Knowledge base generation (Week 2)
- Meta-ML model training (Week 3)
- Feature engineering (Week 3)

### Senior Backend Engineer
- Preprocessing pipeline persistence (Week 1)
- Schema validation (Week 1-2)
- Artifact tracking (Week 2)
- Telemetry system (Week 2)

### Full-Stack Engineer (Optional)
- UI/UX for confidence visualization (Week 4)
- Testing and documentation (Week 4)

---

## 🎬 How to Start RIGHT NOW

```bash
# 1. Create a feature branch
git checkout -b phase3-infrastructure

# 2. Start with the BLOCKER
# Open mlcli/core/data.py
# Add save_pipeline() and load_pipeline() methods

# 3. Test it locally
cd test_workspace
mlcli preprocess --input data/raw/data.csv --target target
# Check: data/processed/preprocessing_pipeline.pkl exists

mlcli predict --input data/raw/data.csv --output predictions.csv
# Check: predictions use the loaded pipeline

# 4. Commit and continue
git add .
git commit -m "feat: add preprocessing pipeline persistence (P0 blocker fix)"
```

---

## 📚 Reference

- **Full Plan:** `docs/phase3_meta_ml_plan.md`
- **Product Doc:** `docs/product_document.md`
- **Current Code:** `mlcli/commands/suggest_cmd.py` (rule-based)

---

**Let's build this the RIGHT way — infrastructure first, then intelligence.**
