# Kazakh Profanity Detection in Product Reviews

DSSC5113 bonus project (KBTU, Fall 2026). Author: **Alibek Ramazanov, student ID 25MD0196**.

A fine-tuned **XLM-RoBERTa-base** classifier reads one Kazakh review or comment and predicts:

| label | meaning |
|---|---|
| `0` | no Kazakh profanity |
| `1` | Kazakh profanity present |

Each prediction also comes with `p(profanity)`, the model's probability for class 1, and the decision threshold that was chosen on the validation set.

> **Content warning.** The data, the lexicon in `src/lexicon.py` and the examples below contain obscene Kazakh words. They are included only because the task is to detect them.

---

## 1. Quick start: predict without retraining

```bash
git clone https://github.com/<github-user>/kazakh-profanity-detection.git
cd kazakh-profanity-detection
pip install -r requirements.txt
# get the trained model (see "Model files" below), then:
python -m src.predict --model_dir outputs/model --text "Өте жақсы тауар, рахмет!" --text "Сапасы қотақ, бағасына татымайды"
python -m src.predict --model_dir outputs/model          # interactive: type comments, empty line to quit
```

From Python or a notebook cell:

```python
from src.predict import ProfanityDetector
det = ProfanityDetector("outputs/model")        # or a Hugging Face repo id
det.predict(["Сатушы алаяқ, тауар келмеді", "Шешеңнің амы, курьер кешікті"])
# [{'label': 0, 'label_name': 'clean', 'prob_profanity': ..., 'threshold': ...}, {'label': 1, ...}]
```

**Model files** (weights are stored outside GitHub because they are about 1.1 GB):
`<link to the Hugging Face repo or GitHub Release model.zip — filled in after training>`.
Unzip `model.zip` into `outputs/model/`. The folder holds `model.safetensors`, `config.json`, the tokenizer files and `inference_config.json` (threshold, maximum length, truncation settings).

---

## 2. Repository layout

```
configs/train_config.json        all training settings and the random seed
data/manual_review.csv           my decisions for every ambiguous lexicon match (414 comments)
data/processed/{train,val,test}.csv   final splits (seed 42) + dataset_stats.json
data/challenge/challenge_set.csv 48 hand-written marketplace comments (synthetic, evaluation only)
notebooks/train_kaggle_colab.ipynb    end-to-end run in a fresh Kaggle / Colab GPU runtime
scripts/get_kazsandra.sh         downloads the original KazSAnDRA files
src/lexicon.py                   rule-based pre-labelling (weak labels), NOT part of the model
src/prepare_data.py              cleaning, labelling, de-duplication, splits, augmentation
src/text_utils.py                preprocessing shared by training and inference
src/model_utils.py               text -> model input, batching, probabilities
src/train.py                     fine-tuning + model/threshold selection on validation
src/evaluate.py                  one-time test evaluation, confusion matrix, error lists
src/predict.py                   inference CLI / Python class for new comments
outputs/results/                 saved metrics, confusion matrix, predictions, errors
```

---

## 3. Data

### 3.1 Source and how it relates to product reviews

The source is **KazSAnDRA** (Kazakh Sentiment Analysis Dataset of Reviews and Attitudes; Yeshpanov & Varol, LREC-COLING 2024), licensed CC BY 4.0, from <https://github.com/IS2AI/KazSAnDRA>. It contains 180,064 Kazakh customer reviews with 1–5 star scores from four domains:
- **Market:** online marketplaces, i.e. product reviews.
- **Appstore:** app reviews.
- **Mapping:** reviews of places and businesses.
- **Bookstore:** book reviews.

All four are customer reviews of goods or services, written by ordinary users. They mix Kazakh with Russian, use Russian letters for Kazakh sounds (к for қ) and sometimes Latin script, which is exactly the input a review moderator sees.

I used the three score-classification files: `06_sc_train_ib`, `09_sc_valid` and `10_sc_test`, 175,158 rows in total. To download them:

```bash
bash scripts/get_kazsandra.sh data/raw
```

**Important property of the source.** Every one of the 1,482 real profane comments comes from the *Appstore*, *Mapping* and *Bookstore* domains (almost all from Appstore). The *Market* reviews contain **no** profanity, presumably because marketplaces already moderate them. The model therefore learns profanity mostly from app reviews and is applied to product reviews. Two consequences:
1. 8,007 clean Market reviews are deliberately included, so the model sees what normal product reviews look like.
2. A small hand-written product-review **challenge set** (section 3.7) is evaluated separately.

### 3.2 What counts as Kazakh profanity: labelling rules

**Label 1** means the comment contains at least one **obscene or vulgar word of Kazakh origin** used as an obscenity. Script, spelling and masking do not matter. This covers:
- genital and sexual vulgarisms (қотақ, ам, амшық, сігу and its curse forms сігейін, сігіл, сіктім...);
- vulgar curse formulas: шешеңнің амы, шешеңді сігейін, әкеңнің аузын ...;
- vulgar slurs: жалап, қаншық, ташақ;
- vulgar "ass" and "shit" words used to insult or evaluate: көт, көтен, боқ, as in «боқ ойын», «көт программа», «қолдарыңнан боқ та келмейді».

The rules for hard cases:

| Case | Rule |
|---|---|
| **Negative review** | Not profanity by itself. «Сапасы нашар, ақшамды қайтарыңдар» → 0. Star scores and sentiment are **not** used as labels. |
| **Insults without obscene words** | 0. ақымақ, мал, есек, топас, алаяқ are rude but not obscene. |
| **Mixed-language comments** | Only Kazakh obscene vocabulary makes a comment 1. A comment whose only obscenities are Russian (бля, сука, пиздец, зайбал...) is labelled **0** and flagged `ru_profanity=1`, because the task is *Kazakh* profanity. Its false-positive rate is reported separately. If both Kazakh and Russian obscenities are present → 1. |
| **Spelling variants** | Count. Russian-keyboard spellings (котак, сигейн, шешенын амы), Latin transliteration (qotaq, kotakbas, sheshen amy, Kazakh Latin 2021 letters ı ǵ ń...), repeated letters (қотааааақ), glued forms (шешесігейін). |
| **Masked words** | Count when the visible letters make the word recognisable: қ*тақ, қ***қ, с*гейін, ж*лаптар, қот@қ. A masked ordinary word (ж*қсы) stays 0. |
| **Meta use** | 0. Talking *about* swearing without swearing: «боқтамаңдар», «боқ ауыз сөздер көп», «кейбір адамдар боқтайды». |
| **Literal or anatomical use** | 0. «баланың көтенін құрғақ ұстайды» (a diaper review), literal hygiene descriptions with боқ. |
| **Homographs** | 0 when the word means something else: кот = SMS code («кот келмеді») or cat, скин = game skin, сгин = Russian «сгинь», амын = a typo of ...амын, боксын = boxing, сиқыр, есік, сигнал, аман, амал, көтеру. |
| **Undecidable** | Dropped from the data. 16 comments: unclear meaning, gibberish, possible typos. |

### 3.3 Labelling method (semi-automatic, with manual review)

1. **Lexicon pre-labelling** (`src/lexicon.py`). The text is mapped to a spelling-insensitive "skeleton":
   - қ/к/q/k → к, ғ/г/g → г, і/и/i → и, ә/a → а, and so on;
   - repeated letters are collapsed.

   Two kinds of regular-expression rules then run on the skeleton:
   - **strong rules** (unambiguous obscenities, plus masked-word matching) set label 1 directly;
   - **ambiguous rules** (боқ, кот/көт, көтен, ам/амы, скин/сгин, жалап, боқтау) only mark a comment for manual review.
2. **Manual review.** I read all **414** comments that matched only ambiguous rules and labelled each one by the rules above: 260 → 1, 138 → 0, 16 dropped. The decisions are in `data/manual_review.csv`, keyed by `custom_id`.
3. **Audits.**
   - About 160 strong-rule matches were read, spread over all rules. No false matches were found.
   - 200 random 1-star comments with no rule match were read to look for missed profanity. One was missed (the glued form «шешессігейн»). The curse-verb rule was widened, and the new matches it produced were read; all were correct.
4. **Rows without any rule match → 0.**

Every row keeps `label_method` (`rule_strong` / `manual` / `rule_negative` / `synthetic`), `rule_hits`, `ambiguous_hits` and `ru_profanity`, so each label can be traced.

**Known weakness:** profanity that no rule catches (new slang, heavy misspellings) is labelled 0 by default. Section 6 shows how this affects the "false positives".

### 3.4 Cleaning, privacy and duplicates

- **Personal information.** URLs → `<url>`, e-mails → `<email>`, @handles → `<user>`, phone-like digit sequences (7 or more digits) → `<phone>`. Reviewer names are not part of KazSAnDRA; only an anonymous `custom_id` is kept. First names mentioned inside review texts were not removed (see Limitations).
- **Exact duplicates** (after lower-casing and whitespace normalisation): 2,426 removed.
- **Near-duplicates.** Comments are grouped when their letter skeletons are identical («Қотақбас!!!» = «котакбас»), or when MinHash LSH over character 4-grams finds Jaccard similarity of at least 0.8. MinHash LSH estimates text overlap quickly without comparing every pair. This gives 235 groups covering 617 comments. Each group is placed entirely in one split.

### 3.5 Sampling and splits

From the labelled pool the data keeps:
- **all** positives;
- all **hard negatives**: every manually reviewed 0 and every Russian-obscenity-only comment;
- **8,000** random clean Market reviews;
- **10,000** random clean reviews from the other domains.

The split is grouped and stratified with `StratifiedGroupKFold(10 folds, shuffle, seed 42)`:
- fold 0 → test;
- fold 1 → validation;
- folds 2–9 → train.

`prepare_data.py` is deterministic: rebuilding from the original files reproduces the CSVs byte for byte, which was checked under different `PYTHONHASHSEED` values.

| split | total | profanity (1) | clean (0) | of which synthetic |
|---|---|---|---|---|
| train | 17,120 | 1,635 | 15,485 | 899 |
| validation | 2,028 | 148 | 1,880 | 0 |
| test | 2,028 | 148 | 1,880 | 0 |
| **total** | **21,176** | **1,931** | **19,245** | **899** |

Real (non-synthetic) comments: **20,277**, of which 1,482 are profane (7.3 %) and 18,795 clean.
By domain: Appstore 11,344, Market 8,007, Mapping 564, Bookstore 362.
The test set contains 59 Russian-obscenity-only clean comments and 36 manually labelled ambiguous comments.

```bash
python -m src.prepare_data --kazsandra_dir data/raw --out_dir data/processed --seed 42
```

### 3.6 Synthetic examples (train only, flagged)

`is_synthetic=1`, with the type in `synthetic_type`. Validation and test contain **no** synthetic rows.

| type | n | label | how it is made |
|---|---|---|---|
| `masked_profanity` | 299 | 1 | the profane word of a real train comment is partly masked: қотақбас → қо#*қбас |
| `masked_clean` | 300 | 0 | an ordinary word of a clean comment is masked, so that `*` alone is not learned as a profanity cue |
| `latin_translit` | 300 | 1 / 0 | the whole comment is transliterated to Latin script (150 per class) |

### 3.7 Challenge set (synthetic, evaluation only)

`data/challenge/challenge_set.csv` holds 48 short **marketplace-style** comments that I wrote with AI assistance:
- 24 profane, including masked, Latin-script, inflected and "positive review with a swear word" cases;
- 24 clean hard negatives: angry but clean reviews, a diaper review with «көтенін», «код келмеді», «скин», Russian-only obscenity, near-homographs (қотыр, амал, аман, сигнал, көтеру, есік, сиқыр).

These comments were never used for training or for choosing any setting.

### 3.8 Labelled examples (real data)

| text | label | why |
|---|---|---|
| Керемет. Каспий дүкенінің қызметкерлеріне рахмет! | 0 | positive product review |
| Тікпейді, сапасы төмен. | 0 | negative review, no profanity |
| Қотақ ойын екен | 1 | Kazakh obscenity |
| Қысқасаын айтқанда к\*т приложения екен дұрыс істемейді | 1 | masked vulgar word |
| Прост көті боқ | 1 | vulgar evaluation (manual review) |
| Ойынға сөз жоқ. Тек адамдардың боқ ауыз сөздері жаман екен. | 0 | talks about swearing, meta (manual review) |
| Жақсы бірақ скин қосыңдаршы. | 0 | «скин» = game skin (manual review) |
| Ойын пиздец қо базар жоқ | 0 | Russian obscenity only, out of scope, `ru_profanity=1` |

---

## 4. Model

### 4.1 Why XLM-RoBERTa-base

- XLM-RoBERTa was pretrained on CommonCrawl text in 100 languages, including **Kazakh and Russian**, so it already "knows" both languages of these code-switched reviews.
- Its SentencePiece subword vocabulary has no out-of-vocabulary problem: an unseen or misspelled word (қотааақбас, котакбас, qotaqbas) is split into known pieces instead of becoming one unknown token.
- With about 278 M parameters it is small enough to fine-tune in roughly 20 minutes on one free Kaggle or Colab GPU.
- The KazSAnDRA authors fine-tuned multilingual transformers, XLM-R among them, on these same reviews for sentiment analysis.
- A character n-gram linear model would be cheaper. However, it cannot use context, and context is what separates «кот келмеді» (the code did not arrive) from «кот ойын» (vulgar).

### 4.2 From raw text to model input

```
raw comment
 └─ normalize()  (src/text_utils.py)
      ├─ remove URLs / e-mails / @handles / phone numbers
      ├─ shorten character floods: "қотааааақ!!!!!!" → "қотааақ!!!"
      └─ collapse whitespace;  case, punctuation, emoji, * @ # are KEPT
 └─ SentencePiece tokenizer of XLM-R → subword ids (punctuation, emoji and masks become their own pieces)
 └─ if more than 126 subwords: keep the first 64 + the last 62 (head + tail truncation)
 └─ add <s> … </s>, pad within the batch, attention mask
 └─ XLM-R encoder → <s> vector → classification head → 2 logits → softmax → p(profanity)
 └─ label = 1 if p(profanity) ≥ threshold (chosen on validation)
```

- **Long comments.** Only a small share of comments is longer than 128 subwords; the exact counts are printed during training and saved in `train_history.json`. Swearing is usually at the start or the end of a review, so head + tail keeps more of it than cutting the tail.
- **Punctuation.** Punctuation is kept, because masks (`*`, `@`) and «!!!» carry signal and the tokenizer handles them.
- **Unknown words.** There is no `<unk>` problem: any string is split into subword pieces, and in the worst case into single characters.

Nothing is fitted on the data during preprocessing: the tokenizer is the pretrained one, and the normalisation rules are fixed. So no vocabulary or statistic leaks from validation or test.

### 4.3 Training settings (`configs/train_config.json`)

| setting | value |
|---|---|
| base model | `FacebookAI/xlm-roberta-base` |
| objective | cross-entropy over 2 classes, with the standard sequence-classification head on the `<s>` token |
| optimizer | AdamW: lr 2e-5, weight decay 0.01 (none on bias and LayerNorm), gradient clipping 1.0 |
| schedule | linear warm-up over 10 % of the steps, then linear decay |
| batch | 32 for training, 64 for evaluation; maximum length 128 subwords |
| epochs | up to 4, early stopping with patience 2 on validation F1 |
| precision | fp16 mixed precision on GPU |
| seed | 42 (Python, NumPy, torch, data order) |
| selection | after every epoch: compute p(profanity) on validation, pick the threshold in [0.05, 0.95] that maximises **F1 of class 1**, keep the epoch with the best validation F1 |

The class imbalance (about 8 % positives in train) is handled by **choosing the threshold** on validation, not by re-weighting the loss.

Exact package versions of the run are saved in `outputs/results/environment.json`. The pinned versions are in `requirements.txt` (local) and `requirements-train.txt` (on top of Kaggle/Colab).

---

## 5. Reproduce

**Option A: Kaggle or Colab (recommended, GPU).**
1. Open `notebooks/train_kaggle_colab.ipynb` in a new Kaggle notebook (Accelerator GPU, Internet on) or in Colab (GPU runtime).
2. Set `REPO_URL`.
3. Run all cells.

The notebook installs the pinned packages, (optionally) rebuilds the data and checks the checksums, trains, evaluates on test once and on the challenge set, lets you type your own comments, and saves or uploads the model.

**Option B: local.**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/get_kazsandra.sh data/raw                                     # optional: rebuild the data
python -m src.prepare_data --kazsandra_dir data/raw --out_dir data/processed --seed 42
python -m src.train    --config configs/train_config.json --data_dir data/processed --out_dir outputs
python -m src.evaluate --model_dir outputs/model --data_dir data/processed --out_dir outputs/results
python -m src.evaluate --model_dir outputs/model --data_dir data/challenge --split challenge_set --out_dir outputs/results
python -m src.predict  --model_dir outputs/model --text "..."
```

`src/evaluate.py` refuses to overwrite existing test results unless `--force` is given. This is a guard against re-using the test set for tuning.

---

## 6. Results

> **To be filled in from `outputs/results/` after the Kaggle run** (test metrics, confusion matrix, training curve, challenge set, error analysis).

### 6.1 Test set: one evaluation, threshold from validation

| metric (class 1 = profanity) | value |
|---|---|
| test examples | 2,028 (148 profane / 1,880 clean) |
| precision | … |
| recall | … |
| F1 | … |
| PR-AUC (area under the precision–recall curve) | … |
| threshold (from validation) | … |

Confusion matrix (`outputs/results/test_confusion_matrix.png`):

| | predicted clean | predicted profanity |
|---|---|---|
| **true clean** | TN = … | FP = … |
| **true profanity** | FN = … | TP = … |

### 6.2 What the errors mean for review moderation

- **False positive (FP):** a clean review is flagged as profane. If flags auto-hide reviews, an honest, often *negative* review disappears. That hurts customers who need that information and looks like censorship of complaints. It also wastes moderator time.
- **False negative (FN):** a review with Kazakh profanity is published. Other customers, including children, see obscene language next to the product, and the platform's quality suffers.
- **Which matters more** depends on how the model is used:
  - If its flags go to a **human moderation queue**, FNs matter more: a FP costs a few seconds of a moderator's time, a FN stays published. Use a lower threshold and prefer recall.
  - If the model **removes reviews automatically**, FPs matter more. Use a higher threshold and prefer precision.

  The saved threshold maximises F1, a balance between the two; `--threshold` in `src/predict.py` changes it without retraining.

### 6.3 Error analysis

> At least three concrete test mistakes, their likely causes and a practical fix — filled in after the run, from `test_false_positives.csv` / `test_false_negatives.csv`.

---

## 7. Limitations

- **Labels come partly from rules.** Strong-rule matches were spot-checked, not all read. Unlisted slang is labelled 0 by default, so some model "false positives" may actually be label errors.
- **Domain shift.** The real profane examples are app reviews. Product-review profanity is measured only on the 48-comment synthetic challenge set, which is too small for precise numbers.
- **Scope decision.** Comments with Russian-only obscenities are labelled 0. A real moderation system would probably want them flagged as well. That needs a second label or a relabelling of the `ru_profanity` rows.
- **Hostility is not profanity.** Rude but non-obscene insults (мал, ақымақ) are 0 by design, so this model is not a general toxicity detector.
- **Names in review text.** First names mentioned inside some comments (for example, one gamer addressing another) were not removed. Only contact details were scrubbed.
- **Single run.** One model, one seed, no repeated-seed variance estimate, as allowed by the task.

---

## 8. AI assistance and my contribution

**AI assistance.** I used Claude (Anthropic's AI assistant) to:
- draft the data-preparation, training, evaluation and inference code;
- design the lexicon and the labelling rules;
- pre-sort the ambiguous lexicon matches;
- write the challenge set and this README.

**My own work and checks.** `<to complete: what you reviewed, changed, ran (the Kaggle training run), verified by hand, and which decisions are yours>`.

I understand and can explain all submitted code.

## 9. Attribution

Data: R. Yeshpanov and H. A. Varol, *KazSAnDRA: Kazakh Sentiment Analysis Dataset of Reviews and Attitudes*, LREC-COLING 2024, CC BY 4.0. The processed splits in `data/processed/` are derived from KazSAnDRA and are shared under the same licence.
Base model: XLM-RoBERTa (Conneau et al., 2020), MIT licence.
