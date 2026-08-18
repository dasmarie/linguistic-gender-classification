# Linguistic Gender Classification


This repository contains code and datasets for linguistic gender classification of English nouns referring to people.

The project builds and compares several systems that classify a target noun in a text passage as unambiguous_female, unambiguous_male, unambiguous_all for mixed-gender groups, or ambiguous if its gender is not specified. Rather than detecting the real-world gender of an entity, the systems detect the linguistic gender that can be deduced from the passage. Some are limited to explicit gender cues (pronouns, gendered titles, and lexical descriptions), while others have access to the full context, allowing a comparison of whether restricting a system to explicit cues reduces misclassifications caused by gender stereotypes. Detecting linguistic gender this way can support downstream tasks such as gender-fair machine translation into languages with grammatical gender, or gender bias detection.


## Methods

- **Rule-Based Classifier**: works with coreference resolution, word definitions, lexicons, and pre-defined dependency paths, which are combined through a set of rules.
- **Feature-Based Classifier (Logistic Regression)**: works with features derived from the mentioned components of the rule-based classifier
- **Rule-Based Classifier + LLM**: combines the rule-based classifier with an LLM (Qwen 2.5-3B) to include cases that cannot be covered by a set of rules. First, the rule-based system classifies all relevant nouns. Then, for all instances that were classified as *ambiguous*, Qwen 2.5-3B is provided with gender cues from the surrounding text passages, which it is then asked to assign to the correct noun.
- **DistilBERT (Fine-tuned for classification)**: DistilBERT (Sanh et al., 2020) is fine-tuned for classification with access to the full context of each example.
- **Qwen 2.5-72B and Qwen 2.5-3B**: work with a few-shot prompt based on a prompt used by Pranav et. al. (2025) for linguistic gender classification. It is edited to contain examples of all major ways in which gender can be indicated.


## Installation

To run the code, install the dependencies from **requirements.txt**:

```bash
pip install -r requirements.txt
```

## Usage

With all dependencies installed, the code should run, provided the repository structure is kept intact. All data and modules are already arranged as the scripts expect. Paths can be adjusted at the beginning of each notebook. 

To apply the different classifiers, use the code below:

(The example sentence used to demonstrate the usage is taken from Anne of Green Gables by Lucy Maud Montgomery)

### Rule-Based Classifier

The rule-based classifier can either label the full text passage:

**Code:**

```python
from modules.explicit_cues import detect_gender

detect_gender("Well, my mother was a teacher in the high school, too")
```

**Output:**

```
([{'mentions': ['my mother', 'a teacher in the High School'],
   'entity_gender': 'unambiguous_female'}],
 'Well, my [unambiguous_female] mother was a [unambiguous_female] teacher in the High School, too',
 ['_', '_', '_', 'unambiguous_female', '_', '_', 'unambiguous_female', '_', '_', '_', '_', '_', '_'])

```

or just label one specified noun:

**Code:**

```python
index = 6   #index for 'teacher'

mentions, text, labels = detect_gender(['Well', ',', 'my', 'mother', 'was', 'a', 'teacher', 'in', 'the', 'High', 'School', 'too'], index) #the index parameter makes sure the noun is classified even if the system does not recognise it as a noun referring to a person

print(labels[index])
```

**Output**

```
unambiguous_female
```


### Feature-Based Classifier


**Code:**

```python

from modules.explicit_cues import feature_gender
import pickle

with open('../../models/logreg_model.pkl', 'rb') as f:
    saved = pickle.load(f)

clf = saved['clf']
vectorizer = saved['vectorizer']

index = 6

print(feature_gender('Well, my mother was a teacher in the high school, too', index, clf, vectorizer))

```

**Output:**

```
unambiguous_female
```

### DistilBERT Classifier

To use this classifier, it first has to be trained using **bert.ipynb** which is located in code/the_systems .

(or you can download it here: https://drive.google.com/drive/folders/13klSmqGx7iAdwtjcGu-Z-mXVowSXVLkR?usp=sharing)

After downloading or training, place the folder including the fine-tuned model and tokenizer in the **models** folder or adjust the path.

**Code:**

```python

from modules.explicit_cues import bert_gender
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_path = '../../models/gender_bert'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

index = 6

bert_gender(['Well', ',', 'my', 'mother', 'was', 'a', 'teacher', 'in', 'the', 'High', 'School', 'too'], index, model, tokenizer)

```

**Output**

```
unambiguous_female
```


## Repository Structure


**code** contains all scripts and Jupyter notebook created to train and evaluate the classifiers. It consists of two subfolders:

- **evaluation**: evaluation-and-analysis.ipynb contains code that evaluates the performance of all systems and conducts and error analysis; dataset-prep.ipynb creates the dataset
- **the_systems**: this folder contains separate notebooks to train each classifier and one notebook (try-classifiers.ipynb) with which the rule-based, feature-based, and BERT-based classifiers can be used on any text passage or sentence.

- Each of the subfolders contains a folder named **modules** which contains explicit_cues.py to build and run the classifiers and evaluate.py for anything to do with evaluating them. They also contain a utils-folder, which is used for lexical gender detection and created/written by Bartl et al. (2022) (not me).


**data** contains all data used to train and evaluate the classifiers.

It contains all separate preprocessed datasets, the original datasets that were needed for the dataset creation, and the full combined datasets. data_that_helps contains data/key words/lexicons used for the dataset creation while winomt_ambiguous_only.tsv contains a test set for nouns with ambiguous linguistic gender in sentences with gendered pronouns.

The original datasets used to create the training/evaluation dataset used here, consists of datasets from Pranav et al. (2025), Piergentili et al. (2024), and Jourdan et al. (2025). I only uploaded them here for an easier grading process (I did not create them; will take them down after the grading is done).


**models** contains the feature-based classifier. It is also where the BERT classifier should be saved after training or downloading it.


**results** contains all results.



### Dataset References

A. Pranav, J. Hackenbuchner, G. Attanasio, M. Lardelli, and A. Lauscher. Glitter: A
multi-sentence, multi-reference benchmark for gender-fair German machine translation. In
C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, editors, Findings of the
Association for Computational Linguistics: EMNLP 2025, pages 18450–18477, Suzhou,
China, Nov. 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi:
10.18653/v1/2025.findings-emnlp.1002. URL
https://aclanthology.org/2025.findings-emnlp.1002/


A. Piergentili, B. Savoldi, D. Fucci, M. Negri, and L. Bentivogli. Hi guys or hi folks?
benchmarking gender-neutral machine translation with the GeNTE corpus. In H. Bouamor,
J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing, pages 14124–14140, Singapore, Dec. 2023. Association for
Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.873. URL
https://aclanthology.org/2023.emnlp-main.873/


F. Jourdan, Y. Chevalier, and C. Favre. Fairtranslate: an english-french dataset for gender
bias evaluation in machine translation by overcoming gender binarity. In Proceedings of the
2025 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’25, page
150–166, New York, NY, USA, 2025. Association for Computing Machinery. ISBN
9798400714825. doi: 10.1145/3715275.3732013. URL
https://doi.org/10.1145/3715275.3732013.



### utils (Reference)

M. Bartl and S. Leavy. Inferring gender: A scalable methodology for gender detection with
online lexical databases. In B. R. Chakravarthi, B. Bharathi, J. P. McCrae, M. Zarrouk,
K. Bali, and P. Buitelaar, editors, Proceedings of the Second Workshop on Language
Technology for Equality, Diversity and Inclusion, pages 47–58, Dublin, Ireland, May 2022.
Association for Computational Linguistics. doi: 10.18653/v1/2022.ltedi-1.7. URL
https://aclanthology.org/2022.ltedi-1.7/


**code** and **data** also contain a folder called modules with functions for evaluation (evaluation.py) and creation (explicit_cues.py) of the classifiers.
