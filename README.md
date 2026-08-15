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

(or you can download it here: )

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


**data** contains all data used to train and evaluate the classifiers.

It contains all separate preprocessed datasets, the original datasets that were needed for the dataset creation, and the full combined datasets. data_that_helps contains data/key words/lexicons used for the dataset creation while winomt_ambiguous_only.tsv contains a test set for nouns with ambiguous linguistic gender in sentences with gendered pronouns.


**models** contains the feature-based classifier. It is also where the BERT classifier should be saved after training or downloading it.


**results** contains all results.



**code** and **data** also contain a folder called modules with functions for evaluation (evaluation.py) and creation (explicit_cues.py) of the classifiers.

