#from maverick import Maverick
import spacy
from spacy.tokens import Doc
import sys
import csv
from modules.utils.dict_utils import check_dictionary
from nltk.corpus import wordnet as wn
from modules.explicit_cues import fem_lex, masc_lex, neutral_lex, detect_gender, parse_sent
import gender_guesser.detector as gender
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from modules.explicit_cues import is_negated, is_person, ner_person, is_plural, conj_chain, longest_cluster_for_token, syntactic_indexes, gender_person_token, modifier_gender, coref_clusters, build_token_to_cluster, pronoun_gender, longest_cluster_for_token, collect_relevant_tokens, lexical_ref_gender, plural_conj_cluster, fallback_cues, closest_gendered_dep_path, find_gendered_words, get_word_gender, dependency_path, dependency_distance, find_lowest_common_ancestor, get_ancestor_chain, modifier_gender, lexical_gender
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

nlp = spacy.load("en_core_web_md")
d = gender.Detector()

#coref_model = Maverick(
 # hf_name_or_path = "sapienzanlp/maverick-mes-preco",
  #device = "cpu"
#)

#coref_model.model = coref_model.model.float() 


fem_set = {'coiffeuse', 'landlady', 'chambermaid', 'comedienne', 'odalisque', 'squire', 'countrywoman', 'nymphet', 'instructress', 'hag', 'spokeswoman', 'flibbertigibbet', 'nanny', 'quadripara', 'wac', 'daughter', 'starlet', 'scold', 'authoress', 'executrix', 'squaw', 'ingenue', 'oarswoman', 'parlormaid', 'stateswoman', 'jilt', 'forewoman', 'sylph', 'englishwoman', 'beard', 'great-aunt', 'heiress', "ma'am", 'schoolgirl', 'cinderella', 'irishwoman', 'protegee', 'debutante', 'foster-nurse', 'stepmother', 'brownie', 'spinster', 'traitress', 'patroness', 'marchioness', 'groupie', 'cornishwoman', 'madwoman', 'beggarwoman', 'mayoress', 'mediatrix', 'signorina', 'prophetess', 'memsahib', 'bondwoman', 'concubine', 'mammy', 'mrs.', 'tomboy', 'co-ed', 'begum', 'assemblywoman', 'usherette', 'deaconess', 'ladylove', 'gamine', 'gitana', 'mother-in-law', 'countess', 'madame', 'ms.', 'primigravida', 'sculptress', 'yenta', 'priestess', 'curandera', 'selectwoman', 'vestrywoman', 'women', 'demimondaine', 'foster-mother', 'spitfire', 'viscountess', 'doyenne', 'mother', 'homegirl', 'ball-buster', 'primipara', 'barmaid', 'bride', 'niece', 'thrush', 'donna', 'godmother', 'hostess', 'coloratura', "light-o'-love", 'manageress', 'czarina', 'damsel', 'postmistress', 'councilwoman', 'cowgirl', 'procuress', 'handmaid', 'secundigravida', 'proprietress', 'foundress', 'girlfriend', 'signora', 'colleen', 'bag', 'milady', 'broad', 'horsewoman', 'baggage', 'beguine', 'enchantress', 'duchess', 'lady', 'female', 'seductress', 'chachka', 'smasher', 'stepdaughter', 'wife', 'sheika', 'mamma', 'salesgirl', 'she-devil', 'geisha', 'granddaughter', 'granny', 'sister-in-law', 'heroine', 'gal', 'jewess', 'dragon', 'bacchante', 'canary', 'duenna', 'daygirl', 'negotiatress', 'frontierswoman', 'circe', 'nan', 'jezebel', 'foremother', 'benefactress', 'coquette', 'lass', 'niqaabi', 'poetess', 'shrew', 'virago', 'dame', 'undoer', 'housewife', 'maenad', 'stewardess', 'villainess', 'beldam', 'foster-sister', 'harridan', 'nullipara', 'fiancee', 'girl', 'charwoman', 'soprano', 'butch', 'princess', 'quintipara', 'housemother', 'bluestocking', 'shiksa', 'newswoman', 'maid', 'nun', 'soubrette', 'bridesmaid', 'shepherdess', 'frump', 'divorcee', 'madam', 'maharani', 'wardress', 'nymph', 'poseuse', 'committeewoman', 'vestal', 'belle', 'muslimah', 'mill-girl', 'skivvy', 'washwoman', 'baroness', 'matron', 'aunt', 'farmerette', 'gravida', 'adventuress', 'dairymaid', 'widow', 'waitress', 'uxor', 'adulteress', 'kinswoman', 'slattern', 'sorceress', 'conductress', 'sibyl', 'archduchess', 'chatelaine', 'woman', 'supermom', 'ladies', 'eyeful', 'abbess', 'millionairess', 'embroideress', 'outdoorswoman', 'scotswoman', 'empress', 'foster-daughter', 'mistress', 'peri', 'wanton', 'nymphomaniac', 'ex-wife', 'contralto', 'amazon', 'great-niece', 'mannequin', 'murderess', 'businesswoman', 'idolatress', 'ma', 'queen', 'huntress', 'songstress', 'actress', 'ayah', 'midwife', 'polyandrist', 'schoolmarm', 'ambassadress', 'rani', 'goddaughter', 'rosebud', 'aviatrix', 'daughter-in-law', 'flapper', 'testatrix', 'matriarch', 'confidante', 'mestiza', 'suffragette', 'puerpera', 'b-girl', 'taskmistress', 'tertigravida', 'ballerina', 'governess', 'inamorata', 'ancestress', 'parisienne', 'headmistress', 'bobbysoxer', 'sister'}

masc_set = {'novillero', 'ponce', 'schoolboy', 'vaquero', 'casanova', 'indiana', 'sheepman', 'inamorato', 'basileus', 'priest', 'signore', 'tom', 'milord', 'pachuco', 'flamen', 'headsman', 'fireman', 'franklin', 'everyman', 'peer', 'brother', 'lawman', 'cavalryman', 'cleric', 'committeeman', 'boyfriend', 'klansman', 'archbishop', 'longbowman', 'quarryman', 'suitor', 'masseur', 'helmsman', 'wittol', 'hobbledehoy', 'man', 'ganger', 'milkman', 'bellboy', 'romeo', 'galoot', 'weatherman', 'swagman', 'liveryman', 'fathead', 'butler', 'king', 'sod', 'emir', 'tenor', 'caveman', 'iceman', 'bushman', 'kinsman', 'proconsul', 'cuckold', 'statesman', 'ex-boyfriend', 'lineman', 'gent', 'ordinary', 'patriarch', 'ethnarch', 'capo', 'infantryman', 'sheik', 'assemblyman', 'frontiersman', 'son', 'womanizer', 'guardsman', 'beggarman', 'orangeman', 'wolf', 'minuteman', 'countertenor', 'mr.', 'burgrave', 'goliard', 'militiaman', 'gillie', 'aircrewman', 'schoolman', 'charon', 'town', 'coiffeur', 'sultan', 'placeman', 'husband', 'sannup', 'geezer', 'tarzan', 'loon', 'gaucho', 'foster-brother', 'paperboy', 'samurai', 'selectman', 'forefather', 'ape-man', 'repairman', 'housefather', 'dalesman', 'don', 'archdeacon', 'mailman', 'raja', 'mafioso', 'cyril', 'polygynist', 'poultryman', 'sandboy', 'imam', 'showman', 'margrave', 'yeoman', 'plainsman', 'businessman', 'dandy', 'sidesman', 'excavator', 'choirboy', 'foster-son', 'plainclothesman', 'aircraftsman', 'plowboy', 'ejaculator', 'warlord', 'outdoorsman', 'vestryman', 'mikado', 'beadsman', 'chapman', 'uxoricide', 'stepbrother', 'councilman', 'irishman', 'sadhu', 'englishman', 'trainman', 'townes', 'grandfather', 'busboy', 'count', 'footman', 'great-nephew', 'liege', 'adonis', 'ottoman', 'oklahoman', 'oilman', 'monsignor', 'viscount', 'crewman', 'sir', 'satyr', 'bey', 'lord', 'jacob', 'roundsman', 'son-in-law', 'bass', 'godson', 'townsman', 'gentlemen', 'yachtsman', 'clansman', 'eparch', 'gentleman', 'brakeman', 'pitchman', 'sire', 'exarch', 'armiger', 'cub', 'centurion', 'father-figure', 'coachman', 'ferryman', 'nuncio', 'vicar-general', 'foster-father', 'duke', 'strongman', 'man-at-arms', 'brunet', 'workman', 'princeling', 'abbot', 'artilleryman', 'rifleman', 'castrato', 'polycarp', 'timberman', 'freedman', 'baritone', 'benedick', 'guy', 'pope', 'nephew', 'ironside', 'rake', 'stepfather', 'salesman', 'posseman', 'lothario', 'congressman', 'bagman', 'welshman', 'horseman', 'plowman', 'tribesman', 'uriah', 'dayboy', 'layman', 'groomsman', 'libertine', 'subdeacon', 'male', 'cameraman', 'trainbandsman', 'spokesman', 'kennan', 'babu', 'policeman', 'shah', 'deliveryman', 'earl', 'wireman', 'maharaja', 'cattleman', 'gunman', 'serviceman', 'widower', 'buddy', 'wingman', 'draftsman', 'pharaoh', 'lumberman', 'father-in-law', 'blade', 'vizier', 'seedsman', 'lobsterman', 'tallyman', 'gasman', 'augustinian', 'roadman', 'knight-errant', 'handyman', 'stableman', 'freeman', 'trappist', 'marquess', 'shaver', 'lackey', 'soundman', 'fauntleroy', 'emperor', 'signor', 'uncle', 'vicar', 'praetor', 'men', 'danseur', 'doge', 'lighterman', 'father', 'mullah', 'manservant', 'alderman', 'fisherman', 'yardman', 'squire', 'palatine', 'catamite', 'fugleman', 'dad', 'gagman', 'stepson', 'kaiser', 'thane', 'doorkeeper', 'pater', 'signalman', 'stockman', 'raftsman', 'friar', 'grandson', 'mollycoddle', 'baron', 'boy', 'pointsman', 'marquis', 'washerman', 'groom', 'sodomite', 'czar', 't-man', 'fellow', 'nabob', 'ex-husband', 'deacon', 'groundsman', 'foreman', 'woodsman', 'esquire', 'clergyman', 'coalman', 'boatman', 'macho', 'curandero', 'brother-in-law', 'linesman', 'cowboy', 'shaheed', 'bondsman', 'doughboy', 'elector', 'jimdandy', 'coastguardsman', 'dog', 'great-uncle', 'muscleman', 'sandwichman', 'limey', 'pederast', 'stiff', 'herr', 'godfather', 'macaroni', 'malik', 'baggageman', 'misogynist', 'page', 'bacchant', 'marksman', 'frenchman', 'letterman', 'dairyman', 'bondman', 'dean', 'popper', 'jawan', 'cornishman', 'homeboy', 'dragoman', 'pendragon', 'senhor', 'headman', 'mestizo', 'watchman', 'dauphin', 'gitano', 'lowerclassman', 'countryman', 'stud', 'townee', 'pilate', 'midshipman', 'mover', 'oarsman', 'g-man', 'roman', 'jute', 'seducer', 'cavalier', 'bullyboy', 'hangman', 'bull', 'brahman', 'archpriest', 'batman', 'wencher', 'monsieur', 'simeon', 'codger', 'apostle', 'linkboy', 'sirrah', 'bachelor', 'craftsman', 'bandsman', 'grandee', 'potboy', 'monk'}



def load_fairtranslate(filepath):
    """
    loads the fairtranslate tsv file (with header).
    columns: seed, sentence, label, stereotype
    """
    examples = []
    with open(filepath) as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        header = next(reader)
        for row in reader:
            if len(row) != 4:
                continue
            seed, sentence, label, stereotype = row
            examples.append({
                'seed': seed,
                'sentence': sentence,
                'label': label,
                'stereotype': stereotype
            })
    return examples

def detect_gender_middle(row):
    seed = row[0]
    s1 = row[1]
    s2 = row[2]
    s3 = row[3]
    full_text = s1 + ' ' + s2 + ' ' + s3
    s2_start_char = len(s1) + 1
    s2_end_char = s2_start_char + len(s2)

    seed_words = seed.strip().split()
    target_word = seed_words[-1]

    spacy_doc = parse_sent(full_text)

    target_index = None
    for token in spacy_doc:
        if token.idx >= s2_start_char and token.idx < s2_end_char:
            if token.lemma_.lower() == target_word.lower() or token.text.lower() == target_word.lower():
                target_index = token.i
                break

    results, labeled_text_output, token_labels = detect_gender(full_text, target_index=target_index)

    if target_index is not None:
        return token_labels[target_index]
    return "nothing"


def write_results(outfile_path, test_rows, predictions):
    with open(outfile_path, 'w') as outfile:
        for row, result in zip(test_rows, predictions):
            outfile.write('\t'.join(row) + '\t' + result + '\n')


def evaluate_test_set(filepath, dataset_type, vectorizer, clf):
    """
    loads a test set, extracts features, predicts with the trained
    classifier, and prints a classification report and confusion matrix.
    """
    X_dicts, y_true = load_dataset(filepath, dataset_type, coref_model)

    X_test = vectorizer.transform(X_dicts)
    y_pred = clf.predict(X_test)

    print(filepath)
    print(classification_report(y_true, y_pred))

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation='vertical')
    plt.title(filepath)
    plt.tight_layout()
    plt.show()


def load_glitter(filepath):
    """
    loads the glitter tsv file (no header).
    columns: seed, prev_context, sentence, next_context, label
    """
    examples = []
    with open(filepath) as f:
        reader = csv.reader(f, delimiter='\t', quoting = csv.QUOTE_NONE)
        for row in reader:
            if len(row) != 5:
                continue
            seed, prev_context, sentence, next_context, label = row
            examples.append({
                'seed': seed,
                'prev_context': prev_context,
                'sentence': sentence,
                'next_context': next_context,
                'label': label
            })
    return examples


def load_gente(filepath):
    """
    loads the gente tsv file (with header).
    columns: seed, sentence, label
    """
    examples = []
    with open(filepath) as f:
        reader = csv.reader(f, delimiter='\t', quoting = csv.QUOTE_NONE)
        header = next(reader)
        for row in reader:
            if len(row) != 3:
                continue
            seed, sentence, label = row
            examples.append({
                'seed': seed,
                'sentence': sentence,
                'label': label
            })
    return examples


def load_lexical(filepath):
    """
    loads the lexical tsv file (with header).
    columns: seed, sentence, label, stereotype
    """
    examples = []
    with open(filepath) as f:
        reader = csv.reader(f, delimiter='\t', quoting = csv.QUOTE_NONE)
        header = next(reader)
        for row in reader:
            if len(row) != 4:
                continue
            seed, sentence, label, stereotype = row
            examples.append({
                'seed': seed,
                'sentence': sentence,
                'label': label,
                'stereotype': stereotype
            })
    return examples



def build_text_and_span(example, dataset_type):
    """
    builds the full text to be parsed, and the character range
    [sentence_start_char, sentence_end_char) of the part of the text
    in which the seed should be looked up.

    for glitter, the previous and next context sentences are included
    in the full text (so that coreference resolution can use them),
    but the seed is only searched for within the main 'sentence' part.

    for gente/lexical, the text is just the sentence itself.
    """
    if dataset_type == 'glitter':
        prev_context = example['prev_context']
        sentence = example['sentence']
        next_context = example['next_context']

        if prev_context:
            prefix = prev_context + ' '
        else:
            prefix = ''

        sentence_start_char = len(prefix)
        full_text = prefix + sentence
        sentence_end_char = len(full_text)

        if next_context:
            full_text = full_text + ' ' + next_context

        return full_text, sentence_start_char, sentence_end_char

    else:
        sentence = example['sentence']
        return sentence, 0, len(sentence)


def find_seed_token_index(spacy_doc, seed, sentence_start_char, sentence_end_char):
    """
    finds the index of the head token of the seed phrase within
    spacy_doc, restricted to the character range
    [sentence_start_char, sentence_end_char).

    the seed is matched case-insensitively as a substring of the text
    in that range. if the matched character span aligns with token
    boundaries, the root (head) of that span is returned - this gives
    the head noun for multi-word seeds (e.g. 'team leaders' -> 'leaders').

    if alignment fails, falls back to matching the last word of the
    seed phrase as a single token.

    returns the token index, or None if no match is found.
    """
    full_text = spacy_doc.text
    section = full_text[sentence_start_char:sentence_end_char]

    seed_lower = seed.lower()
    pos_in_section = section.lower().find(seed_lower)

    if pos_in_section == -1:
        return None

    char_start = sentence_start_char + pos_in_section
    char_end = char_start + len(seed)

    span = spacy_doc.char_span(char_start, char_end, alignment_mode='expand')

    if span is not None and len(span) > 0:
        return span.root.i

    # fallback: match the last word of the seed phrase as a single token
    seed_words = seed_lower.split()
    last_word = seed_words[-1]

    for token in spacy_doc:
        if token.idx < sentence_start_char or token.idx >= sentence_end_char:
            continue
        if token.text.lower() == last_word:
            return token.i

    return None


def prepare_examples(filepath, dataset_type):
    """
    loads a dataset file and prepares each example for feature extraction.

    param filepath: path to the tsv file
    param dataset_type: one of 'glitter', 'gente', 'lexical'

    returns a list of dicts, each with:
        'text': the full text to be parsed
        'spacy_doc': the parsed spacy doc
        'seed_token_index': index of the head token of the seed noun,
            or None if it could not be located
        'seed': the seed string from the dataset
        'label': the gold label
        'stereotype': only present for lexical examples
    """
    if dataset_type == 'glitter':
        raw_examples = load_glitter(filepath)
    elif dataset_type == 'gente':
        raw_examples = load_gente(filepath)
    elif dataset_type == 'lexical':
        raw_examples = load_lexical(filepath)
    elif dataset_type == 'fairtranslate':
        raw_examples = load_fairtranslate(filepath)
    else:
        raise ValueError('unknown dataset_type: ' + dataset_type)

    prepared = []

    for example in raw_examples:
        full_text, sentence_start_char, sentence_end_char = build_text_and_span(example, dataset_type)

        spacy_doc = nlp(full_text)

        seed_token_index = find_seed_token_index(
            spacy_doc, example['seed'], sentence_start_char, sentence_end_char
        )

        entry = {
            'text': full_text,
            'spacy_doc': spacy_doc,
            'seed_token_index': seed_token_index,
            'seed': example['seed'],
            'label': example['label']
        }

        if 'stereotype' in example:
            entry['stereotype'] = example['stereotype']

        prepared.append(entry)

    return prepared



def evaluate_test_set_rule_based(filepath, dataset_type):
    """
    loads a test set, runs the rule-based system, and prints a
    classification report and confusion matrix.

    for glitter, rows are read directly from the tsv and passed to
    detect_gender_middle, since that function expects the raw
    seed/s1/s2/s3 structure.

    for gente/lexical, examples are loaded via prepare_examples and
    run through detect_gender directly.

    in both cases, examples where the seed token could not be located
    are skipped rather than counted.
    """
    y_true = []
    y_pred = []
    skipped = 0

    if dataset_type == 'glitter':
        with open(filepath) as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) != 5:
                    continue
                seed, s1, s2, s3, label = row

                pred_label = detect_gender_middle([seed, s1, s2, s3])

                if pred_label == '_':
                    skipped = skipped + 1
                    continue

                y_true.append(label)
                y_pred.append(pred_label)

    else:
        examples = prepare_examples(filepath, dataset_type)

        for example in examples:
            token_index = example['seed_token_index']

            if token_index is None:
                skipped = skipped + 1
                continue

            text = example['text']
            results, labeled_text_output, token_labels = detect_gender(text)
            pred_label = token_labels[token_index]

            y_true.append(example['label'])
            y_pred.append(pred_label)

    print(filepath)
    #print('skipped (seed not located):', skipped)
    print(classification_report(y_true, y_pred))

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation='vertical')
    plt.title(filepath)
    plt.tight_layout()
    plt.show()

def llm_filter(model, tokenizer, row, label, dataset = 'glitter'):
    seed = row[0]
    if dataset == 'glitter':
        s1 = row[1]
        s2 = row[2]
        s3 = row[3]
        full_text = s1 + ' ' + s2 + ' ' + s3
    else:
        full_text = row[1]

    genders_llm = []
    prompts_and_responses = []
    male_count = 0
    female_count = 0

    if label != 'ambiguous':
        geschlecht = label
        #labels_after_llm.append(label)
        return geschlecht
    else:
        gender, genders, include, both = fallback_cues(nlp(full_text))

        if genders != []:
            for tup in genders:
                prompt = f"""You are given the following sentence: "{str(full_text)}" In this specific text passage, does the word '{str(tup[1])}' influence the gender of the word '{str(seed)}'? Reply with one word"""
                #print(prompt)

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    temperature = 0.2,
                    do_sample = False
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                prompts_and_responses.append([prompt, response])
                
                #print()
                #print(seed)
                #print(label, gold)
                #print(response)
                #print()

                
                if 'Yes' in response:
                    genders_llm.append(tup[0])
                else:
                    genders_llm.append('ambiguous')

        if genders_llm:
            for g in genders_llm:
                if g == 'female':
                    female_count += 1
                elif g == 'male':
                    male_count += 1

            if female_count > 0 and male_count == 0:
                geschlecht = 'unambiguous_female'
            elif female_count == 0 and male_count > 0:
                geschlecht = 'unambiguous_male'
            elif female_count > 0 and male_count > 0:
                geschlecht = 'unambiguous_all'
            else:
                geschlecht = 'ambiguous'
        else:
            geschlecht = 'ambiguous'

        return geschlecht, prompts_and_responses

        

def run_rule_based(filepath, dataset_type, outfile_path=None):
    """
    loads a test set and runs the rule-based system, returning the
    gold labels, predictions, a skipped count, and the rows used
    (for downstream llm fallback).

    for glitter, rows are read directly from the tsv ([seed, s1, s2, s3])
    and passed to detect_gender_middle.

    for gente/lexical/fairtranslate, examples are loaded via
    prepare_examples and run through detect_gender directly. rows are
    returned as [seed, text] for these datasets.

    in both cases, examples where the seed token could not be located
    are skipped rather than counted.
    """
    gold = []
    predictions = []
    rows = []
    seeds = []
    texts = []
    skipped = 0

    if dataset_type == 'glitter':
        with open(filepath) as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) != 5:
                    continue
                seed, s1, s2, s3, label = row
                pred_label = detect_gender_middle([seed, s1, s2, s3])
                if pred_label == '_':
                    skipped = skipped + 1
                    continue
                gold.append(label)
                predictions.append(pred_label)
                seeds.append(seed)
                texts.append(s1 + ' ' + s2 + ' ' + s3)
                rows.append([seed, s1, s2, s3])

    else:
        examples = prepare_examples(filepath, dataset_type)
        for example in examples:
            token_index = example['seed_token_index']
            if token_index is None:
                skipped = skipped + 1
                continue
            text = example['text']
            results, labeled_text_output, token_labels = detect_gender(text)
            pred_label = token_labels[token_index]
            gold.append(example['label'])
            predictions.append(pred_label)
            seeds.append(example['seed'])
            texts.append(text)
            rows.append([example['seed'], text])

    if outfile_path is not None:
        with open(outfile_path, 'w', encoding='utf8') as outfile:
            outfile.write('seed\ttext\tgold\tprediction\n')
            for seed, text, gold_label, pred_label in zip(seeds, texts, gold, predictions):
                outfile.write(seed + '\t' + text + '\t' + gold_label + '\t' + pred_label + '\n')

    return gold, predictions, skipped, rows


def evaluate_rule_based(filepath, y_true, y_pred, skipped=None):
    """
    prints a classification report and plots a confusion matrix
    for already-computed predictions and gold labels.
    """
    print(filepath)
    if skipped is not None:
        print('skipped (seed not located):', skipped)
    print(classification_report(y_true, y_pred))

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation='vertical')
    plt.title(filepath)
    plt.tight_layout()
    plt.show()

def fallback_cues(spacy_text):
    genders =  []
    gender = None
    include = False
    both = False
    female = ['female', 'woman', 'women']
    male = ['male', 'man', 'men']
    mixed = ['mixed-gender', 'genders', 'mixed']
    fem = 0
    masc = 0
    mix = 0
    
    for token in spacy_text:
        lexical_gender = 'neutral'
        lemma = token.lemma_
        if lemma == 'include' or lemma == 'consist' or lemma == 'contain':
            include = True
        if lemma in fem_lex:
            lexical_gender = 'fem'
            #print(lemma, 'fem lex')
        elif lemma in masc_lex:
            lexical_gender = 'masc'
            #print(lemma, 'masc lex')
        #elif token.pos_ == 'NOUN' and is_person(lemma) == True:
            #print('looking at lexical gender', token)
            #lexical_gender = check_dictionary(word=lemma, 
             #            dict_abbrev='wordnet', 
         #                heuristics=True, 
          #               seed_pairs=5, no_words=35, no_defs=10)
        if token in female or lexical_gender == 'fem':
            genders.append(('female', token))
            #print('female', token)
        if token in male or lexical_gender == 'masc':
            genders.append(('male', token))
        if token in mixed:
            genders.append(('mixed', token))



    if len(genders) == 1:
        gender = genders[0][0]
    elif genders == []:
        gender = 'ambiguous'
    else:
        for g in genders:
            if g[0] == 'female':
                fem += 1
            elif g[0] == 'male':
                masc += 1
            elif g[0] == 'mixed':
                mix += 1

    if masc > 0 and fem == 0 and mix == 0:
        gender = 'male'
    elif masc == 0 and fem > 0 and mix == 0:
        gender = 'female'
    elif masc == 0 and fem == 0 and mix > 0:
        gender = 'mixed'
    elif masc > 0 and fem > 0:
        gender = 'mixed'
        both = True
        

    return gender, genders, include, both
    
def run_rule_llm_fallback(test_rows, predictions, llm_model, llm_tokenizer, responses_outfile=None, results_outfile=None):

    rule_llm_predictions = []
    llm_responses = []

    for row, label in zip(test_rows, predictions):
        if label != 'ambiguous':
            rule_llm_predictions.append(label)
        else:
            gender, prompts_and_responses = llm_filter(llm_model, llm_tokenizer, row, label)
            rule_llm_predictions.append(gender)
            llm_responses.append(prompts_and_responses)

    if responses_outfile is not None:
        with open(responses_outfile, 'w') as outfile:
            for case in llm_responses:
                for example in case:
                    if len(example) == 2:
                        prompt = example[0]
                        answer = example[1]
                        outfile.write(prompt + '\t' + answer + '\n')

    if results_outfile is not None:
        with open(results_outfile, 'w') as outfile:
            for row, result in zip(test_rows, rule_llm_predictions):
                outfile.write('\t'.join(row) + '\t' + result + '\n')

    return rule_llm_predictions


def pronoun_gender(token_index, entity_clusters, spacy_doc):
    """
    checks whether the token at token_index is the head of a mention span
    in its coreference cluster, and whether the cluster contains male/female pronouns.

    returns 'masc', 'fem', or 'neutral'.
    """

    token_to_cluster = build_token_to_cluster(entity_clusters, spacy_doc)
    
    if token_index not in token_to_cluster:
        return 'neutral'

    cluster_words, cluster_spans = token_to_cluster[token_index]

    is_head = False
    for span in cluster_spans:
        head = spacy_doc[span[0]:(span[1] + 1)].root
        if head.i == token_index:
            is_head = True
            break

    if not is_head:
        return 'neutral'

    masculine_pronouns = 0
    feminine_pronouns = 0
    for word in cluster_words:
        if word.lower() in ['he', 'him', 'his', 'himself']:
            masculine_pronouns += 1
        elif word.lower() in ['she', 'her', 'herself']:
            feminine_pronouns += 1

    if masculine_pronouns > feminine_pronouns:
        return 'masc'
    elif feminine_pronouns > masculine_pronouns:
        return 'fem'
    else:
        return 'neutral'


def build_feature_dict(token_index, token, spacy_doc, entity_dict, fem_mod_indexes, masc_mod_indexes, person_indexes):
    """
    builds the feature dict for one token.
    """
    return {
        'lexical_gender': lexical_gender(token.lemma_),
        'modifier_gender': modifier_gender(token, fem_mod_indexes, masc_mod_indexes),
        'plural_conj_cluster': plural_conj_cluster(
            token, entity_dict, spacy_doc,
            fem_mod_indexes, masc_mod_indexes, person_indexes
        ),
        #'lexical_ref_gender': lexical_ref_gender(
         #   token.i, entity_dict, spacy_doc,
          #  fem_mod_indexes, masc_mod_indexes, person_indexes
        #),
        'pronoun_gender': pronoun_gender(token_index, entity_dict, spacy_doc),
        'closest_masc_path': closest_gendered_dep_path(
            token, spacy_doc, fem_mod_indexes, masc_mod_indexes, person_indexes, gender='masc'
        ),
        'closest_fem_path': closest_gendered_dep_path(
            token, spacy_doc, fem_mod_indexes, masc_mod_indexes, person_indexes, gender='fem'
        ),
    }


def load_dataset(filepath, dataset_type, coref_model):
    """
    loads a dataset file, parses each example, builds
    feature dict and label for the seed token.

    returns two lists: X_dicts (feature dicts) and y (gold labels).
    examples where the seed token could not be located are skipped.
    """
    examples = prepare_examples(filepath, dataset_type)

    X_dicts = []
    y = []
    skipped = 0

    for example in examples:
        spacy_doc = example['spacy_doc']
        token_index = example['seed_token_index']

        if token_index is None:
            skipped = skipped + 1
            continue

        token = spacy_doc[token_index]

        fem_mod_indexes, masc_mod_indexes, attr_indexes, person_indexes = syntactic_indexes(spacy_doc)
        tokens = [t.text for t in spacy_doc]
        entity_dict = coref_model.predict(tokens, singletons=True)

        feature_dict = build_feature_dict(
            token, spacy_doc, entity_dict,
            fem_mod_indexes, masc_mod_indexes, person_indexes
        )

        X_dicts.append(feature_dict)
        y.append(example['label'])

    print(filepath, 'loaded:', len(X_dicts), 'skipped:', skipped)

    return X_dicts, y


def evaluate_rule_llm(gold, predictions, title=None):
    """
    prints a classification report and plots a confusion matrix
    for rule-based+llm fallback predictions.
    """
    labels = sorted(set(predictions))

    report = classification_report(gold, predictions, digits=3, target_names=labels)
    print(report)

    cm = confusion_matrix(gold, predictions, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def build_text_and_span(example, dataset_type):
    """
    returns the sentence text and the character range in which the seed
    should be looked up.
    
    """
    sentence = example['sentence']
    return sentence, 0, len(sentence)



def find_seed_token_index(spacy_doc, seed, sentence_start_char, sentence_end_char):
    """
    returns the index of the head token of the seed phrase in the spacy doc within the specified span
    """
    full_text = spacy_doc.text
    section = full_text[sentence_start_char:sentence_end_char]

    seed_lower = seed.lower()
    pos_in_section = section.lower().find(seed_lower)

    if pos_in_section == -1:
        return None

    char_start = sentence_start_char + pos_in_section
    char_end = char_start + len(seed)

    span = spacy_doc.char_span(char_start, char_end, alignment_mode='expand')

    if span is not None and len(span) > 0:
        return span.root.i

    seed_words = seed_lower.split()
    last_word = seed_words[-1]

    for token in spacy_doc:
        if token.idx < sentence_start_char or token.idx >= sentence_end_char:
            continue
        if token.text.lower() == last_word:
            return token.i

    return None

    

def run_rule_llm_fallback(rows, predictions, llm_model, llm_tokenizer, responses_outfile, results_outfile, gold):

    rule_llm_predictions = []
    rule_based_llm_responses = []

    for row, label in zip(rows, predictions):
        if label != 'ambiguous':
            rule_llm_predictions.append(label)
            #print('just normal stuff, nothing to see here')
        if label == 'ambiguous':
            gender, prompts_and_responses = llm_filter(llm_model, llm_tokenizer, row, label)
            rule_llm_predictions.append(gender)
            #print('omg the llm intervened')
            rule_based_llm_responses.append(prompts_and_responses)

    with open(responses_outfile, 'w') as outfile:
        for case in rule_based_llm_responses:
            for example in case:
                if len(example) == 2:
                    prompt = example[0]
                    answer = example[1]
                    outfile.write(prompt + '\t' + answer + '\n')

    with open(results_outfile, 'w') as outfile:
        for row, result in zip(rows, rule_llm_predictions):
            outfile.write('\t'.join(row) + '\t' + result + '\n')

    return rule_llm_predictions


def write_results(outfile_path, test_rows, predictions):
    with open(outfile_path, 'w') as outfile:
        for row, result in zip(test_rows, predictions):
            outfile.write('\t'.join(row) + '\t' + result + '\n')



def parse_sent(sentence):
    """
    parses a sentence that is either a string or a list of tokens.
    returns spacy doc.

    param sentence: string or list of strings
    """
    if type(sentence) == list:
        spacy_doc = Doc(nlp.vocab, words=sentence)
        for what, docdoc in nlp.pipeline:
            spacy_doc = docdoc(spacy_doc)
    else:
        spacy_doc = nlp(sentence)
    return spacy_doc


def load_glitter(filepath):
    """
    loads the glitter tsv file (no header).
    columns: seed, prev_context, sentence, next_context, label
    """
    examples = []
    with open(filepath) as f:
        reader = csv.reader(f, delimiter='\t', quoting = csv.QUOTE_NONE)
        for row in reader:
            if len(row) != 5:
                continue
            seed, prev_context, sentence, next_context, label = row
            examples.append({'seed': seed,
                'prev_context': prev_context,
                'sentence': sentence,
                'next_context': next_context,
                'label': label})
    return examples



def load_fairtranslate(filepath):
    """
    loads the fairtranslate tsv file (with header).
    columns: seed, sentence, label, stereotype
    """
    examples = []
    with open(filepath) as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        header = next(reader)
        for row in reader:
            if len(row) != 4:
                continue
            seed, sentence, label, stereotype = row
            
            examples.append({'seed': seed,
                'sentence': sentence,
                'label': label,
                'stereotype': stereotype})
    return examples

def load_gente(filepath):
    """
    loads the gente tsv file (with header).
    columns: seed, sentence, label
    """
    examples = []
    with open(filepath) as f:
        reader = csv.reader(f, delimiter='\t', quoting = csv.QUOTE_NONE)
        header = next(reader)
        for row in reader:
            if len(row) != 3:
                continue
            seed, sentence, label = row
            examples.append({'seed': seed,
                'sentence': sentence,
                'label': label})
    return examples


def load_lexical(filepath):
    """
    loads the lexical tsv file (with header).
    columns: seed, sentence, label, stereotype
    """
    examples = []
    with open(filepath) as f:
        reader = csv.reader(f, delimiter='\t', quoting = csv.QUOTE_NONE)
        header = next(reader)
        for row in reader:
            if len(row) != 4:
                continue
            seed, sentence, label, stereotype = row
            examples.append({'seed': seed,
                'sentence': sentence,
                'label': label,
                'stereotype': stereotype})
    return examples


def file_to_dict(path):
    
    with open(path) as infile:
        dict_reader = DictReader(infile, delimiter = '\t', quoting = csv.QUOTE_NONE)
        list_of_dicts = list(dict_reader)
    
    return list_of_dicts


def report_and_matrix(path, dataset, system = 'not llm'):
    predictions = []
    gold = []

    dataset_dcts = file_to_dict(path)

    print(dataset_dcts[0])
    
    if system == 'llm':
        qwen3b_dicts = []

        with open(path) as infile:
            content = infile.read()
        
        lines = content.splitlines()
        headers = lines[0].split("\t")
        n_tabs = len(headers) - 1  
        
        complete_lines = []
        current = ""
        for line in lines[1:]:
            current = current + "\n" + line if current else line
            if current.count("\t") == n_tabs:
                complete_lines.append(current)
                current = ""
        
        rows = []
        for line in complete_lines:
            values = line.split("\t")
            rows.append(dict(zip(headers, values)))

        dataset_dcts = rows
        
    if dataset == 'all':
        for dct in dataset_dcts:
            if system == 'llm':
                predictions.append(dct['llm_label'])
            else:
                predictions.append(dct['predicted'])
            gold.append(dct['label'])
    else:
        for dct in dataset_dcts:
            if dct['dataset_type'] == dataset:
                if system == 'llm':
                    predictions.append(dct['llm_label'])
                else:
                    predictions.append(dct['predicted'])
                gold.append(dct['label'])

    #print(predictions)            

    if system == 'llm':
        llm_prds = []
        for prd in predictions:
            if prd == 'AMBIGUOUS':
                llm_prds.append('ambiguous')
            elif prd == 'UNAMBIGUOUS (FEMALE)':
                llm_prds.append('unambiguous_female')
            elif prd == 'UNAMBIGUOUS (MALE)':
                llm_prds.append('unambiguous_male')
            elif prd == 'UNAMBIGUOUS (BOTH)':
                llm_prds.append('unambiguous_all')
            #else:
                #print('help do something')
                #print(prd)

        predictions = llm_prds
        #print(predictions)
        
    print(classification_report(gold, predictions, digits = 3))

    labels = list(set(gold + predictions))

    cm = confusion_matrix(gold, predictions, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    plt.show()

def f1(result_dict, cue):
    pred = []
    gold = []
    
    for dct in result_dict:
        if dct['cue_type'] == cue:
            pred.append(dct['predicted'])
            gold.append(dct['label'])
    f1 = f1_score(gold, pred, average="weighted")
    return f1