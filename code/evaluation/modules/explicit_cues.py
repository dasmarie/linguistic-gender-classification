#from maverick import Maverick
import spacy
from spacy.tokens import Doc
import sys
from modules.utils.dict_utils import check_dictionary
from nltk.corpus import wordnet as wn
import gender_guesser.detector as gender
from transformers import AutoModelForCausalLM, AutoTokenizer



nlp = spacy.load("en_core_web_md")
d = gender.Detector()

#model = Maverick(
 # hf_name_or_path = "sapienzanlp/maverick-mes-preco",
  #device = "cpu"
#)

#model.model = model.model.float() 


masc_lex = ['abbot', 'abbots', 'alderman', 'artilleryman', 'assemblyman', 'bachelor', 'batsman', 'beau', 'boy', 'boyfriend', 'boyfriends', 'boys', 'brother', 'brothers', 'bull', 'businessman', 'businessmen', 'cameraman', 'cameramen', 'cavalrymen', 'chairman', 'chevalier', 'churchmen', 'clergyman', 'countryman', 'cowboy', 'cowboys', 'craftsmen', 'crewmen', 'dad', 'daddy', 'draughtsman', 'dudes', 'emperor', 'emperors', 'eunuchs', 'father', 'fathers', 'fireman', 'firemen', 'fisherman', 'fishermen', 'freedmen', 'freemen', 'freshman', 'freshmen', 'gentlemen', 'godfather', 'gods', 'grandfather', 'grandnephew', 'grandson', 'guy', 'guys', 'helmsman', 'henchman', 'henchmen', 'husband', 'husbands', 'infantrymen', 'king', 'kinsman', 'landlord', 'landlords', 'laymen', 'lineman', 'male', 'males', 'man', 'men', 'middleman', 'middlemen', 'midshipman', 'milkman', 'monk', 'monks', 'nephew', 'nephews', 'nobleman', 'patriarch', 'patrolman', 'policeman', 'pope', 'postman', 'prince', 'princes', 'rabbi', 'showman', 'sir', 'sire', 'son', 'sons', 'spokesman', 'sportsman', 'statesman', 'stepbrother', 'stepbrothers', 'stepfather', 'stepson', 'strongman', 'tradesmen', 'trainmen', 'uncle', 'uncles', 'underclassmen']


fem_lex = ['actress', 'actresses', 'aunt', 'barmaid', 'bride', 'brides', 'bridesmaid', 'chambermaid', 'councilwoman', 'courtesan', 'courtesans', 'dames', 'daughter', 'daughters', 'empress', 'female', 'females', 'gal', 'geishas', 'girl', 'girlfriend', 'girlfriends', 'girls', 'granddaughter', 'grandmother', 'heiress', 'hostess', 'housemaid', 'housewife', 'ladies', 'lady', 'landlady', 'lesbian', 'lesbians', 'madam', 'maid', 'maiden', 'maids', 'mater', 'miss', 'mistress', 'mistresses', 'mother', 'mothers', 'mummy', 'niece', 'nun', 'nuns', 'priestesses', 'princess', 'princesses', 'prioress', 'prostitute', 'prostitutes', 'queen', 'sister', 'sisters', 'wench', 'widow', 'widows', 'wife', 'wives', 'woman', 'women']


neutral_lex = ['admirer', 'baggage', 'baritone', 'bass', 'beast', 'beasts', 'beauty', 'beaver', 'bombshells', 'buddy', 'cousin', 'cows', 'creole', 'cuckoo', 'dean', 'devil', 'devils', 'dish', 'dishes', 'diva', 'divas', 'dive', 'dolls', 'dragon', 'dragons', 'drone', 'dwarf', 'dwarfs', 'equestrian', 'fairy', 'fellow', 'fellows', 'flirt', 'gob', 'god', 'ham', 'headmaster', 'hero', 'heroes', 'heros', 'hijackers', 'hogs', 'homunculus', 'housekeeper', 'intersex', 'jade', 'jewel', 'junior', 'knockout', 'knockouts', 'letters', 'lizard', 'louts', 'lover', 'lovers', 'master', 'member', 'members', 'metropolitan', 'mezzo', 'minor', 'minors', 'name', 'names', 'nurse', 'nurses', 'orphan', 'parachutists', 'patron', 'patrons', 'peaches', 'pop', 'punchers', 'punk', 'rams', 'rapists', 'reaper', 'rock', 'rocks', 'sage', 'sages', 'satyr', 'schoolmaster', 'scrubber', 'senior', 'seniors', 'siren', 'skinhead', 'skinheads', 'skirt', 'skirts', 'soprano', 'spectator', 'spectators', 'sphinx', 'stud', 'sweethearts', 'tart', 'tenor', 'thugs', 'virgin', 'wolf', 'wolves', 'youth', 'youths']




seeds_gente = [
    'Ambassador', 'Chairman', 'Chief', 'Commissioner', 'Councillor',
    'General', 'Member', 'Minister', 'Ombudsman', 'President', 'Secretary',
    'artist', 'author', 'activist', 'actor', 'advocate', 'ancestor',
    'assistant', 'attorney', 'beneficiary', 'breeder', 'broadcaster',
    'businessman', 'candidate', 'champion', 'chairperson', 'chairman',
    'child', 'citizen', 'colleague', 'compatriot', 'consumer', 'coordinator',
    'criminal', 'culprit', 'daughter', 'dealer', 'defendant', 'delegate',
    'democrat', 'demonstrator', 'deputy', 'dietician', 'director', 'distributor',
    'doctor', 'draftsman', 'draftsperson', 'driver', 'economist', 'entrepreneur',
    'expert', 'executive', 'farmer', 'federalist', 'feminist', 'fisherman',
    'friend', 'functionary', 'governor', 'girl', 'grower', 'guardian',
    'historian', 'immigrant', 'inhabitant', 'inspector', 'investor',
    'journalist', 'judge', 'justice', 'labourer', 'lawyer', 'leader',
    'manager', 'manufacturer', 'member', 'militant', 'minister', 'murderer',
    'negotiator', 'observer', 'officer', 'official', 'ombudsman', 'operator',
    'owner', 'parliamentarian', 'parent', 'patient', 'pensioner', 'player',
    'politician', 'president', 'prisoner', 'producer', 'promoter',
    'prosecutor', 'rapporteur', 'reader', 'refugee', 'representative',
    'researcher', 'resident', 'scientist', 'servant', 'sister', 'socialist',
    'son', 'specialist', 'spokesperson', 'student', 'surgeon', 'teacher',
    'technician', 'terrorist', 'thief', 'traveller', 'veteran', 'victim',
    'visitor', 'winner', 'worker', 'woman', 'boy', 'brother', 'man',
    'seaman', 'rapporteur', 'representative', 'official']


def coref_clusters(spacy_doc):
    tokens = []
    for token in spacy_doc:
        tokens.append(token.text)
    entity_dict = coref_model.predict(tokens, singletons=True)
    return entity_dict


def lexical_gender(lemma):
    if lemma in fem_lex:
        lex_gender = 'fem'
    elif lemma in masc_lex:
        lex_gender = 'masc'
    elif lemma in neutral_lex:
        lex_gender = 'neutral'
    else:
        lex_gender = check_dictionary(
            word=lemma, dict_abbrev='wordnet',
            heuristics=True, seed_pairs=5,
            no_words=35, no_defs=10)
        
    return lex_gender


def collect_relevant_tokens(head, spacy_doc):
    """
    collects the head, its conjuncts, and additionally, if the head has a
    prepositional phrase with 'of' (e.g. 'a group of men and women'), also
    the pobj of that preposition and its conjuncts.
    """
    tokens = [head] + conj_chain(head)

    for child in head.children:
        if child.dep_ == 'prep' and child.lemma_.lower() == 'of':
            for pobj in child.children:
                if pobj.dep_ == 'pobj':
                    tokens.append(pobj)
                    tokens.extend(conj_chain(pobj))

    return tokens


def lexical_ref_gender(token_index, entity_dict, spacy_doc,
                        fem_mod_indexes, masc_mod_indexes, person_indexes):

    cluster_index = longest_cluster_for_token(token_index, entity_dict)

    if cluster_index is None:
        return 'no_coref'

    cluster_spans = entity_dict['clusters_token_offsets'][cluster_index]

    has_plural = False
    for span in cluster_spans:
        head = spacy_doc[span[0]:(span[1] + 1)].root
        if is_plural(head):
            has_plural = True

    if has_plural:
        number_part = 'plur'
    else:
        number_part = 'sing'

    found_genders = []

    for span in cluster_spans:
        head = spacy_doc[span[0]:(span[1] + 1)].root
        tokens_to_check = collect_relevant_tokens(head, spacy_doc)

        for t in tokens_to_check:
            is_person_token = (t.pos_ in ('NOUN', 'PROPN') and is_person(t.lemma_)) or t.i in person_indexes
            if not is_person_token:
                continue

            in_masc = masc_mod_indexes and t.i in masc_mod_indexes
            in_fem = fem_mod_indexes and t.i in fem_mod_indexes

            if in_masc and 'm' not in found_genders:
                found_genders.append('m')
            if in_fem and 'f' not in found_genders:
                found_genders.append('f')

            if t.lemma_ in masc_lex and 'm' not in found_genders:
                found_genders.append('m')
            if t.lemma_ in fem_lex and 'f' not in found_genders:
                found_genders.append('f')

    if len(found_genders) == 0:
        gender_part = 'neut'
    else:
        gender_part = ''
        for g in found_genders:
            gender_part = gender_part + g

    feature = number_part + '_' + gender_part
    return feature


def is_person(word):
    synsets = wn.synsets(word, pos=wn.NOUN)
    for syn in synsets:
        for hypernym in syn.closure(lambda s: s.hypernyms()):
            if hypernym == wn.synset('person.n.01'):
                return True
    return False
    
def is_plural(token):
    """
    checks whether token is plural.
    
    param token: token taken from spacy doc.
    
    """
    return token.morph.get('Number') == ['Plur']


def parse_sent(sentence):
    """
    parses a sentence that is either a string or a list of tokens.
    returns spacy doc.

    param sentence: string or list of strings
    """
    if type(sentence) == list:
        spacy_doc = Doc(nlp.vocab, words=sentence)
        for pipe_name, pipe in nlp.pipeline:
            spacy_doc = pipe(spacy_doc)
    else:
        spacy_doc = nlp(sentence)
    return spacy_doc


def conj_chain(token):
    """
    collects all conjuncts of a token (recursively).

    param token: token taken from spacy doc.
    """
    conjuncts = []
    for child in token.children:
        if child.dep_ == 'conj':
            conjuncts.append(child)
            conjuncts.extend(conj_chain(child))
    return conjuncts


def resolve_plural_gender(masc_count, fem_count):
    if masc_count > 0 and fem_count == 0:
        return 'unambiguous_male'
    if fem_count > 0 and masc_count == 0:
        return 'unambiguous_female'
    if masc_count > 0 and fem_count > 0:
        return 'unambiguous_all'
    return 'ambiguous'

def ner_person(token, spacy_doc, cluster_words=None):
    masc_titles = ['mr', 'sir', 'lord']
    fem_titles = ['mrs', 'ms', 'miss', 'lady', 'dame']

    for i in [token.i - 1, token.i - 2, token.i]:
        if i < 0:
            continue
        preceding = spacy_doc[i].text.lower().rstrip('.')
        if preceding in masc_titles:
            return 'unambiguous_male'
        if preceding in fem_titles:
            return 'unambiguous_female'

    if cluster_words:
        masc = sum(1 for w in cluster_words if w.lower() in ['he', 'him', 'his', 'himself'])
        fem = sum(1 for w in cluster_words if w.lower() in ['she', 'her', 'herself'])
        if masc > fem:
            return 'unambiguous_male'
        if fem > masc:
            return 'unambiguous_female'

    if d.get_gender(token.text) == 'female':
        return 'unambiguous_female'
    if d.get_gender(token.text) == 'male':
        return 'unambiguous_male'

    return 'ambiguous'



def gender_person_token(token, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc, force_indexes=None):
    
    if token.i in person_indexes:
        return ner_person(token, spacy_doc, cluster_words)

    is_forced = force_indexes and token.i in force_indexes

    if token.pos_ not in ('NOUN', 'PROPN') and not is_forced:
        return None

    if not is_person(token.lemma_) and not is_forced:
        return None

    in_masc = masc_mod_indexes and token.i in masc_mod_indexes
    in_fem = fem_mod_indexes and token.i in fem_mod_indexes

    if in_masc and in_fem:
        mod_gender = 'both'
    elif in_masc:
        mod_gender = 'masc'
    elif in_fem:
        mod_gender = 'fem'
    else:
        mod_gender = 'neutral'

    if token.lemma_ in masc_lex:
        lexical_gender = 'masc'
    elif token.lemma_ in fem_lex:
        lexical_gender = 'fem'
    elif token.lemma_ in neutral_lex:
        lexical_gender = 'neutral'
    else:
        lexical_gender = check_dictionary(
            word=token.lemma_, dict_abbrev='wordnet',
            heuristics=True, seed_pairs=5,
            no_words=35, no_defs=10)

    if lexical_gender not in ('masc', 'fem', 'both', 'neutral'):
        lexical_gender = 'neutral'

    masculine_pronouns = 0
    feminine_pronouns = 0

    for w in cluster_words:
        if w.lower() in ['he', 'him', 'his', 'himself']:
            masculine_pronouns += 1
        elif w.lower() in ['she', 'her', 'herself']:
            feminine_pronouns += 1

    if feminine_pronouns > masculine_pronouns:
        ref_gender = 'fem'
    elif masculine_pronouns > feminine_pronouns:
        ref_gender = 'masc'
    else:
        ref_gender = 'neutral'

    if ref_gender != 'neutral':
        fin_gender = ref_gender
    elif mod_gender != 'neutral':
        fin_gender = mod_gender
    elif lexical_gender != 'neutral':
        fin_gender = lexical_gender
    else:
        fin_gender = 'neutral'

    if fin_gender == 'fem':
        fin_gender = 'unambiguous_female'
    elif fin_gender == 'masc':
        fin_gender = 'unambiguous_male'
    elif fin_gender == 'both':
        fin_gender = 'unambiguous_all'
    elif fin_gender == 'neutral':
        fin_gender = 'ambiguous'

    return fin_gender



def attr_matches_head(subj_i, head, spacy_doc):
    if subj_i == head.i:
        return True
    for conj in conj_chain(head):
        if conj.i == subj_i:
            return True
    if head.dep_ == 'conj':
        conj_root = head.head
        if conj_root.i == subj_i:
            return True
        for conj in conj_chain(conj_root):
            if conj.i == subj_i:
                return True
    return False



def gender_cluster(cluster_words, cluster_spans, spacy_doc,
                  fem_mod_indexes, masc_mod_indexes, attr_indexes,
                  dependent_to_head, person_indexes):
    """
    assigns a gender label to one coreference cluster.
    """

    singular_masc = 0
    singular_fem = 0
    singular_results = []
    found_person = False

    for ref in cluster_spans:

        head = spacy_doc[ref[0]:(ref[1] + 1)].root

        if head.i in dependent_to_head:
            continue

        conjuncts = []
        for c in conj_chain(head):
            if c.i >= ref[0] and c.i <= ref[1] and c.i not in dependent_to_head:
                conjuncts.append(c)
        has_ambiguous = False
        has_gendered = False
        for t in [head] + conjuncts:
            if not (is_person(t.lemma_) or t.i in person_indexes):
                continue
            l = gender_person_token(t, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
            if l == 'ambiguous':
                has_ambiguous = True
            elif is_gendered(l):
                has_gendered = True
        
        if has_ambiguous and has_gendered:
            continue
        for token in [head] + conjuncts:
            if is_plural(token):
                continue
            label = gender_person_token(token, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
            if label is None:
                continue
            if is_person(token.lemma_) or token.i in person_indexes:
                found_person = True
            had_conjuncts = len(conjuncts) > 0
            singular_results.append((token.text, label, had_conjuncts))
            if label == 'unambiguous_male':
                singular_masc += 1
            elif label == 'unambiguous_female':
                singular_fem += 1

        # tokens connected through copulatives
        for (attr_i, subj_i, hd_i, is_partitive) in attr_indexes:
            if not attr_matches_head(subj_i, head, spacy_doc):
                continue
            attr_token = spacy_doc[attr_i]
            if is_plural(attr_token):
                continue
            attr_label = gender_person_token(attr_token, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
            if attr_label is None:
                continue
            if is_person(attr_token.lemma_) or attr_token.i in person_indexes:
                found_person = True
            # attr token keeps its own label; subj/head gets unambiguous_all if partitive
            subj_label_for_count = 'unambiguous_all' if (is_partitive and is_gendered(attr_label)) else attr_label
            singular_results.append((attr_token.text, attr_label, False))
            if subj_label_for_count == 'unambiguous_all':
                singular_masc += 1
                singular_fem += 1
            elif attr_label == 'unambiguous_male':
                singular_masc += 1
            elif attr_label == 'unambiguous_female':
                singular_fem += 1

    # plural tokens
    plural_masc = singular_masc
    plural_fem = singular_fem
    has_plural = False

    for ref in cluster_spans:
        head = spacy_doc[ref[0]:(ref[1] + 1)].root

        if head.i in dependent_to_head:
            continue

        for token in [head] + conj_chain(head):
            if not is_plural(token):
                continue
            label = gender_person_token(token, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
            has_plural = True
            if label == 'unambiguous_male':
                plural_masc += 1
            elif label == 'unambiguous_female':
                plural_fem += 1
            elif label == 'unambiguous_all':
                plural_masc += 1
                plural_fem  += 1

        for (attr_i, subj_i, whatever, is_partitive) in attr_indexes:
            if not attr_matches_head(subj_i, head, spacy_doc):
                continue
            attr_token = spacy_doc[attr_i]
            if not is_plural(attr_token):
                continue
            attr_label = gender_person_token(attr_token, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
            has_plural = True
            subj_label_for_count = 'unambiguous_all' if (is_partitive and is_gendered(attr_label)) else attr_label
            if subj_label_for_count == 'unambiguous_all':
                plural_masc += 1
                plural_fem += 1
            elif attr_label == 'unambiguous_male':
                plural_masc += 1
            elif attr_label == 'unambiguous_female':
                plural_fem += 1

    if not found_person and not has_plural:
        return None

    # final label
    if has_plural:
        entity_gender = resolve_plural_gender(plural_masc, plural_fem)
    else:
        masc = 0
        fem = 0
        for entry in singular_results:
            label = entry[1]
            if label == 'unambiguous_male':
                masc += 1
            elif label == 'unambiguous_female':
                fem += 1

        if masc > fem:
            entity_gender = 'unambiguous_male'
        elif fem > masc:
            entity_gender = 'unambiguous_female'
        else:
            entity_gender = 'ambiguous'

    return {'entity_gender': entity_gender}



def update_cluster_gender(results, spacy_doc):
    """
    updates a label to the more informative one if a singleton cluster shares a token with a larger cluster.

    param results: list of dicts with mentions and entity gender.
    """
    
    label_priority = ['unambiguous_all', 'unambiguous_female', 'unambiguous_male', 'ambiguous']

    #dict with gender labels: maps token (index) to gender label
    token_gender_dict = {}
    for entity in results:
        for mention in entity['mentions']:
            span = find_span(mention, spacy_doc)
            if span is None:
                
                continue
            for token in span:
                token_gender_dict[token.i] = entity['entity_gender']
            for conj in conj_chain(span.root):
                token_gender_dict[conj.i] = entity['entity_gender']

    #update singleton gender
    for entity in results:
        if len(entity['mentions']) != 1:
            continue
        span = find_span(entity['mentions'][0], spacy_doc)
        if span is None:
            continue
        for token in span:
            if token.i in token_gender_dict:
                new_label = token_gender_dict[token.i]
                if label_priority.index(new_label) < label_priority.index(entity['entity_gender']):
                    entity['entity_gender'] = new_label
                break

    return results

    #for token in span:
     #   if token.i in token_gender_dict:
            #print('match found for', token.text, '->', token_gender_dict[token.i])



def is_negated(token):
    return any(child.dep_ == 'neg' for child in token.children)


def syntactic_indexes(spacy_doc):
    partitive_quantifiers = {
        'some', 'several', 'many', 'few', 'half', 'part', 'most',
        'majority', 'minority', 'any', 'certain', 'various',
        'handful', 'couple', 'bunch', 'number', 'portion',
        'fraction', 'share', 'lot', 'none'
    }

    inclusive_verbs = {
        'include', 'contain', 'feature', 'encompass', 'involve'
    }

    exhaustive_verbs_prep = {
        'consist'
    }

    exhaustive_verbs_dobj = {
        'comprise'
    }

    fem_mod_indexes  = list()
    masc_mod_indexes = list()
    attr_indexes = list()
    person_indexes = list()

    for ent in spacy_doc.ents:
        if ent.label_ == 'PERSON':
            for token in spacy_doc[ent.start:ent.end]:
                person_indexes.append(token.i)

    for word in spacy_doc:

        if word.dep_ == 'amod':
            all_mods = [word] + conj_chain(word)
            for mod in all_mods:
                if str(mod).lower() == 'female':
                    fem_mod_indexes.append(word.head.i)
                elif str(mod).lower() == 'male':
                    masc_mod_indexes.append(word.head.i)

        elif word.dep_ == 'compound':
            all_mods = [word] + conj_chain(word)
            for mod in all_mods:
                if str(mod.lemma_) == 'woman' or str(mod.lemma_) == 'women':
                    fem_mod_indexes.append(word.head.i)
                elif str(mod.lemma_) == 'man':
                    masc_mod_indexes.append(word.head.i)

        elif word.dep_ == 'attr':
            if not is_negated(word.head):
                for w in spacy_doc:
                    if w.dep_ == 'nsubj' and w.head.i == word.head.i:
                        attr_indexes.append((word.i, w.i, word.head.i, False))

        elif word.dep_ == 'relcl':
            head = word.head
            if head.dep_ == 'conj':
                head = head.head

            nsubj = None
            for child in word.children:
                if child.dep_ == 'nsubj':
                    nsubj = child
                    break

            is_partitive = (
                nsubj is not None and (
                    nsubj.lemma_.lower() in partitive_quantifiers
                    or nsubj.pos_ == 'NUM'
                    or nsubj.dep_ == 'nummod'
                )
            )

            if not is_negated(word):
                for child in word.children:
                    if child.dep_ == 'attr':
                        attr_indexes.append((child.i, head.i, word.i, is_partitive))

                        for head_child in head.children:
                            if head_child.dep_ == 'prep':
                                for pobj in head_child.children:
                                    if pobj.dep_ == 'pobj' and pobj.pos_ in ('NOUN', 'PROPN'):
                                        attr_indexes.append((child.i, pobj.i, word.i, is_partitive))

                if word.lemma_.lower() in inclusive_verbs:
                    for child in word.children:
                        if child.dep_ == 'dobj' and child.pos_ in ('NOUN', 'PROPN'):
                            attr_indexes.append((child.i, head.i, word.i, True))
                            for head_child in head.children:
                                if head_child.dep_ == 'prep':
                                    for pobj in head_child.children:
                                        if pobj.dep_ == 'pobj' and pobj.pos_ in ('NOUN', 'PROPN'):
                                            attr_indexes.append((child.i, pobj.i, word.i, True))
                            for conj in conj_chain(child):
                                if conj.pos_ in ('NOUN', 'PROPN'):
                                    attr_indexes.append((conj.i, head.i, word.i, True))

                elif word.lemma_.lower() in exhaustive_verbs_prep:
                    for child in word.children:
                        if child.dep_ == 'prep' and child.lemma_.lower() == 'of':
                            for pobj in child.children:
                                if pobj.dep_ == 'pobj' and pobj.pos_ in ('NOUN', 'PROPN'):
                                    attr_indexes.append((pobj.i, head.i, word.i, False))
                                    for head_child in head.children:
                                        if head_child.dep_ == 'prep':
                                            for grandpobj in head_child.children:
                                                if grandpobj.dep_ == 'pobj' and grandpobj.pos_ in ('NOUN', 'PROPN'):
                                                    attr_indexes.append((pobj.i, grandpobj.i, word.i, False))
                                    for conj in conj_chain(pobj):
                                        if conj.pos_ in ('NOUN', 'PROPN'):
                                            attr_indexes.append((conj.i, head.i, word.i, False))
                                            for head_child in head.children:
                                                if head_child.dep_ == 'prep':
                                                    for grandpobj in head_child.children:
                                                        if grandpobj.dep_ == 'pobj' and grandpobj.pos_ in ('NOUN', 'PROPN'):
                                                            attr_indexes.append((conj.i, grandpobj.i, word.i, False))

                elif word.lemma_.lower() in exhaustive_verbs_dobj:
                    for child in word.children:
                        if child.dep_ == 'dobj' and child.pos_ in ('NOUN', 'PROPN'):
                            attr_indexes.append((child.i, head.i, word.i, False))
                            for head_child in head.children:
                                if head_child.dep_ == 'prep':
                                    for pobj in head_child.children:
                                        if pobj.dep_ == 'pobj' and pobj.pos_ in ('NOUN', 'PROPN'):
                                            attr_indexes.append((child.i, pobj.i, word.i, False))
                            for conj in conj_chain(child):
                                if conj.pos_ in ('NOUN', 'PROPN'):
                                    attr_indexes.append((conj.i, head.i, word.i, False))
                                    for head_child in head.children:
                                        if head_child.dep_ == 'prep':
                                            for pobj in head_child.children:
                                                if pobj.dep_ == 'pobj' and pobj.pos_ in ('NOUN', 'PROPN'):
                                                    attr_indexes.append((conj.i, pobj.i, word.i, False))

        # inclusive verbs in main clause (not relcl)
        if word.lemma_.lower() in inclusive_verbs and word.dep_ != 'relcl':
            if not is_negated(word):
                nsubj = None
                for child in word.children:
                    if child.dep_ == 'nsubj':
                        nsubj = child
                        break
                if nsubj is not None:
                    for child in word.children:
                        if child.dep_ == 'dobj' and child.pos_ in ('NOUN', 'PROPN'):
                            attr_indexes.append((child.i, nsubj.i, word.i, True))
                            for conj in conj_chain(child):
                                if conj.pos_ in ('NOUN', 'PROPN'):
                                    attr_indexes.append((conj.i, nsubj.i, word.i, True))

    for token in spacy_doc:
        if token.dep_ == 'appos':
            head = token.head
            attr_indexes.append((token.i, head.i, None, False))
            for conj in conj_chain(token):
                attr_indexes.append((conj.i, head.i, None, False))

    return fem_mod_indexes, masc_mod_indexes, attr_indexes, person_indexes

def is_gendered(label):
    """returns True if the label is a clear gender (not ambiguous or neutral)."""
    return label != 'ambiguous' and label != 'neutral' and label is not None

def resolve_from_labels(label_list):
    has_male = False
    has_female = False
    has_ambiguous = False
    for l in label_list:
        if l == 'unambiguous_male':
            has_male = True
        elif l == 'unambiguous_female':
            has_female = True
        elif l == 'ambiguous':
            has_ambiguous = True
    if has_ambiguous and (has_male or has_female):
        return 'ambiguous'
    if has_male and has_female:
        return 'unambiguous_all'
    if has_male:
        return 'unambiguous_male'
    if has_female:
        return 'unambiguous_female'
    return 'ambiguous'



def find_span(mention_text, spacy_doc):
    """
    finds the first span in spacy_doc whose text matches mention_text.
    """
    for i in range(len(spacy_doc)):
        for j in range(i + 1, len(spacy_doc) + 1):
            if spacy_doc[i:j].text == mention_text:
                return spacy_doc[i:j]
    return None


def labeled_text(spacy_doc, label_map):
    """
    inserts gender labels into input text.
    """
    tokens = []
    for token in spacy_doc:
        if token.i in label_map:
            tokens.append('[' + label_map[token.i] + '] ' + token.text)
        else:
            tokens.append(token.text)
    output = ''
    for i in range(len(tokens)):
        output += tokens[i] + spacy_doc[i].whitespace_
    return output



def is_more_informative_label(new_label, current_label):
    """
    returns True if new_label is more informative than current_label.
    order from most to least informative:
    unambiguous_all > unambiguous_female > unambiguous_male > ambiguous
    """
    label_priority = ['unambiguous_all', 'unambiguous_female', 'unambiguous_male', 'ambiguous']
    if new_label not in label_priority:
        return False
    if current_label not in label_priority:
        return True
    return label_priority.index(new_label) < label_priority.index(current_label)


def update_label_map(label_map, token_i, new_label, individual_labels=None):
    if new_label is None:
        return
    if individual_labels and token_i in individual_labels:
        return
    if token_i not in label_map:
        label_map[token_i] = new_label
    elif is_more_informative_label(new_label, label_map[token_i]):
        label_map[token_i] = new_label


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

#def fallback_nli(token_id, spacy_doc, model_nli=model_nli, tokenizer_nli=tokenizer_nli):
#    gender, genderlist, include, both = fallback_cues(spacy_doc)
#    if gender == 'ambiguous':
#        return 'ambiguous'
#    elif gender != 'mixed' and include == False:
#        ask = [gender]
#    elif gender == 'mixed' and both == False:
#        ask = ['mixed-gender']
#    elif (gender == 'mixed' and both == True) or (gender != 'mixed' and include == True):
#        ask = ['men and women', 'female', 'male']
#    else:
#        return 'ambiguous'

#    if len(ask) == 1:
#        features_nli = tokenizer_nli([spacy_doc.text], ['The ' + spacy_doc[token_id].text + ' are ' + ask[0]], padding=True, truncation=True, return_tensors="pt")
#        model_nli.eval()
#        with torch.no_grad():
#            scores = model_nli(**features_nli).logits
#            label_mapping = ['contradiction', 'entailment', 'neutral']
#            labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
#        if labels[0] == 'entailment':
#            if ask[0] == 'mixed-gender':
#                return 'unambiguous_all'
#            elif ask[0] == 'female':
#                return 'unambiguous_female'
#            elif ask[0] == 'male':
#                return 'unambiguous_male'
#        else:
#            return 'ambiguous'

#    elif len(ask) > 1:
#        for i in range(len(ask)):
#            features_nli = tokenizer_nli([spacy_doc.text], ['The ' + spacy_doc[token_id].text + ' are ' + ask[i]], padding=True, truncation=True, return_tensors="pt")
#            model_nli.eval()
#            with torch.no_grad():
#                scores = model_nli(**features_nli).logits
#                label_mapping = ['contradiction', 'entailment', 'neutral']
#                labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
#                if labels[0] == 'entailment':
#                    break
#        if labels[0] == 'entailment':
#            if ask[i] == 'mixed-gender':
#                return 'unambiguous_all'
#            elif ask[i] == 'female':
#                return 'unambiguous_female'
#            elif ask[i] == 'male':
#                return 'unambiguous_male'
#        else:
#            return 'ambiguous'

    
            


def build_token_to_cluster(entity_dict, spacy_doc):
    """
    builds a dict mapping token index -> (cluster_words, cluster_spans)
    for every token that appears in any coref cluster.
    """
    token_to_cluster = {}
    for cluster_words, cluster_spans in zip(
        entity_dict['clusters_token_text'],
        entity_dict['clusters_token_offsets']
    ):
        for span in cluster_spans:
            for i in range(span[0], span[1] + 1):
                token_to_cluster[i] = (cluster_words, cluster_spans)
    return token_to_cluster

def update_label_map(label_map, token_i, new_label, individual_labels=None):
    if new_label is None:
        return
    if individual_labels and token_i in individual_labels:
        return
    if token_i not in label_map:
        label_map[token_i] = new_label
    elif is_more_informative_label(new_label, label_map[token_i]):
        label_map[token_i] = new_label

def longest_cluster_for_token(token_index, entity_dict):
    """
    finds the cluster (by index into entity_dict) containing token_index
    with the most mentions. ties are broken by total token length
    across all mentions.
    """
    cluster_spans_list = entity_dict['clusters_token_offsets']

    best_index = None
    best_mentions = -1
    best_length = -1

    for c in range(len(cluster_spans_list)):
        spans = cluster_spans_list[c]
        found = False
        for span in spans:
            if token_index >= span[0] and token_index <= span[1]:
                found = True
                break

        if not found:
            continue

        num_mentions = len(spans)
        total_length = 0
        for span in spans:
            total_length = total_length + (span[1] - span[0] + 1)

        if num_mentions > best_mentions:
            best_index = c
            best_mentions = num_mentions
            best_length = total_length
        elif num_mentions == best_mentions and total_length > best_length:
            best_index = c
            best_mentions = num_mentions
            best_length = total_length

    return best_index

def plural_conj_cluster(token, entity_dict, spacy_doc, fem_mod_indexes, masc_mod_indexes, person_indexes):
    
    if is_plural(token):
        number_part = 'plural'
    else:
        number_part = 'sing'

    is_conj = False
    if token.dep_ == 'conj':
        is_conj = True
    for child in token.children:
        if child.dep_ == 'conj':
            is_conj = True

    if is_conj:
        conj_part = 'conj'
    else:
        conj_part = 'noconj'

    cluster_index = longest_cluster_for_token(token.i, entity_dict)

    if cluster_index is None:
        cluster_plural_part = 'noplural'
        gender_set_part = 'n'
        cluster_words = []
    else:
        cluster_spans = entity_dict['clusters_token_offsets'][cluster_index]
        cluster_words = entity_dict['clusters_token_text'][cluster_index]

        cluster_has_plural = False
        for span in cluster_spans:
            head = spacy_doc[span[0]:(span[1] + 1)].root
            if is_plural(head):
                cluster_has_plural = True

        if cluster_has_plural:
            cluster_plural_part = 'plural'
        else:
            cluster_plural_part = 'noplural'

        gender_letters = []
        for span in cluster_spans:
            head = spacy_doc[span[0]:(span[1] + 1)].root
            tokens_to_check = collect_relevant_tokens(head, spacy_doc)

            for ref_token in tokens_to_check:
                if not (is_person(ref_token.lemma_) or ref_token.i in person_indexes):
                    continue

                ref_label = gender_person_token(
                    ref_token, cluster_words,
                    fem_mod_indexes, masc_mod_indexes,
                    person_indexes, spacy_doc
                )

                if ref_label == 'unambiguous_male':
                    letter = 'm'
                elif ref_label == 'unambiguous_female':
                    letter = 'f'
                elif ref_label == 'unambiguous_all':
                    letter = 'a'
                elif ref_label == 'ambiguous':
                    letter = 'n'
                else:
                    letter = None

                if letter is not None and letter not in gender_letters:
                    gender_letters.append(letter)

        gender_set_part = ''
        if 'm' in gender_letters:
            gender_set_part = gender_set_part + 'm'
        if 'f' in gender_letters:
            gender_set_part = gender_set_part + 'f'
        if 'n' in gender_letters:
            gender_set_part = gender_set_part + 'n'
        if 'a' in gender_letters:
            gender_set_part = gender_set_part + 'a'

        if gender_set_part == '':
            gender_set_part = 'n'

    own_label = gender_person_token(
        token, cluster_words,
        fem_mod_indexes, masc_mod_indexes,
        person_indexes, spacy_doc
    )

    if own_label == 'unambiguous_male':
        own_part = 'masc'
    elif own_label == 'unambiguous_female':
        own_part = 'fem'
    elif own_label == 'unambiguous_all':
        own_part = 'all'
    else:
        own_part = 'neut'

    feature = number_part + '_' + cluster_plural_part + '_' + gender_set_part + '_' + own_part
    return feature

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
        if lemma == 'include' or lemma == 'contain':
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

def get_ancestor_chain(token):
    """
    returns a list of tokens from token itself up to the root
    (inclusive), following head relations.
    """
    chain = []
    chain.append(token)
    current = token
    while current.head.i != current.i:
        current = current.head
        chain.append(current)
    return chain


def find_lowest_common_ancestor(token1, token2):
    chain1 = get_ancestor_chain(token1)
    chain2 = get_ancestor_chain(token2)

    lca = None
    for a1 in chain1:
        for a2 in chain2:
            if a1.i == a2.i:
                lca = a1
                break
        if lca is not None:
            break

    return lca


def dependency_distance(token1, token2):
    """
    returns the number of edges between token1 and token2 in the
    dependency tree, via their lowest common ancestor.

    returns None if token1 and token2 have no common ancestor
    (e.g. they are in different sentences).
    """
    lca = find_lowest_common_ancestor(token1, token2)

    if lca is None:
        return None

    up_steps = 0
    current = token1
    while current.i != lca.i:
        up_steps = up_steps + 1
        current = current.head

    down_steps = 0
    current = token2
    while current.i != lca.i:
        down_steps = down_steps + 1
        current = current.head

    return up_steps + down_steps

def dependency_path(token1, token2):
    """
    builds a directed dependency path string from token1 to token2,
    via their lowest common ancestor.

    each step going up the tree is recorded as 'up_<dep>', each step
    going down is recorded as 'down_<dep>'.
    """
    lca = find_lowest_common_ancestor(token1, token2)

    up_parts = []
    current = token1
    while current.i != lca.i:
        up_parts.append('up_' + current.dep_)
        current = current.head

    down_parts = []
    current = token2
    while current.i != lca.i:
        down_parts.append('down_' + current.dep_)
        current = current.head
    down_parts.reverse()

    path_parts = up_parts + down_parts
    path_string = '_'.join(path_parts)

    if path_string == '':
        path_string = 'self'

    return path_string


def get_word_gender(token, fem_mod_indexes, masc_mod_indexes):
    """
    determines whether token is lexically or through a modifier
    masculine, feminine, both, or neither.

    returns 'masc', 'fem', 'both', or 'none'.
    """
    is_masc = False
    is_fem = False

    if token.lemma_ in masc_lex:
        is_masc = True
    if token.lemma_ in fem_lex:
        is_fem = True

    if masc_mod_indexes and token.i in masc_mod_indexes:
        is_masc = True
    if fem_mod_indexes and token.i in fem_mod_indexes:
        is_fem = True

    if is_masc and is_fem:
        return 'both'
    elif is_masc:
        return 'masc'
    elif is_fem:
        return 'fem'
    else:
        return 'none'


def find_gendered_words(spacy_doc, fem_mod_indexes, masc_mod_indexes, person_indexes, gender='any'):
    """
    finds all person noun tokens in spacy_doc that are gendered, either
    lexically (via fem_lex/masc_lex) or through a gendered modifier
    (fem_mod_indexes/masc_mod_indexes).

    param gender: 'any', 'masc', or 'fem'. if 'masc' or 'fem', only
    tokens matching that gender (or 'both') are returned.

    returns a list of token indexes.
    """
    gendered_indexes = []

    for token in spacy_doc:
        is_person_token = (token.pos_ in ('NOUN', 'PROPN') and is_person(token.lemma_)) or token.i in person_indexes
        if not is_person_token:
            continue

        word_gender = get_word_gender(token, fem_mod_indexes, masc_mod_indexes)

        if word_gender == 'none':
            continue

        if gender == 'any':
            gendered_indexes.append(token.i)
        elif gender == 'masc' and word_gender in ('masc', 'both'):
            gendered_indexes.append(token.i)
        elif gender == 'fem' and word_gender in ('fem', 'both'):
            gendered_indexes.append(token.i)

    return gendered_indexes

def modifier_gender(token, fem_mod_indexes, masc_mod_indexes):
    """
    checks whether token appears as a head in fem_mod_indexes or
    masc_mod_indexes, indicating it carries a gendered modifier
    (e.g. 'female doctor', 'woman warrior').

    returns 'masc', 'fem', 'both', or 'none'.
    """
    in_masc = masc_mod_indexes and token.i in masc_mod_indexes
    in_fem = fem_mod_indexes and token.i in fem_mod_indexes

    if in_masc and in_fem:
        return 'both'
    elif in_masc:
        return 'masc'
    elif in_fem:
        return 'fem'
    else:
        return 'none'

def closest_gendered_dep_path(token, spacy_doc, fem_mod_indexes, masc_mod_indexes, person_indexes, gender='any'):

    own_gender = get_word_gender(token, fem_mod_indexes, masc_mod_indexes)

    if gender == 'any' and own_gender != 'none':
        return 'is_cue'
    if gender == 'masc' and own_gender in ('masc', 'both'):
        return 'is_masc'
    if gender == 'fem' and own_gender in ('fem', 'both'):
        return 'is_fem'

    gendered_indexes = find_gendered_words(spacy_doc, fem_mod_indexes, masc_mod_indexes, person_indexes, gender)

    best_index = None
    best_distance = None

    for g_i in gendered_indexes:
        if g_i == token.i:
            continue
        other_token = spacy_doc[g_i]
        distance = dependency_distance(token, other_token)

        if distance is None:
            continue

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = g_i

    if best_index is None:
        return 'none'

    closest_token = spacy_doc[best_index]
    path = dependency_path(token, closest_token)
    return path
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


def longest_cluster_for_token(token_index, entity_dict):
    """
    finds the cluster (by index into entity_dict) containing token_index
    with the most mentions. ties are broken by total token length
    across all mentions.
    """
    cluster_spans_list = entity_dict['clusters_token_offsets']

    best_index = None
    best_mentions = -1
    best_length = -1

    for c in range(len(cluster_spans_list)):
        spans = cluster_spans_list[c]
        found = False
        for span in spans:
            if token_index >= span[0] and token_index <= span[1]:
                found = True
                break

        if not found:
            continue

        num_mentions = len(spans)
        total_length = 0
        for span in spans:
            total_length = total_length + (span[1] - span[0] + 1)

        if num_mentions > best_mentions:
            best_index = c
            best_mentions = num_mentions
            best_length = total_length
        elif num_mentions == best_mentions and total_length > best_length:
            best_index = c
            best_mentions = num_mentions
            best_length = total_length

    return best_index
    

def detect_gender(text, target_index=None):
    """
    detects and labels gender for every person noun in the text.
    coref clusters are used for referential gender enrichment,
    but token selection is independent of coref.
    """

    spacy_doc = parse_sent(text)
    tokens = [token.text for token in spacy_doc]
    entity_dict = model.predict(tokens, singletons=True)

    fem_mod_indexes, masc_mod_indexes, attr_indexes, person_indexes = syntactic_indexes(spacy_doc)

    force_indexes = {target_index} if target_index is not None else set()

    dependent_to_head = {}
    for token in spacy_doc:
        if token.dep_ == 'relcl':
            for child in token.children:
                if child.dep_ == 'attr':
                    dependent_to_head[child.i] = token.head.i
                    for conj in conj_chain(child):
                        dependent_to_head[conj.i] = token.head.i
        if token.dep_ == 'appos':
            dependent_to_head[token.i] = token.head.i
            for conj in conj_chain(token):
                dependent_to_head[conj.i] = token.head.i

    token_to_cluster = build_token_to_cluster(entity_dict, spacy_doc)

    results = []
    label_map = {}
    individual_labels = set()

    for token in spacy_doc:

        if token.i in dependent_to_head and token.i not in force_indexes:
            continue

        is_candidate = (
            (token.pos_ in ('NOUN', 'PROPN') and is_person(token.lemma_))
            or token.i in person_indexes
            or token.i in force_indexes
        )

        if not is_candidate:
            continue

        if token.i in token_to_cluster:
            cluster_words, _ = token_to_cluster[token.i]
        else:
            cluster_words = []

        label = gender_person_token(
            token, cluster_words,
            fem_mod_indexes, masc_mod_indexes,
            person_indexes, spacy_doc,
            force_indexes=force_indexes
        )
        if label is not None:
            update_label_map(label_map, token.i, label, individual_labels)

    seen_clusters = set()

    for cluster_words, cluster_spans in zip(
        entity_dict['clusters_token_text'],
        entity_dict['clusters_token_offsets']
    ):
        cluster_key = tuple(tuple(s) for s in cluster_spans)
        if cluster_key in seen_clusters:
            continue
        seen_clusters.add(cluster_key)

        cluster_result = gender_cluster(
            cluster_words, cluster_spans, spacy_doc,
            fem_mod_indexes, masc_mod_indexes, attr_indexes,
            dependent_to_head, person_indexes
        )

        if cluster_result is None:
            continue

        mentions = []
        for ref in cluster_spans:
            mentions.append(spacy_doc[ref[0]:ref[1]+1].text)
        entity_gender = cluster_result['entity_gender']

        results.append({'mentions': mentions, 'entity_gender': entity_gender})

        for ref in cluster_spans:
            head = spacy_doc[ref[0]:ref[1]+1].root
            if (head.pos_ in ('NOUN', 'PROPN') and is_person(head.lemma_)) or head.i in person_indexes:
                conjuncts = conj_chain(head)
                has_different = False
                for conj in conjuncts:
                    conj_label = gender_person_token(conj, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
                    head_label = gender_person_token(head, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
                    if conj_label != head_label:
                        has_different = True
                        break
                if len(conjuncts) > 0 and has_different:
                    update_label_map(label_map, head.i, gender_person_token(head, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc), individual_labels)
                    individual_labels.add(head.i)
                else:
                    if is_plural(head):
                        has_ambiguous_subj_conj = False
                        for (attr_i, subj_i, _, __) in attr_indexes:
                            if attr_i != head.i:
                                continue
                            subj_token = spacy_doc[subj_i]
                            for conj in conj_chain(subj_token):
                                conj_label = gender_person_token(conj, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
                                if not is_gendered(conj_label):
                                    has_ambiguous_subj_conj = True
                                    break
                            subj_label = gender_person_token(subj_token, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
                            if not is_gendered(subj_label):
                                has_ambiguous_subj_conj = True

                        head_label = gender_person_token(head, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
                        if has_ambiguous_subj_conj and not is_gendered(head_label):
                            update_label_map(label_map, head.i, 'ambiguous', individual_labels)
                        else:
                            update_label_map(label_map, head.i, entity_gender, individual_labels)
                    else:
                        update_label_map(label_map, head.i, entity_gender, individual_labels)

            for conj in conj_chain(head):
                if (conj.pos_ in ('NOUN', 'PROPN') and is_person(conj.lemma_)) or conj.i in person_indexes:
                    own_label = gender_person_token(conj, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
                    head_label = gender_person_token(head, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
                    if own_label != head_label:
                        update_label_map(label_map, conj.i, own_label, individual_labels)
                        individual_labels.add(conj.i)
                    else:
                        update_label_map(label_map, conj.i, entity_gender, individual_labels)

    results = update_cluster_gender(results, spacy_doc)

    for entity in results:
        entity_gender = entity['entity_gender']
        for mention in entity['mentions']:
            span = find_span(mention, spacy_doc)
            if span is None:
                continue
            head = span.root
            for (attr_i, subj_i, _, is_partitive) in attr_indexes:
                if not attr_matches_head(subj_i, head, spacy_doc):
                    continue
                attr_token = spacy_doc[attr_i]
                if attr_token.pos_ not in ('NOUN', 'PROPN') or not is_person(attr_token.lemma_) or not attr_token.i:
                    continue

                conj_partners = []
                for (other_attr_i, other_subj_i, _, __) in attr_indexes:
                    if other_attr_i == attr_i:
                        continue
                    if not attr_matches_head(other_subj_i, head, spacy_doc):
                        continue
                    for conj in conj_chain(attr_token):
                        if conj.i == other_attr_i:
                            conj_partners.append(spacy_doc[other_attr_i])

                subj_token = spacy_doc[subj_i]

                for (other_attr_i, other_subj_i, _, __) in attr_indexes:
                    if other_attr_i == subj_token.i:
                        subj_token = spacy_doc[other_subj_i]
                        break

                if attr_i in token_to_cluster:
                    cluster_words, _ = token_to_cluster[attr_i]
                elif subj_token.i in token_to_cluster:
                    cluster_words, _ = token_to_cluster[subj_token.i]
                else:
                    cluster_words = []

                subj_conjuncts = conj_chain(subj_token)
                attr_label = gender_person_token(attr_token, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
                subj_label = gender_person_token(subj_token, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)

                if is_partitive and is_gendered(attr_label):
                    subj_label_resolved = 'unambiguous_all'
                elif subj_label is not None:
                    subj_label_resolved = subj_label
                else:
                    subj_label_resolved = 'ambiguous'

                attr_is_plural = is_plural(attr_token)
                subj_is_plural = is_plural(subj_token)

                if not attr_is_plural and not subj_is_plural:

                    if len(conj_partners) == 0 and len(subj_conjuncts) == 0:
                        if is_gendered(attr_label):
                            update_label_map(label_map, attr_i, attr_label, individual_labels)
                            if not is_gendered(subj_label):
                                update_label_map(label_map, subj_token.i, subj_label_resolved, individual_labels)
                            else:
                                update_label_map(label_map, subj_token.i, subj_label, individual_labels)
                        elif is_gendered(subj_label):
                            update_label_map(label_map, subj_token.i, subj_label, individual_labels)
                            update_label_map(label_map, attr_i, subj_label, individual_labels)
                        else:
                            update_label_map(label_map, attr_i, 'ambiguous', individual_labels)
                            update_label_map(label_map, subj_token.i, 'ambiguous', individual_labels)

                    elif len(conj_partners) == 0 and len(subj_conjuncts) > 0:
                        if is_gendered(attr_label):
                            update_label_map(label_map, attr_i, attr_label, individual_labels)
                            update_label_map(label_map, subj_token.i, subj_label_resolved, individual_labels)
                            for conj in subj_conjuncts:
                                if (conj.pos_ in ('NOUN', 'PROPN') and is_person(conj.lemma_)) or conj.i in person_indexes:
                                    update_label_map(label_map, conj.i, subj_label_resolved, individual_labels)
                        else:
                            conj_labels = []
                            for conj in subj_conjuncts:
                                if (conj.pos_ in ('NOUN', 'PROPN') and is_person(conj.lemma_)) or conj.i in person_indexes:
                                    conj_label = gender_person_token(conj, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
                                    conj_labels.append((conj, conj_label))
                            all_labels = [subj_label]
                            for _, l in conj_labels:
                                all_labels.append(l)
                            group_label = resolve_from_labels(all_labels)
                            update_label_map(label_map, attr_i, group_label, individual_labels)
                            update_label_map(label_map, subj_token.i, subj_label if is_gendered(subj_label) else group_label, individual_labels)
                            for conj, conj_label in conj_labels:
                                update_label_map(label_map, conj.i, conj_label if is_gendered(conj_label) else group_label, individual_labels)

                    elif len(conj_partners) > 0 and len(subj_conjuncts) == 0:
                        if is_gendered(subj_label) and not subj_is_plural:
                            update_label_map(label_map, subj_token.i, subj_label, individual_labels)
                            update_label_map(label_map, attr_i, subj_label, individual_labels)
                            for partner in conj_partners:
                                update_label_map(label_map, partner.i, subj_label, individual_labels)
                        else:
                            partner_labels = []
                            for partner in conj_partners:
                                partner_label = gender_person_token(partner, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
                                partner_labels.append((partner, partner_label))
                            all_labels = [attr_label]
                            for _, l in partner_labels:
                                all_labels.append(l)
                            group_label = resolve_from_labels(all_labels)
                            update_label_map(label_map, subj_token.i, subj_label_resolved if is_partitive else group_label, individual_labels)
                            update_label_map(label_map, attr_i, attr_label if is_gendered(attr_label) else group_label, individual_labels)
                            for partner, partner_label in partner_labels:
                                update_label_map(label_map, partner.i, partner_label if is_gendered(partner_label) else group_label, individual_labels)

                    else:
                        if attr_i not in label_map:
                            update_label_map(label_map, attr_i, entity_gender, individual_labels)
                        elif len(entity['mentions']) != 1:
                            update_label_map(label_map, attr_i, entity_gender, individual_labels)

                else:

                    if len(conj_partners) == 0 and len(subj_conjuncts) == 0:
                        if is_gendered(attr_label):
                            update_label_map(label_map, attr_i, attr_label, individual_labels)
                            if not is_gendered(subj_label):
                                update_label_map(label_map, subj_token.i, subj_label_resolved, individual_labels)
                            else:
                                update_label_map(label_map, subj_token.i, subj_label, individual_labels)
                        elif is_gendered(subj_label):
                            update_label_map(label_map, subj_token.i, subj_label, individual_labels)
                            update_label_map(label_map, attr_i, subj_label, individual_labels)
                        else:
                            update_label_map(label_map, attr_i, 'ambiguous', individual_labels)
                            update_label_map(label_map, subj_token.i, 'ambiguous', individual_labels)

                    elif len(conj_partners) == 0 and len(subj_conjuncts) > 0:
                        if is_gendered(attr_label):
                            update_label_map(label_map, attr_i, attr_label, individual_labels)
                            update_label_map(label_map, subj_token.i, subj_label_resolved, individual_labels)
                            for conj in subj_conjuncts:
                                if (conj.pos_ in ('NOUN', 'PROPN') and is_person(conj.lemma_)) or conj.i in person_indexes:
                                    update_label_map(label_map, conj.i, subj_label_resolved, individual_labels)
                        else:
                            conj_labels = []
                            for conj in subj_conjuncts:
                                if (conj.pos_ in ('NOUN', 'PROPN') and is_person(conj.lemma_)) or conj.i in person_indexes:
                                    conj_label = gender_person_token(conj, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
                                    conj_labels.append((conj, conj_label))
                            all_labels = [subj_label]
                            for _, l in conj_labels:
                                all_labels.append(l)
                            group_label = resolve_from_labels(all_labels)
                            update_label_map(label_map, attr_i, group_label, individual_labels)
                            update_label_map(label_map, subj_token.i, subj_label if is_gendered(subj_label) else group_label, individual_labels)
                            for conj, conj_label in conj_labels:
                                update_label_map(label_map, conj.i, conj_label if is_gendered(conj_label) else group_label, individual_labels)

                    elif len(conj_partners) > 0 and len(subj_conjuncts) == 0:
                        if is_gendered(subj_label) and not subj_is_plural:
                            update_label_map(label_map, subj_token.i, subj_label, individual_labels)
                            update_label_map(label_map, attr_i, subj_label, individual_labels)
                            for partner in conj_partners:
                                update_label_map(label_map, partner.i, subj_label, individual_labels)
                        else:
                            partner_labels = []
                            for partner in conj_partners:
                                partner_label = gender_person_token(partner, cluster_words, fem_mod_indexes, masc_mod_indexes, person_indexes, spacy_doc)
                                partner_labels.append((partner, partner_label))
                            all_labels = [attr_label]
                            for _, l in partner_labels:
                                all_labels.append(l)
                            group_label = resolve_from_labels(all_labels)
                            update_label_map(label_map, subj_token.i, subj_label_resolved if is_partitive else group_label, individual_labels)
                            update_label_map(label_map, attr_i, attr_label if is_gendered(attr_label) else group_label, individual_labels)
                            for partner, partner_label in partner_labels:
                                update_label_map(label_map, partner.i, partner_label if is_gendered(partner_label) else group_label, individual_labels)

                    else:
                        if attr_i not in label_map:
                            update_label_map(label_map, attr_i, entity_gender, individual_labels)
                        elif len(entity['mentions']) != 1:
                            update_label_map(label_map, attr_i, entity_gender, individual_labels)

    labeled_text_output = labeled_text(spacy_doc, label_map)
    token_labels = [
        label_map[token.i] if token.i in label_map else '_'
        for token in spacy_doc
    ]

    return results, labeled_text_output, token_labels