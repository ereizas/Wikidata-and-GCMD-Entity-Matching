from nltk import edit_distance
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from config import gemini_api_key, google_cloud_api_key
import time
import itertools
from collections import defaultdict
from embeddings_caching import *

# TODO: relabel GCMD entities that have phrases like "pertaining" and "measure"

def format_entity_no_path(entity):
    """
    Formats the entity term and definition into a single string with no path info

    :param entity : entity with a term and definition
    @return : single string in lowercase in the format: <term> - <definition>
    """
    return f"{entity["term"]} - {entity["definition"]}".lower()

def format_entity(entity):
    """
    Formats the entity term and definition into a single string

    :param entity: entity with a term and definition
    :return: single string in lowercase in the format: <term> - <definition>
    """
    return f"{f"{entity["path"]} | " if "path" in entity else ""}{entity["term"]} - {entity["definition"]}".lower()

def rank_by_n_gram(target:dict, candidates:dict, threshold=0, use_path=False):
    """
    Gets the ranking for each candidate based on cosine simalarity with the target
    :param target: GCMD entity to match in the format
    :param candidates: potential matches to target from Wikidata
    :param threshold: threshold for cosine similarity
    :return: vector for the target, vectors for the candidates
    """
    target = format_entity_no_path(target) if not use_path else format_entity(target)
    ids = candidates.keys()
    candidate_texts = [format_entity_no_path(candidates[uuid]) for uuid in candidates]
    texts = [target] + candidate_texts
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform(texts)
    target_vector = tfidf_matrix[0]
    candidate_vectors = tfidf_matrix[1:]
    similarities = ""
    if 0 not in candidate_vectors.shape:
        similarities = cosine_similarity(target_vector, candidate_vectors).flatten()
        similarities = zip(ids, similarities)
        similarities = [(ent, score) for ent,score in similarities if score>=threshold]
    return sorted(similarities, key=lambda x: x[1], reverse=True) if type(similarities)!=str else []

def rank_by_edit_dist(target:str, wikidata_search_res:dict, threshold=0.52):
    """
    Gets the ranking for each candidate based on edit distance from target

    :param target: GCMD entity to match
    :param wikidata_search_res: WikiData search results for the target
    :param threshold: threshold for edit distance
    :param wikidata_search_res: search results on WikiData for the target GCMD entity
    """
    ranking = []
    for res in wikidata_search_res:
        score = edit_distance(target.upper(),wikidata_search_res[res]["term"].upper())/len(target)
        if score<threshold:
            ranking.append((res,score))
    return sorted(ranking, key=lambda x: x[1])

def chunked_iterable(iterable, size):
    """
    Get generator object for iterable in batches

    :param iterable: the iterable to batch
    :param size: size of the batch
    """
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, size))
        if not batch:
            break
        yield batch

def build_batch_payload(batch_uuids, gcmd_ents, wikidata_search_res, use_path=False):
    """
    Build the batched payload to send to the LLM API

    :param batch_uuids: a batch of uuids
    :param gcmd_ents: entities from GCMD
    :param wikidata_search_res: GCMD uuid mapped to at most 10 top search results in WikiData
    :return: prompt for the LLM
    """
    task_text = ""
    for uuid in batch_uuids:
        if wikidata_search_res[uuid]:
            ent = gcmd_ents[uuid]
            hierarchy = (" > ".join(ent.get("path", [])) if ent.get("path") else "N/A") if use_path else None
            hierarchy_str = f"hierarchy='{hierarchy}', " if use_path else ""
            task_text += (
                "TASK_START\n"
                f"UUID: {uuid} "
                f"term='{ent['term']}', "
                f"definition='{ent['definition']}', "
                f"{hierarchy_str}"
                f"candidates={json.dumps(wikidata_search_res[uuid], ensure_ascii=False)}\nTASK_END\n\n"
            )
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "You are an expert at matching GCMD entities to Wikidata entities.\n"
                            "Each GCMD target entity has a UUID, term, definition, and keyword hierarchy.\n"
                            "Do NOT alter, modify, truncate, or rewrite UUIDs.\n"
                            "Rank candidates based on semantic equivalence, considering both the term and definition for each and every GCMD entity.\n"
                            "Use the keyword hierarchy to resolve ambiguities if given.\n"
                            "Exclude candidates that do not reperesent the same variable, measurement, concept or context (e.g. GCMD 'tractor hardware' should not be matched with a general definition for 'hardware').\n"
                            "Respond ONLY in valid JSON with this schema:\n\n"
                            "{ \"results\": { \"<uuid>\": [\"<candidate_id>\", ...], ... } }\n\n"
                            "If no candidates are valid, return an empty list for that uuid.\n\n"
                            f"Tasks:\n{task_text}"
                        )
                    }
                ]
            }
        ],
        "generation_config": {
            "response_mime_type": "application/json"
        }
    }

def rate_limited_post(url, payload, last_call_time, min_interval=5.0):
    """
    Send POST request while adhering to rate limit

    :param url: link
    :param payload: data to send
    :param last_call_time: last time the API was queried
    :param min_interval: minimum amount of time until the next request
    """
    # min_interval = 4–5s keeps ~12–15 req/min safe
    now = time.time()
    elapsed = now - last_call_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    resp = requests.post(url, json=payload)
    return resp, time.time()

API_URL = f"https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash-lite:generateContent?key={gemini_api_key}"

# TODO: add path to LLM API query
def process_batches(gcmd_ents, wikidata_search_res, num_samples, batch_size=20, use_path=False):
    """
    Query the LLM to match each target in the batch

    :param gcmd_ents: entities from GCMD
    :param wikidata_search_res: GCMD uuid mapped to at most 10 top search results in WikiData
    :param num_samples: number of labeled samples
    :param batch_size: size of the batch
    """
    last_call = 0
    results = {}
    for batch in chunked_iterable(list(gcmd_ents.keys())[:num_samples], batch_size):
        payload = build_batch_payload(batch, gcmd_ents, wikidata_search_res, use_path=use_path)
        resp, last_call = rate_limited_post(API_URL, payload, last_call)

        try:
            data = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(content)
            results.update(parsed["results"])
        except Exception as e:
            print("Error parsing response:", e, resp.text)
    return results

def build_unique_text_reprs(gcmd_ents:dict, wikidata_search_res:dict, limit, use_path=False):
    """
    Build unique text representations and map back to sources

    :param gcmd_ents: entities from GCMD
    :param wikidata_search_res: GCMD uuid mapped to at most 10 top search results in WikiData
    :return: mapping of text to sources, set of all unique texts
    """
    text_sources = defaultdict(list)  # maps text -> list of ("gcmd"|"candidate", uuid, candidate_id)
    all_texts = set()
    items = gcmd_ents.items() if limit is None else list(gcmd_ents.items())[:limit]
    for uuid, gcmd_entity in items:
        # GCMD entity text
        gcmd_text = format_entity_no_path(gcmd_entity) if not use_path else format_entity(gcmd_entity)
        all_texts.add(gcmd_text)
        text_sources[gcmd_text].append(("gcmd", uuid))

        # Candidates
        for cand_uuid, canditate_entity in wikidata_search_res[uuid].items():
            c_text = format_entity_no_path(canditate_entity)
            all_texts.add(c_text)
            text_sources[c_text].append(("candidate", uuid, cand_uuid))
    return text_sources, all_texts

def rank_by_embedding(target_embedding, candidate_embeddings, candidate_ids, threshold=0.685):
    """
    Rank candidates based on cosine similarity with the target embedding

    :param target_embedding: embedding vector for the target
    :param candidate_embeddings: list of embedding vectors for candidates
    :param candidate_ids: list of candidate ids corresponding to embeddings
    :param threshold: threshold for cosine similarity
    :return: sorted list of (candidate_id, similarity_score) tuples
    """
    if not candidate_embeddings:
        return []
    similarities = cosine_similarity([target_embedding], candidate_embeddings).flatten()
    similarities = zip(candidate_ids, similarities)
    similarities = [(ent, score) for ent,score in similarities if score>=threshold]
    return sorted(similarities, key=lambda x: x[1], reverse=True)

def get_dup_ents(gcmd_ents, ground_truth):
    dups = {}
    for uuid in gcmd_ents:
        txt = f"{gcmd_ents[uuid]["term"]} - {gcmd_ents[uuid]["definition"]}"
        dup_get = dups.get(txt, [])
        if not ground_truth[uuid] and (dup_get and not ground_truth[dup_get[0]]):
            continue
        dup_get.append(uuid)
        dups[txt]=dup_get
    dups = {txt: uuids for txt, uuids in dups.items() if len(uuids)>=2}
    with open("dups.json", "w") as file:
        json.dump(dups, file)

# TODO: try Mistral with the update of checking
def update_stats(top_candidate:str|None, ground_truth:list, stats:dict):
    """Update the statistics needed to calculate accuracy, recall, and precision
    :param rank: list of ids ranked by how well they match the target
    :param ground_truth: list of valid matches
    :param stats: dictionary of statistics (e.g. true positive, false negative)
    """
    if top_candidate in ground_truth:
        stats["tp"]+=1
    elif top_candidate and top_candidate not in ground_truth:
        stats["fp"]+=1
    elif not top_candidate and ground_truth==[""]:
        stats["tn"]+=1
    elif not top_candidate and ground_truth!=[""]:
        stats["fn"]+=1

def print_performance(method_name:str, stats:dict):
    """
    Prints the performance metrics for a method
    :param method_name: string name of the method
    :param stats: dictionary of statistics (e.g. true positive, false negative)
    """
    print(f"Accuracy of {method_name}: {(stats["tp"]+stats["tn"])/float(stats["tp"]+stats["fp"]+stats["tn"]+stats["fn"])}")
    prec = stats["tp"]/float(stats["tp"]+stats["fp"]) if stats["tp"] or stats["fp"] else 0
    print(f"Precision of {method_name}: {prec}")
    recall = stats["tp"]/float(stats["tp"]+stats["fn"]) if stats["tp"] or stats["fn"] else 0
    print(f"Recall of {method_name}: {recall}")
    print(f"F1 score of {method_name}: {2*prec*recall/(prec+recall) if prec or recall else 0}")

if __name__=="__main__":
    file = open("gcmd_ents.json","r")
    gcmd_ents = json.load(file)
    file.close()
    file = open("search_res.json","r")
    wikidata_search_res = json.load(file)
    file.close()
    file = open("ground_truth.json","r")
    ground_truth = json.load(file)
    file.close()
    # adjust as needed
    USE_PATH = True
    LABELED_SAMPLES = 475
    """llm_stats = {"tp":0, "fp":0, "tn":0, "fn":0}
    llm_outputs = process_batches(gcmd_ents, wikidata_search_res, LABELED_SAMPLES, use_path=USE_PATH)
    for uuid in llm_outputs:
        ground_truth_matches = ground_truth[uuid].split(",")
        print(f"UUID: {uuid} Ranking: {llm_outputs[uuid]}")
        update_stats(llm_outputs[uuid][0] if llm_outputs[uuid] else None, ground_truth_matches, llm_stats)
    for ind in llm_stats:
        print(f"LLM {ind}: {llm_stats[ind]}")
    print("")
    print_performance("LLM", llm_stats)"""
    # embedding
    # init_db()
    """text_sources, all_texts = build_unique_text_reprs(gcmd_ents, wikidata_search_res, LABELED_SAMPLES, use_path=USE_PATH)
    all_texts = list(all_texts)
    embeddings = batch_embeddings_with_cache(all_texts, api_key=google_cloud_api_key, db_path="embeddings_cache_no_path.db" if not USE_PATH else "embeddings_cache.db")
    text_to_emb = dict(zip(list(all_texts), embeddings))
    gcmd_embeddings = {}
    candidate_embeddings = defaultdict(dict)
    for text, usages in text_sources.items():
        for info in usages:
            if info[0] == "gcmd":
                gcmd_embeddings[info[1]] = text_to_emb[text]
            elif info[0] == "candidate":
                candidate_embeddings[info[1]][info[2]] = text_to_emb[text]"""
    
    edit_dist_stats = {"tp":0, "fp":0, "tn":0, "fn":0}
    n_gram_stats = {"tp":0, "fp":0, "tn":0, "fn":0}
    
    embedding_stats = {"tp":0, "fp":0, "tn":0, "fn":0}
    num_samples = 0
    for uuid in gcmd_ents:
        if wikidata_search_res[uuid]:
            ground_truth_matches = ground_truth[uuid].split(",")
            """edit_dist_rank = [item[0] for item in rank_by_edit_dist(gcmd_ents[uuid]["term"], wikidata_search_res[uuid])]
            update_stats(edit_dist_rank[0] if edit_dist_rank else None, ground_truth_matches, edit_dist_stats)"""
            """n_gram_rank = [item[0] for item in rank_by_n_gram(gcmd_ents[uuid],wikidata_search_res[uuid],use_path=USE_PATH)]
            update_stats(n_gram_rank[0] if n_gram_rank else None, ground_truth_matches, n_gram_stats)"""
            """embedding_rank = [
                item[0] for item in rank_by_embedding(gcmd_embeddings[uuid],
                list(candidate_embeddings[uuid].values()),
                list(candidate_embeddings[uuid].keys()))
            ]
            update_stats(embedding_rank[0] if embedding_rank else None, ground_truth_matches, embedding_stats)"""
        num_samples+=1
        if num_samples==LABELED_SAMPLES:
            break
    """for ind in edit_dist_stats:
        print(f"Edit dist {ind}: {edit_dist_stats[ind]}")
    print("")
    print_performance("edit distance", edit_dist_stats)
    print("")"""
    """for ind in n_gram_stats:
        print(f"N gram {ind}: {n_gram_stats[ind]}")
    print("")
    print_performance("n gram", n_gram_stats)"""
    """for ind in embedding_stats:
        print(f"Embedding {ind}: {embedding_stats[ind]}")
    print("")
    print_performance("embedding", embedding_stats)
    print("")"""

    # Test if path info helps best method with matching
    uuids = set()
    dups = {}
    with open("dups.json") as file:
        dups = json.load(file)
    for txt in dups:
        for uuid in dups[txt]:
            uuids.add(uuid)
    
    gcmd_ents = {uuid:ent for uuid, ent in gcmd_ents.items() if uuid in uuids}
    wikidata_search_res = {uuid:ent for uuid, ent in wikidata_search_res.items() if uuid in uuids}
    dup_llm_stats = {"tp":0, "fp":0, "tn":0, "fn":0}
    for uuid in gcmd_ents:
        print(f"{uuid}: {ground_truth[uuid]}")
    exit()
    llm_outputs = process_batches(gcmd_ents, wikidata_search_res, len(gcmd_ents.keys()), batch_size=15, use_path=USE_PATH)
    for uuid in llm_outputs:
        ground_truth_matches = ground_truth[uuid].split(",")
        print(f"UUID: {uuid} Ranking: {llm_outputs[uuid]}")
        update_stats(llm_outputs[uuid][0] if llm_outputs[uuid] else None, ground_truth_matches, dup_llm_stats)
    for ind in dup_llm_stats:
        print(f"LLM {ind}: {dup_llm_stats[ind]}")
    print("")
    print_performance("LLM", dup_llm_stats)