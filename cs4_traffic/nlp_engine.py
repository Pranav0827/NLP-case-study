"""
Case Study 4: NLP and Location Named Entity Recognition (NER) Engine
Specialized for informal Pune Traffic Complaints.
Key capabilities:
  1. Robust Location NER Engine:
     - Handles spelling errors, missing words, all-lowercase, abbreviations
     - Extracts single or multiple locations with character spans
     - Normalizes to canonical Pune landmarks and geocodes coordinates (lat, lon)
     - Interactive HTML annotation renderer
  2. Text Preprocessing & Typo Correction
  3. Multi-Model 6-Category Traffic Classifier
  4. Multi-Problem & Secondary Category Detector
  5. NER Evaluation Metrics (Precision, Recall, F1, Exact Match, Partial Match)
"""

import re
import json
import math
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

# Complete Pune Gazetteer with Canonical Names, Aliases, Coordinates, and Zones
PUNE_GAZETTEER = {
    "Swargate": {
        "coords": [18.5018, 73.8585],
        "zone": "Central Pune",
        "aliases": ["swargate", "swargat", "swargte", "swargat chowk", "swargate bus stand", "swargate junction", "swargate flyover", "swargate depot"]
    },
    "Shivaji Nagar": {
        "coords": [18.5314, 73.8446],
        "zone": "Central Pune",
        "aliases": ["shivaji nagar", "shivajinagar", "shivajinagr", "shivajinagar chowk", "shimla office chowk", "shivaji nagar station", "coep junction", "sancheti hospital chowk"]
    },
    "Hinjawadi": {
        "coords": [18.5913, 73.7389],
        "zone": "West Pune (IT Hub)",
        "aliases": ["hinjawadi", "hinjewadi", "hinjewdi", "hinjawadi phase 1", "hinjawadi phase 2", "hinjawadi phase 3", "hinjawadi ph 1", "hinjawadi ph 2", "hinjawadi ph 3", "hinjewadi flyover", "marunji chowk"]
    },
    "Katraj": {
        "coords": [18.4529, 73.8553],
        "zone": "South Pune",
        "aliases": ["katraj", "katraj ghat", "katraj ghaat", "katraj chowk", "katraj tunnel", "katraj bypass", "katraj zoo", "katraj bus stand"]
    },
    "Kothrud": {
        "coords": [18.5074, 73.8077],
        "zone": "West Pune",
        "aliases": ["kothrud", "kothrud stand", "kothrud depot", "paud road kothrud", "karishma chowk", "kothrud chowk", "vanaz corner", "ideal colony kothrud"]
    },
    "Deccan Gymkhana": {
        "coords": [18.5167, 73.8415],
        "zone": "Central Pune",
        "aliases": ["deccan gymkhana", "deccan", "decan", "deccan bus stand", "z-bridge deccan", "goodluck chowk", "garware bridge", "deccan corner"]
    },
    "Nal Stop": {
        "coords": [18.5089, 73.8295],
        "zone": "West Pune",
        "aliases": ["nal stop", "nal stop chowk", "nal stop bridge", "nalstop flyover", "nal stop signal", "nalstop", "nalstop chowk"]
    },
    "Chandani Chowk": {
        "coords": [18.5081, 73.7925],
        "zone": "West Pune",
        "aliases": ["chandani chowk", "chandni chowk", "chandnichowk", "chandani chowk flyover", "chandani chwk", "bavdhan chandani chowk", "kothrud chandani chowk"]
    },
    "Hadapsar": {
        "coords": [18.5018, 73.9268],
        "zone": "East Pune",
        "aliases": ["hadapsar", "hadapsar gadital", "hadapsr", "hadapsar flyover", "gadital hadapsar", "hadapsar bypass", "saswad road hadapsar", "hadapsar chowk"]
    },
    "Viman Nagar": {
        "coords": [18.5679, 73.9143],
        "zone": "North-East Pune",
        "aliases": ["viman nagar", "vimannagar", "viman nagar corner", "phoenix mall viman nagar", "viman nagr", "airport road viman nagar", "viman nagar junction"]
    },
    "Kalyani Nagar": {
        "coords": [18.5463, 73.9034],
        "zone": "North-East Pune",
        "aliases": ["kalyani nagar", "kalyaninagar", "kalyani nagr", "kalyani nagar bridge", "bishop school kalyani nagar", "kalyani nagar junction"]
    },
    "Baner": {
        "coords": [18.5590, 73.7868],
        "zone": "North-West Pune",
        "aliases": ["baner", "baner road", "baner phata", "pan card club road baner", "baner chowk", "baner highway", "balewadi high street baner"]
    },
    "Wakad": {
        "coords": [18.5987, 73.7684],
        "zone": "North-West Pune",
        "aliases": ["wakad", "wakd", "wakad bridge", "bhujbal chowk wakad", "wakad chowk", "wakad highway", "dange chowk wakad", "wakad flyover"]
    },
    "Aundh": {
        "coords": [18.5626, 73.8087],
        "zone": "North-West Pune",
        "aliases": ["aundh", "aundh road", "bremen chowk aundh", "parihar chowk aundh", "aundh chest hospital", "andh", "aundh bridge"]
    },
    "Magarpatta": {
        "coords": [18.5165, 73.9284],
        "zone": "East Pune",
        "aliases": ["magarpatta", "magarpatta city", "magarpatta south gate", "cybercity magarpatta", "magarpatta chowk", "seasons mall corner", "magarpatta flyover"]
    },
    "Senapati Bapat Road": {
        "coords": [18.5308, 73.8290],
        "zone": "Central Pune",
        "aliases": ["senapati bapat road", "sb road", "sb rd", "chatushrungi sb road", "symbiosis sb road", "sb road signal", "jw marriott sb road", "sb road junction"]
    },
    "FC Road": {
        "coords": [18.5236, 73.8418],
        "zone": "Central Pune",
        "aliases": ["fc road", "fc rd", "fergusson college road", "fc road starbucks", "goodluck to fc road", "tukaram paduka chowk", "fc road signal"]
    },
    "Karve Road": {
        "coords": [18.5050, 73.8250],
        "zone": "West Pune",
        "aliases": ["karve road", "karve rd", "karve road signal", "ayurved rasashala karve road", "paud phata karve rd", "karve statue"]
    },
    "Sinhagad Road": {
        "coords": [18.4850, 73.8300],
        "zone": "South Pune",
        "aliases": ["sinhagad road", "sinhgad road", "sinhagad rd", "rajaram bridge sinhagad rd", "vadgaon phata sinhagad rd", "manik baug sinhgad rd", "dhayari sinhagad road"]
    },
    "Warje": {
        "coords": [18.4810, 73.7980],
        "zone": "West Pune",
        "aliases": ["warje", "warje flyover", "warje bridge", "warje malwadi", "warje chowk", "mutha river bridge warje", "warje highway"]
    },
    "Vishrantwadi": {
        "coords": [18.5658, 73.8769],
        "zone": "North Pune",
        "aliases": ["vishrantwadi", "vishrantwadi chowk", "vishrantvadi", "tingre nagar vishrantwadi", "alandi road vishrantwadi", "vishrantwadi signal"]
    },
    "Yerwada": {
        "coords": [18.5529, 73.8828],
        "zone": "East Pune",
        "aliases": ["yerwada", "yerwada bridge", "yerwada gunjan chowk", "gunjan chowk yerwada", "shastri nagar yerwada", "yerwada jail chowk", "golf course yerwada"]
    },
    "Pune Station": {
        "coords": [18.5284, 73.8744],
        "zone": "Central Pune",
        "aliases": ["pune station", "pune railway station", "pune rly stn", "station chowk", "sasoon hospital junction", "station road pune"]
    },
    "Camp": {
        "coords": [18.5160, 73.8770],
        "zone": "Cantonment",
        "aliases": ["camp", "mg road camp", "camp mg rd", "east street camp", "camp poolgate", "cantonment camp", "moledina road camp"]
    },
    "Bhosari": {
        "coords": [18.6279, 73.8442],
        "zone": "PCMC / North",
        "aliases": ["bhosari", "bhosari midc", "bhosari flyover", "landewadi bhosari", "bhosari chowk", "telco bhosari"]
    },
    "Wagholi": {
        "coords": [18.5793, 73.9822],
        "zone": "East Corridor",
        "aliases": ["wagholi", "wagholi chowk", "kesnand phata wagholi", "waghli", "wagholi highway", "bakori phata wagholi", "wagholi bypass"]
    },
    "University Circle": {
        "coords": [18.5390, 73.8300],
        "zone": "Central-North",
        "aliases": ["university circle", "pune university circle", "sppu circle", "pune vidyapeeth chowk", "university flyover junction", "pashan university circle"]
    },
    "Pashan": {
        "coords": [18.5385, 73.7900],
        "zone": "West Pune",
        "aliases": ["pashan", "pashan circle", "pashan sus road", "pashan lake road", "abhimanshree pashan", "sus road pashan"]
    },
    "Fatima Nagar": {
        "coords": [18.5060, 73.8990],
        "zone": "East Pune",
        "aliases": ["fatima nagar", "fatimanagar", "fatima nagr", "fatima nagar junction", "solapur road fatima nagar", "fatima nagar bridge"]
    },
    "Khadki": {
        "coords": [18.5650, 73.8500],
        "zone": "North Pune",
        "aliases": ["khadki", "khadki bazaar", "khadki railway station", "khadki underpass", "bopodi khadki", "khadki cantonment"]
    }
}

CATEGORIES = [
    "Traffic Congestion & Bottlenecks",
    "Traffic Signal Breakdown & Timing",
    "Illegal & Obstructive Parking",
    "Reckless & Wrong-Side Driving",
    "Road Hazards, Potholes & Waterlogging",
    "Encroachment & Footpath Obstruction"
]

CATEGORY_ACTION_MAP = {
    "Traffic Congestion & Bottlenecks": {
        "action_unit": "Traffic Police - Beat Marshal & Corridor Clearance",
        "sla": "15-20 Mins",
        "equipment": "Patrol Bikes, Traffic Cones, Public Address System"
    },
    "Traffic Signal Breakdown & Timing": {
        "action_unit": "ATMS Signal Automation & Technical Maintenance Cell",
        "sla": "30 Mins",
        "equipment": "Signal Technician Van, Power Backup, Junction Controller"
    },
    "Illegal & Obstructive Parking": {
        "action_unit": "Pune Traffic Towing Crane & Wheel Clamping Squad",
        "sla": "20 Mins",
        "equipment": "Hydraulic Tow Truck, Wheel Clamps, E-Challan Handheld"
    },
    "Reckless & Wrong-Side Driving": {
        "action_unit": "E-Challan Flying Squad & Interceptor Team",
        "sla": "15 Mins",
        "equipment": "Speed Radar Guns, Body Cams, Interceptor Vehicle"
    },
    "Road Hazards, Potholes & Waterlogging": {
        "action_unit": "PMC Road Maintenance & Disaster Response Cell",
        "sla": "45 Mins",
        "equipment": "Dewatering Pumps, Cold Asphalt Patch Mix, Safety Barricades"
    },
    "Encroachment & Footpath Obstruction": {
        "action_unit": "Anti-Encroachment Dept & Ward Traffic Task Force",
        "sla": "60 Mins",
        "equipment": "Encroachment Clearance Truck, Police Bandobast"
    }
}

def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

class PuneLocationNER:
    """
    Named Entity Recognition (NER) specialized for informal, misspelled Pune traffic complaints.
    Combines:
      - Exact multi-token phrase matching
      - Contextual trigger pattern detection ('near...', 'between... and...', 'opp...', 'towards...')
      - Fuzzy Levenshtein / character n-gram matching for misspellings (e.g. hinjewdi -> Hinjawadi)
      - Entity normalization & geocoding to GPS coordinates
    """
    def __init__(self, gazetteer=PUNE_GAZETTEER, fuzzy_threshold=0.82):
        self.gazetteer = gazetteer
        self.fuzzy_threshold = fuzzy_threshold
        
        # Build flattened alias index mapping (lowercased alias) -> canonical name
        self.alias_to_canonical = {}
        for canon, data in self.gazetteer.items():
            self.alias_to_canonical[canon.lower()] = canon
            for alias in data.get("aliases", []):
                self.alias_to_canonical[alias.lower()] = canon
                
        # Sort aliases by length descending so longer phrases match first
        self.sorted_aliases = sorted(self.alias_to_canonical.keys(), key=lambda x: len(x), reverse=True)
        
        # Contextual prefix triggers common in traffic grievances
        self.triggers = [
            r'\b(?:near|nr|at|opp|opposite|towards|toward|from|between|around|under|on|across|outside|beside)\s+',
            r'\b(?:chowk|signal|flyover|bridge|junction|underpass|corner|stand|depot|phata|circle)\b'
        ]

    def extract_locations(self, text):
        """
        Extracts location entities from input text.
        Returns a list of dicts:
          [
            {
              "text": "katraj ghaat",
              "canonical": "Katraj",
              "start": 18,
              "end": 30,
              "confidence": 0.95,
              "coords": [18.4529, 73.8553],
              "zone": "South Pune",
              "method": "exact|fuzzy|contextual"
            }, ...
          ]
        """
        if not text or not isinstance(text, str):
            return []
            
        lower_text = text.lower()
        extracted = []
        occupied_spans = []  # To avoid overlapping duplicate extractions
        
        def is_overlapping(start, end):
            for s, e in occupied_spans:
                if max(start, s) < min(end, e):
                    return True
            return False

        # Phase 1: Direct substring search with sorted aliases (exact match)
        for alias in self.sorted_aliases:
            # Use word boundary search
            pattern = r'\b' + re.escape(alias) + r'\b'
            for match in re.finditer(pattern, lower_text):
                start, end = match.start(), match.end()
                if not is_overlapping(start, end):
                    canon = self.alias_to_canonical[alias]
                    meta = self.gazetteer.get(canon, {})
                    extracted.append({
                        "text": text[start:end],
                        "canonical": canon,
                        "start": start,
                        "end": end,
                        "confidence": 1.0,
                        "coords": meta.get("coords", [18.5204, 73.8567]),
                        "zone": meta.get("zone", "Pune"),
                        "method": "exact_gazetteer"
                    })
                    occupied_spans.append((start, end))

        # Phase 2: Token-level and n-gram fuzzy matching for misspelled names
        words = list(re.finditer(r'\b[a-zA-Z0-9_\-]+\b', lower_text))
        n_words = len(words)
        
        # Test 1-word, 2-word, and 3-word windows
        for n in range(3, 0, -1):
            for i in range(n_words - n + 1):
                start = words[i].start()
                end = words[i + n - 1].end()
                
                if is_overlapping(start, end):
                    continue
                    
                candidate_str = lower_text[start:end]
                # Skip trivial stop words
                if len(candidate_str) < 4:
                    continue
                    
                best_sim = 0.0
                best_alias = None
                
                for alias in self.sorted_aliases:
                    # Quick length check
                    if abs(len(alias) - len(candidate_str)) > 4:
                        continue
                    sim = string_similarity(candidate_str, alias)
                    if sim > best_sim:
                        best_sim = sim
                        best_alias = alias
                        
                if best_sim >= self.fuzzy_threshold and best_alias:
                    canon = self.alias_to_canonical[best_alias]
                    meta = self.gazetteer.get(canon, {})
                    extracted.append({
                        "text": text[start:end],
                        "canonical": canon,
                        "start": start,
                        "end": end,
                        "confidence": round(best_sim, 3),
                        "coords": meta.get("coords", [18.5204, 73.8567]),
                        "zone": meta.get("zone", "Pune"),
                        "method": "fuzzy_typo_match"
                    })
                    occupied_spans.append((start, end))

        # Sort extracted entities by appearance in text
        extracted.sort(key=lambda x: x["start"])
        return extracted

    def render_ner_html(self, text, entities, theme="light"):
        """
        Renders HTML markup highlighting detected location entities with tags.
        """
        if not entities:
            color = "#0f172a" if theme == "light" else "#f8fafc"
            return f"<div style='font-size:15px; color:{color}; line-height:1.6;'>{text}</div>"
            
        highlight_bg = "#dbeafe" if theme == "light" else "#1e3a8a"
        highlight_border = "#3b82f6"
        badge_bg = "#2563eb"
        badge_txt = "#ffffff"
        text_color = "#0f172a" if theme == "light" else "#f8fafc"
        
        chunks = []
        last_idx = 0
        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            
            # Text before entity
            if start > last_idx:
                chunks.append(text[last_idx:start])
                
            # Entity span with badge
            ent_text = text[start:end]
            canon = ent["canonical"]
            badge_html = (
                f"<mark style='background-color:{highlight_bg}; border:1px solid {highlight_border}; "
                f"border-radius:5px; padding:2px 6px; font-weight:600; color:{text_color}; display:inline-block; margin:0 2px;'>"
                f"{ent_text} "
                f"<span style='background-color:{badge_bg}; color:{badge_txt}; border-radius:3px; padding:1px 5px; font-size:10px; text-transform:uppercase;'>LOC: {canon}</span>"
                f"</mark>"
            )
            chunks.append(badge_html)
            last_idx = end
            
        if last_idx < len(text):
            chunks.append(text[last_idx:])
            
        return f"<div style='font-size:15px; line-height:1.8; color:{text_color};'>" + "".join(chunks) + "</div>"


def evaluate_ner_performance(df, ner_model=None):
    """
    Evaluates Location NER extraction against gold annotations in dataset.
    Computes:
      - Precision, Recall, F1-Score (Entity level)
      - Exact Match Accuracy
      - Partial / Fuzzy Match Rate
      - Robustness on informal text (typos, lowercase, multi-location)
    """
    if ner_model is None:
        ner_model = PuneLocationNER()
        
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    exact_matches = 0
    total_gold_entities = 0
    total_pred_entities = 0
    
    # Subgroup tracking for informal robustness
    subgroup_metrics = {
        "has_typos": {"gold": 0, "tp": 0, "pred": 0},
        "is_all_lowercase": {"gold": 0, "tp": 0, "pred": 0},
        "is_multi_location": {"gold": 0, "tp": 0, "pred": 0}
    }
    
    for _, row in df.iterrows():
        text = row["complaint_text"]
        gold_locs = json.loads(row["locations"]) if isinstance(row["locations"], str) else row["locations"]
        gold_canonicals = set(gold_locs)
        total_gold_entities += len(gold_canonicals)
        
        pred_entities = ner_model.extract_locations(text)
        pred_canonicals = set([e["canonical"] for e in pred_entities])
        total_pred_entities += len(pred_canonicals)
        
        # Exact set match check
        if gold_canonicals == pred_canonicals:
            exact_matches += 1
            
        tp = len(gold_canonicals.intersection(pred_canonicals))
        fp = len(pred_canonicals - gold_canonicals)
        fn = len(gold_canonicals - pred_canonicals)
        
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        
        # Subgroups
        for flag in ["has_typos", "is_all_lowercase", "is_multi_location"]:
            if row.get(flag, False):
                subgroup_metrics[flag]["gold"] += len(gold_canonicals)
                subgroup_metrics[flag]["pred"] += len(pred_canonicals)
                subgroup_metrics[flag]["tp"] += tp
                
    precision = true_positives / total_pred_entities if total_pred_entities > 0 else 0.0
    recall = true_positives / total_gold_entities if total_gold_entities > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match_rate = exact_matches / len(df) if len(df) > 0 else 0.0
    
    sub_results = {}
    for flag, d in subgroup_metrics.items():
        sub_p = d["tp"] / d["pred"] if d["pred"] > 0 else 0.0
        sub_r = d["tp"] / d["gold"] if d["gold"] > 0 else 0.0
        sub_f1 = (2 * sub_p * sub_r) / (sub_p + sub_r) if (sub_p + sub_r) > 0 else 0.0
        sub_results[flag] = {
            "precision": round(sub_p, 4),
            "recall": round(sub_r, 4),
            "f1": round(sub_f1, 4),
            "count": d["gold"]
        }
        
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "exact_match_rate": round(exact_match_rate, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_gold": total_gold_entities,
        "total_pred": total_pred_entities,
        "subgroup_robustness": sub_results
    }
