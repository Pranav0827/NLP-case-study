"""
Case Study 4: Pune Traffic Management Complaint Dataset Generator.
Synthesizes realistic, informal Pune traffic complaints with:
  - 6 Core Categories for Traffic Management
  - Informal text nuances: spelling mistakes, missing words, all-lowercase text
  - Single and multiple location mentions per complaint
  - Ground-truth location spans and canonical entity labels for NER evaluation
  - Standard reproducible random seed
"""

import os
import json
import random
import pandas as pd

random.seed(42)

# Canonical Pune locations with coordinates, zones, and informal misspellings / aliases
PUNE_LOCATIONS = {
    "Swargate": {
        "coords": [18.5018, 73.8585],
        "zone": "Central Pune",
        "aliases": ["swargate", "swargat", "swargte", "swargat chowk", "swargate bus stand", "swargate junction", "swargate flyover"]
    },
    "Shivaji Nagar": {
        "coords": [18.5314, 73.8446],
        "zone": "Central Pune",
        "aliases": ["shivaji nagar", "shivajinagar", "shivajinagr", "shivajinagar chowk", "shimla office chowk", "shivaji nagar station", "coep junction"]
    },
    "Hinjawadi": {
        "coords": [18.5913, 73.7389],
        "zone": "West Pune (IT Hub)",
        "aliases": ["hinjawadi", "hinjewadi", "hinjewdi", "hinjawadi phase 1", "hinjawadi ph 1", "hinjawadi ph 2", "hinjawadi ph 3", "hinjewadi flyover", "marunji chowk"]
    },
    "Katraj": {
        "coords": [18.4529, 73.8553],
        "zone": "South Pune",
        "aliases": ["katraj", "katraj ghat", "katraj ghaat", "katraj chowk", "katraj tunnel", "katraj bypass", "katraj zoo"]
    },
    "Kothrud": {
        "coords": [18.5074, 73.8077],
        "zone": "West Pune",
        "aliases": ["kothrud", "kothrud stand", "kothrud depot", "paud road kothrud", "karishma chowk", "kothrud chowk", "vanaz corner"]
    },
    "Deccan Gymkhana": {
        "coords": [18.5167, 73.8415],
        "zone": "Central Pune",
        "aliases": ["deccan gymkhana", "deccan", "decan", "deccan bus stand", "z-bridge deccan", "goodluck chowk", "garware bridge"]
    },
    "Nal Stop": {
        "coords": [18.5089, 73.8295],
        "zone": "West Pune",
        "aliases": ["nal stop", "nal stop chowk", "nal stop bridge", "nalstop flyover", "nal stop signal", "nalstop"]
    },
    "Chandani Chowk": {
        "coords": [18.5081, 73.7925],
        "zone": "West Pune",
        "aliases": ["chandani chowk", "chandni chowk", "chandnichowk", "chandani chowk flyover", "chandani chwk", "bavdhan chandani chowk"]
    },
    "Hadapsar": {
        "coords": [18.5018, 73.9268],
        "zone": "East Pune",
        "aliases": ["hadapsar", "hadapsar gadital", "hadapsr", "hadapsar flyover", "gadital hadapsar", "hadapsar bypass", "saswad road hadapsar"]
    },
    "Viman Nagar": {
        "coords": [18.5679, 73.9143],
        "zone": "North-East Pune",
        "aliases": ["viman nagar", "vimannagar", "viman nagar corner", "phoenix mall viman nagar", "viman nagr", "airport road viman nagar"]
    },
    "Kalyani Nagar": {
        "coords": [18.5463, 73.9034],
        "zone": "North-East Pune",
        "aliases": ["kalyani nagar", "kalyaninagar", "kalyani nagr", "kalyani nagar bridge", "bishop school kalyani nagar"]
    },
    "Baner": {
        "coords": [18.5590, 73.7868],
        "zone": "North-West Pune",
        "aliases": ["baner", "baner road", "baner phata", "pan card club road baner", "baner chowk", "baner highway"]
    },
    "Wakad": {
        "coords": [18.5987, 73.7684],
        "zone": "North-West Pune",
        "aliases": ["wakad", "wakd", "wakad bridge", "bhujbal chowk wakad", "wakad chowk", "wakad highway", "dange chowk wakad"]
    },
    "Aundh": {
        "coords": [18.5626, 73.8087],
        "zone": "North-West Pune",
        "aliases": ["aundh", "aundh road", "bremen chowk aundh", "parihar chowk aundh", "aundh chest hospital", "andh"]
    },
    "Magarpatta": {
        "coords": [18.5165, 73.9284],
        "zone": "East Pune",
        "aliases": ["magarpatta", "magarpatta city", "magarpatta south gate", "cybercity magarpatta", "magarpatta chowk", "seasons mall corner"]
    },
    "Senapati Bapat Road": {
        "coords": [18.5308, 73.8290],
        "zone": "Central Pune",
        "aliases": ["senapati bapat road", "sb road", "sb rd", "chatushrungi sb road", "symbiosis sb road", "sb road signal", "jw marriott sb road"]
    },
    "FC Road": {
        "coords": [18.5236, 73.8418],
        "zone": "Central Pune",
        "aliases": ["fc road", "fc rd", "fergusson college road", "fc road starbucks", "goodluck to fc road", "tukaram paduka chowk"]
    },
    "Karve Road": {
        "coords": [18.5050, 73.8250],
        "zone": "West Pune",
        "aliases": ["karve road", "karve rd", "karve road signal", "ayurved rasashala karve road", "paud phata karve rd"]
    },
    "Sinhagad Road": {
        "coords": [18.4850, 73.8300],
        "zone": "South Pune",
        "aliases": ["sinhagad road", "sinhgad road", "sinhagad rd", "rajaram bridge sinhagad rd", "vadgaon phata sinhagad rd", "manik baug sinhgad rd"]
    },
    "Warje": {
        "coords": [18.4810, 73.7980],
        "zone": "West Pune",
        "aliases": ["warje", "warje flyover", "warje bridge", "warje malwadi", "warje chowk", "mutha river bridge warje"]
    },
    "Vishrantwadi": {
        "coords": [18.5658, 73.8769],
        "zone": "North Pune",
        "aliases": ["vishrantwadi", "vishrantwadi chowk", "vishrantvadi", "tingre nagar vishrantwadi", "alandi road vishrantwadi"]
    },
    "Yerwada": {
        "coords": [18.5529, 73.8828],
        "zone": "East Pune",
        "aliases": ["yerwada", "yerwada bridge", "yerwada gunjan chowk", "gunjan chowk yerwada", "shastri nagar yerwada", "yerwada jail chowk"]
    },
    "Pune Station": {
        "coords": [18.5284, 73.8744],
        "zone": "Central Pune",
        "aliases": ["pune station", "pune railway station", "pune rly stn", "station chowk", "sasoon hospital junction"]
    },
    "Camp": {
        "coords": [18.5160, 73.8770],
        "zone": "Cantonment",
        "aliases": ["camp", "mg road camp", "camp mg rd", "east street camp", "camp poolgate", "cantonment camp"]
    },
    "Bhosari": {
        "coords": [18.6279, 73.8442],
        "zone": "PCMC / North",
        "aliases": ["bhosari", "bhosari midc", "bhosari flyover", "landewadi bhosari", "bhosari chowk"]
    },
    "Wagholi": {
        "coords": [18.5793, 73.9822],
        "zone": "East Corridor",
        "aliases": ["wagholi", "wagholi chowk", "kesnand phata wagholi", "waghli", "wagholi highway", "bakori phata wagholi"]
    },
    "University Circle": {
        "coords": [18.5390, 73.8300],
        "zone": "Central-North",
        "aliases": ["university circle", "pune university circle", "sppu circle", "pune vidyapeeth chowk", "university flyover junction"]
    },
    "Pashan": {
        "coords": [18.5385, 73.7900],
        "zone": "West Pune",
        "aliases": ["pashan", "pashan circle", "pashan sus road", "pashan lake road", "abhimanshree pashan"]
    },
    "Fatima Nagar": {
        "coords": [18.5060, 73.8990],
        "zone": "East Pune",
        "aliases": ["fatima nagar", "fatimanagar", "fatima nagr", "fatima nagar junction", "solapur road fatima nagar"]
    },
    "Khadki": {
        "coords": [18.5650, 73.8500],
        "zone": "North Pune",
        "aliases": ["khadki", "khadki bazaar", "khadki railway station", "khadki underpass", "bopodi khadki"]
    }
}

# The 6 Specific Traffic Management Complaint Categories
CATEGORIES = [
    "Traffic Congestion & Bottlenecks",
    "Traffic Signal Breakdown & Timing",
    "Illegal & Obstructive Parking",
    "Reckless & Wrong-Side Driving",
    "Road Hazards, Potholes & Waterlogging",
    "Encroachment & Footpath Obstruction"
]

CATEGORY_ACTION_UNITS = {
    "Traffic Congestion & Bottlenecks": "Traffic Police - Beat Marshal & Corridor Clearance",
    "Traffic Signal Breakdown & Timing": "ATMS Signal Automation & Technical Maintenance Cell",
    "Illegal & Obstructive Parking": "Pune Traffic Towing Crane & Wheel Clamping Squad",
    "Reckless & Wrong-Side Driving": "E-Challan Flying Squad & Interceptor Team",
    "Road Hazards, Potholes & Waterlogging": "PMC Road Maintenance & Disaster Response Cell",
    "Encroachment & Footpath Obstruction": "Anti-Encroachment Dept & Ward Traffic Task Force"
}

# Distinct phrase templates per category
TEMPLATES = {
    "Traffic Congestion & Bottlenecks": [
        "huge bumper to bumper jam near {loc1} vehicles crawling from last 45 mins no cop",
        "massive bottleneck at {loc1} flyover traffic completely halted pl send traffic police asap",
        "extreme gridlock between {loc1} and {loc2} stuck for 1 hr straight emergency ambulance also caught",
        "traffic chocked from {loc1} towards {loc2} worst peak hour bottleneck pls clear",
        "heavy jam at {loc1} square road is narrow vehicles moving at snail pace",
        "terrible slow moving traffic near {loc1} chowk signals not helping clearance",
        "complete chaos at {loc1} underpass all vehicles blocking each other no lane discipline",
        "vehicle queue stretching over 2km from {loc1} to {loc2} need marshals urgently",
        "horrible traffic jam outside {loc1} everyday evening 6pm to 9pm pls deploy personnel",
        "heavy congestion near {loc1} junction office commuters stranded since 30 mins",
        "unbearable traffic mess around {loc1} commuters honking non stop no movement",
        "crawling traffic on main carriageway near {loc1} took 50 minutes to cross 500 meters",
        "terrible choke point created near {loc1} bottleneck during heavy evening rains",
        "standstill traffic situation from {loc1} right up to {loc2} junction",
        "severe peak hour congestion at {loc1} connecting road need urgent traffic warden"
    ],
    "Traffic Signal Breakdown & Timing": [
        "traffic signal dead near {loc1} all lights off since morning heavy chaos",
        "signal stuck on red timer at {loc1} chowk vehicles waiting 10 mins without green",
        "both sides green at {loc1} junction very dangerous close call accident avoided",
        "signal blinking yellow continuously at {loc1} crossroad commuters getting confused",
        "timer at {loc1} is only 15 seconds green for huge queue and 180 sec red please recalibrate",
        "power failure at {loc1} traffic light no cop to manage manual traffic flow",
        "pedestrian crossing signal broken near {loc1} public unable to cross busy road",
        "signal synchronization failed between {loc1} and {loc2} stopping cars at every 100 meters",
        "traffic light pole knocked down near {loc1} wires exposed and signal dark",
        "at {loc1} automated atms signal camera not detecting queue giving 10 sec green only",
        "traffic light timer countdown stuck at zero at {loc1} junction motorists confused",
        "no traffic signal functioning at major intersection near {loc1} high risk of collisions",
        "green light timing too short barely 4 cars pass at {loc1} signal please increase timer",
        "signal lights damaged after rainfall at {loc1} chowk please dispatch repair team",
        "erratic signal cycles between {loc1} and {loc2} corridor creating artificial congestion"
    ],
    "Illegal & Obstructive Parking": [
        "cars parked in double line opp {loc1} market blocking half carriage way send towing crane",
        "illegal auto rickshaw stand near {loc1} bus bay blocking whole left lane",
        "abandoned truck parked for 3 days near {loc1} junction causing severe choke point",
        "private tourist buses parked haphazardly along {loc1} road at night zero space for traffic",
        "suvs parked on no parking zone right under {loc1} flyover clamp them immediately",
        "shopkeepers and customers parking on road near {loc1} chowk footpath also taken over",
        "two wheelers parked diagonally on both sides of {loc1} narrow lane no ambulance can enter",
        "heavy vehicle parking prohibited but trucks loading unloading at {loc1} main road during peak hour",
        "valet parking of hotel taking up entire public lane near {loc1} please fine them",
        "cars double parked near {loc1} opposite hospital gate completely blocking emergency exit",
        "commercial delivery tempos parked haphazardly right at the corner of {loc1} turning",
        "unauthorized taxi parking encroaching bus stop near {loc1} buses stopping middle of road",
        "row of cars parked on yellow line near {loc1} causing 1 lane blockage all day",
        "school buses parked illegally along {loc1} service road blocking resident vehicles",
        "no parking sign completely ignored by luxury vehicles near {loc1} cafe cluster"
    ],
    "Reckless & Wrong-Side Driving": [
        "bikers driving wrong side in speed on {loc1} one way very dangerous almost hit pedestrian",
        "commercial delivery boys and autos jumping red signal at {loc1} chowk continuously",
        "rash driving by luxury car on {loc1} road doing 90kmph in school zone install speed camera",
        "triple seat bikers without helmet cutting lanes recklessly near {loc1} flyover",
        "heavy dumpers and water tankers speeding dangerously on {loc1} bypass at night",
        "vehicles taking illegal u-turn over broken divider near {loc1} causing collisions",
        "wrong side riding epidemic from {loc1} towards {loc2} to avoid taking 500m detour",
        "bus driver driving rashly overtaking from extreme left near {loc1} junction",
        "stunt biking and rash overtakes happening near {loc1} on weekend nights pl set up checkpost",
        "motorists ignoring zebra crossing and speeding past pedestrians at {loc1} signal",
        "high speed overtaking from wrong side on the blind curve near {loc1}",
        "two wheelers openly driving on footpath against oncoming traffic near {loc1} chowk",
        "speed demons racing late night near {loc1} corridor need speed breakers immediately",
        "rash auto drivers cutting sharp turns without indicator near {loc1} intersection",
        "heavy vehicles violating daytime entry ban and driving recklessly through {loc1} road"
    ],
    "Road Hazards, Potholes & Waterlogging": [
        "huge crater sized pothole near {loc1} chowk two wheeler riders skidding and falling",
        "severe water logging under {loc1} bridge after 30 mins rain knee deep water traffic stopped",
        "open drain manhole without lid near {loc1} on main road major disaster waiting to happen",
        "broken road divider with iron rebar protruding on {loc1} road damaged two car tires",
        "pipeline trench dug up by pmc near {loc1} left open with no caution barricade or light",
        "slippery oil spill on {loc1} flyover multiple bikes slipped this morning wash road immediately",
        "tree branch fallen down half blocking road near {loc1} vehicles swerving dangerously",
        "gravel and construction sand spilled across {loc1} junction causing vehicles to skid",
        "deep potholes near {loc1} causing vehicles to brake suddenly resulting in rear end crashes",
        "drainage water overflowing on road near {loc1} terrible stench and road submerged",
        "massive water accumulation at {loc1} underpass cars getting submerged in water",
        "sharp iron plate left on dug road near {loc1} bursting vehicle tyres",
        "road sinking and cave-in observed near {loc1} road curb please barricade immediately",
        "loose gravel left after patchwork near {loc1} two wheelers skidding while braking",
        "submerged open trench near {loc1} during waterlogging deadly safety risk for commuters"
    ],
    "Encroachment & Footpath Obstruction": [
        "illegal hawkers and food stalls occupying half the road near {loc1} commuters forced on road",
        "construction debris and cement bricks dumped right on {loc1} road lane blocking traffic",
        "tea stalls and pan shops putting tables chairs on footpath near {loc1} pedestrians in danger",
        "illegal political hoardings and bamboo scaffolding blocking view of signal at {loc1} chowk",
        "garage mechanics repairing vehicles on public road near {loc1} keeping tires on street",
        "vegetable market vendors spreading wares on {loc1} junction everyday causing total obstruction",
        "commercial shops extending display sheds onto road near {loc1} corner",
        "unauthorized taxi booth erected on footpath at {loc1} blocking pedestrian access",
        "building material gravel sand dumped on road at {loc1} for past one week zero clearance",
        "encroachment by street food carts near {loc1} creates bottlenecks every evening",
        "fruit juice vendors and tea tapris occupying turning lane near {loc1} chowk",
        "debris of broken concrete pipes left dumped on {loc1} side road for 20 days",
        "illegal pandal erected on main road near {loc1} blocking 2 lanes of vehicular movement",
        "footpath completely swallowed by illegal kiosks near {loc1} school zone",
        "shop display boards placed directly on active roadway near {loc1} market stretch"
    ]
}

SECONDARY_TEMPLATES = {
    "Illegal & Obstructive Parking": "and cars are also parked illegally double lane here",
    "Encroachment & Footpath Obstruction": "also street hawkers have encroached half the turning",
    "Traffic Signal Breakdown & Timing": "plus the traffic light is completely dead",
    "Road Hazards, Potholes & Waterlogging": "also deep potholes making it impossible to pass",
    "Reckless & Wrong-Side Driving": "along with bikes coming fast from wrong side",
    "Traffic Congestion & Bottlenecks": "causing massive 2km gridlock"
}

def inject_informal_noise(text, has_typos=True, make_lowercase=True, drop_words=True):
    res = text
    if make_lowercase:
        res = res.lower()
    
    typo_map = {
        "traffic": "trafic",
        "signal": "signl",
        "signals": "signls",
        "police": "polce",
        "junction": "juncshun",
        "pothole": "pothol",
        "potholes": "pothols",
        "flyover": "flyovr",
        "blocking": "bloking",
        "accident": "acident",
        "vehicles": "vehicals",
        "dangerously": "dangerosly",
        "pedestrian": "pedestrain",
        "waterlogging": "waterloging",
        "bottleneck": "botlneck",
        "congestion": "congeston",
        "parking": "parkng",
        "immediately": "immeditely",
        "broken": "brokn",
        "cleared": "cleard"
    }
    
    if has_typos:
        words = res.split()
        corrupted = []
        for w in words:
            clean_w = w.strip(".,!?:;")
            if clean_w in typo_map and random.random() < 0.65:
                w = w.replace(clean_w, typo_map[clean_w])
            corrupted.append(w)
        res = " ".join(corrupted)
        
    if drop_words:
        drop_candidates = {"the", "is", "at", "a", "an", "on", "in", "and", "please", "pls"}
        words = res.split()
        if len(words) > 8:
            words = [w for w in words if not (w in drop_candidates and random.random() < 0.35)]
            res = " ".join(words)
            
    return res

def generate_dataset(n_samples_per_category=105, output_csv="pune_traffic_complaints.csv"):
    records = []
    location_names = list(PUNE_LOCATIONS.keys())
    complaint_idx = 1
    
    for cat in CATEGORIES:
        templates = TEMPLATES[cat]
        for i in range(n_samples_per_category):
            tmpl = random.choice(templates)
            
            is_multi_loc = ("{loc2}" in tmpl) or (random.random() < 0.22)
            
            loc1_canonical = random.choice(location_names)
            loc1_alias = random.choice(PUNE_LOCATIONS[loc1_canonical]["aliases"])
            
            loc2_canonical = None
            loc2_alias = None
            if is_multi_loc:
                candidates = [l for l in location_names if l != loc1_canonical]
                loc2_canonical = random.choice(candidates)
                loc2_alias = random.choice(PUNE_LOCATIONS[loc2_canonical]["aliases"])
                
            if "{loc2}" in tmpl:
                text_draft = tmpl.format(loc1=loc1_alias, loc2=loc2_alias)
                extracted_locations = [loc1_canonical, loc2_canonical]
                extracted_aliases = [loc1_alias, loc2_alias]
            elif is_multi_loc and random.random() < 0.5:
                multi_connector = random.choice([
                    f" stretching all the way back to {loc2_alias}",
                    f" situation equally bad near {loc2_alias}",
                    f" and queue extending till {loc2_alias}"
                ])
                text_draft = tmpl.format(loc1=loc1_alias) + multi_connector
                extracted_locations = [loc1_canonical, loc2_canonical]
                extracted_aliases = [loc1_alias, loc2_alias]
            else:
                text_draft = tmpl.format(loc1=loc1_alias)
                extracted_locations = [loc1_canonical]
                extracted_aliases = [loc1_alias]
                
            secondary_cat = None
            has_multi_problem = False
            if random.random() < 0.18:
                other_cats = [c for c in CATEGORIES if c != cat]
                secondary_cat = random.choice(other_cats)
                sec_text = SECONDARY_TEMPLATES[secondary_cat]
                text_draft = f"{text_draft} {sec_text}"
                has_multi_problem = True
                
            has_typos = random.random() < 0.55
            make_lowercase = random.random() < 0.65
            drop_words = random.random() < 0.45
            
            final_text = inject_informal_noise(
                text_draft,
                has_typos=has_typos,
                make_lowercase=make_lowercase,
                drop_words=drop_words
            )
            
            high_sev_keywords = ["accident", "acident", "emergency", "ambulance", "waterloging", "water logging", "hospital", "collapsed", "crater", "crawling from last", "dead"]
            if any(k in final_text.lower() for k in high_sev_keywords):
                priority = "High"
            elif cat in ["Road Hazards, Potholes & Waterlogging", "Reckless & Wrong-Side Driving"]:
                priority = random.choice(["High", "High", "Medium"])
            elif cat in ["Traffic Congestion & Bottlenecks", "Traffic Signal Breakdown & Timing"]:
                priority = random.choice(["High", "Medium", "Medium"])
            else:
                priority = random.choice(["Medium", "Medium", "Low"])
                
            cid = f"PUN-TRF-{complaint_idx:04d}"
            complaint_idx += 1
            
            # Extract ground truth character spans for NER evaluation
            gold_spans = []
            for alias, canon in zip(extracted_aliases, extracted_locations):
                lower_text = final_text.lower()
                lower_alias = alias.lower()
                idx = lower_text.find(lower_alias)
                if idx != -1:
                    gold_spans.append({
                        "text": final_text[idx:idx+len(alias)],
                        "start": idx,
                        "end": idx + len(alias),
                        "label": "LOCATION",
                        "canonical": canon
                    })
                else:
                    first_word = alias.split()[0].lower()
                    idx = lower_text.find(first_word)
                    if idx != -1:
                        gold_spans.append({
                            "text": final_text[idx:idx+len(first_word)],
                            "start": idx,
                            "end": idx + len(first_word),
                            "label": "LOCATION",
                            "canonical": canon
                        })

            records.append({
                "complaint_id": cid,
                "complaint_text": final_text,
                "category": cat,
                "secondary_category": secondary_cat if has_multi_problem else "",
                "locations": json.dumps(extracted_locations),
                "primary_location": extracted_locations[0],
                "gold_spans": json.dumps(gold_spans),
                "priority": priority,
                "action_unit": CATEGORY_ACTION_UNITS[cat],
                "has_typos": has_typos,
                "is_all_lowercase": make_lowercase,
                "is_multi_location": len(extracted_locations) > 1,
                "is_multi_problem": has_multi_problem
            })
            
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df

if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "pune_traffic_complaints.csv")
    df = generate_dataset(n_samples_per_category=105, output_csv=out_file)
    print(f"Generated {len(df)} records saved to {out_file}")
    print("\nCategory Distribution:")
    print(df['category'].value_counts())
    print(f"\nTypos: {df['has_typos'].mean()*100:.1f}% | All-lowercase: {df['is_all_lowercase'].mean()*100:.1f}% | Multi-loc: {df['is_multi_location'].mean()*100:.1f}%")
