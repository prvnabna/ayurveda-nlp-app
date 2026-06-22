"""
NLP-03: Annotation Guideline Creation & Application  [UPGRADED v2]
===================================================================
Extended seed dictionaries (300+ herbs, 60+ diseases, full Panchakarma list).
Overlap-aware span deduplication with priority scoring.
Produces annotated spans in JSON compatible with spaCy, Label Studio, Doccano.

Deliverable: NLP, Documentation
"""

import re
import json
from pathlib import Path


# ------------------------------------------------------------------ #
#  Ayurveda Annotation Schema                                          #
# ------------------------------------------------------------------ #

ANNOTATION_GUIDELINES = {
    "version": "2.0",
    "domain":  "Ayurveda",
    "labels": {
        "HERB":        "Medicinal plant or herbal preparation (e.g., Ashwagandha, Neem)",
        "DISEASE":     "Disease, disorder, or ailment (e.g., Jwara, Prameha)",
        "DOSHA":       "Ayurvedic constitution element: Vata, Pitta, or Kapha",
        "PROCEDURE":   "Treatment or therapeutic procedure (e.g., Panchakarma, Vamana)",
        "BODY_PART":   "Anatomical structure or organ (e.g., Liver, Hridaya)",
        "INGREDIENT":  "Formulation ingredient or compound",
        "QUANTITY":    "Dosage or measurement (e.g., 3 grams, twice daily)",
        "SOURCE_REF":  "Reference to classical text or author (e.g., Charaka Samhita)",
        "PLANT_PART":  "Part of a plant used (e.g., root, leaf, bark)",
        "PROPERTY":    "Ayurvedic property (Rasa, Guna, Virya, Vipaka)",
        "FORMULATION": "Classical Ayurveda formulation (e.g., Triphala Churna, Dashamoola Kashaya)",
        "DIET":        "Dietary item or Pathya/Apathya mention",
    }
}

# ── Label priority (higher = preferred when spans overlap) ──────────
LABEL_PRIORITY = {
    "SOURCE_REF": 10, "FORMULATION": 9, "HERB": 8, "DISEASE": 8,
    "PROCEDURE": 7, "DOSHA": 7, "BODY_PART": 6, "PROPERTY": 6,
    "PLANT_PART": 5, "QUANTITY": 5, "INGREDIENT": 4, "DIET": 3,
}

# ── EXPANDED Entity seed dictionaries ──────────────────────────────
ENTITY_SEEDS = {

    "HERB": [
        # ── Adaptogens / Rasayana ──
        r"\b(ashwagandha|withania\s*somnifera|aswagandha|asagandha)\b",
        r"\b(brahmi|bacopa\s*monnieri|jalabrahmi)\b",
        r"\b(shatavari|asparagus\s*racemosus|satamuli)\b",
        r"\b(guduchi|tinospora\s*cordifolia|amrita|giloy)\b",
        r"\b(amalaki|emblica\s*officinalis|amla|indian\s*gooseberry)\b",
        r"\b(haritaki|terminalia\s*chebula|black\s*myrobalan)\b",
        r"\b(bibhitaki|terminalia\s*bellirica|bahera)\b",
        r"\b(triphala)\b",
        r"\b(shankhpushpi|convolvulus\s*pluricaulis)\b",
        r"\b(jatamansi|nardostachys\s*jatamansi|spikenard)\b",
        r"\b(bala|sida\s*cordifolia)\b",
        r"\b(yastimadhu|glycyrrhiza\s*glabra|licorice|mulethi)\b",
        # ── Anti-inflammatory / Immune ──
        r"\b(turmeric|curcuma\s*longa|haridra|haldi)\b",
        r"\b(neem|azadirachta\s*indica|nimba|margosa)\b",
        r"\b(tulsi|ocimum\s*sanctum|holy\s*basil)\b",
        r"\b(nirgundi|vitex\s*negundo)\b",
        r"\b(guggul|commiphora\s*mukul|guggulu)\b",
        r"\b(shallaki|boswellia\s*serrata)\b",
        r"\b(manjistha|rubia\s*cordifolia|indian\s*madder)\b",
        r"\b(sariva|hemidesmus\s*indicus|indian\s*sarsaparilla)\b",
        r"\b(kantakari|solanum\s*xanthocarpum)\b",
        r"\b(punarnava|boerhavia\s*diffusa|hogweed)\b",
        # ── Digestive ──
        r"\b(ginger|zingiber\s*officinale|shunthi|ardrak|ardraka)\b",
        r"\b(pippali|piper\s*longum|long\s*pepper)\b",
        r"\b(maricha|piper\s*nigrum|black\s*pepper)\b",
        r"\b(trikatu)\b",
        r"\b(vidanga|embelia\s*ribes)\b",
        r"\b(chitrak|plumbago\s*zeylanica|chitraka)\b",
        r"\b(musta|cyperus\s*rotundus|nagarmotha)\b",
        r"\b(ajwain|trachyspermum\s*ammi|carom\s*seeds?)\b",
        r"\b(hing|ferula\s*asafoetida|asafoetida)\b",
        r"\b(dadima|punica\s*granatum|pomegranate)\b",
        r"\b(bilwa|aegle\s*marmelos|bael)\b",
        # ── Liver / Blood ──
        r"\b(bhringraj|eclipta\s*alba|bhringaraja)\b",
        r"\b(kutki|picrorhiza\s*kurroa|katuki)\b",
        r"\b(kalmegh|andrographis\s*paniculata|kiratatikta)\b",
        r"\b(rohitaka|tecomella\s*undulata)\b",
        r"\b(kakmachi|solanum\s*nigrum)\b",
        # ── Nervous system ──
        r"\b(vacha|acorus\s*calamus|sweet\s*flag)\b",
        r"\b(kushtha|saussurea\s*lappa|costus)\b",
        r"\b(tagara|valeriana\s*wallichii|indian\s*valerian)\b",
        r"\b(kapikachhu|mucuna\s*pruriens|velvet\s*bean)\b",
        # ── Urinary / Renal ──
        r"\b(gokshura|tribulus\s*terrestris|gokhru)\b",
        r"\b(varuna|crataeva\s*nurvala)\b",
        r"\b(pashanabheda|bergenia\s*ligulata)\b",
        r"\b(shilajit|mineral\s*pitch|asphaltum)\b",
        # ── Women's health ──
        r"\b(lodhra|symplocos\s*racemosa)\b",
        r"\b(ashoka|saraca\s*indica|saraca\s*asoca)\b",
        r"\b(kumari|aloe\s*vera|ghritkumari)\b",
        # ── Respiratory ──
        r"\b(vasa|adhatoda\s*vasica|malabar\s*nut)\b",
        r"\b(kantakari|solanum\s*xanthocarpum)\b",
        r"\b(pushkarmool|inula\s*racemosa)\b",
        r"\b(haridra|curcuma\s*longa)\b",
        # ── Classical formulations ──
        r"\b(dashamoola|dasamoola)\b",
        r"\b(sapta\s*parna|alstonia\s*scholaris|saptaparna)\b",
        r"\b(devadaru|cedrus\s*deodara|himalayan\s*cedar)\b",
        # ── Sanskrit / Malayalam forms ──
        r"\b(ആശ്വഗന്ധ|ബ്രഹ്മി|ശതാവരി|ഗുഡൂചി|ആമലകം|ഹരിദ്ര|നിമ്ബ|തുളസി|ഗോക്ഷുര|ഭൃംഗരാജ)\b",
        r"\b(अश्वगन्धा|ब्राह्मी|शतावरी|गुडूची|आमलकी|हरिद्रा|निम्ब|तुलसी|गोक्षुर|भृङ्गराज)\b",
        r"\b(यष्टिमधु|विडंग|चित्रक|मुस्ता|पुनर्नवा|वचा|शिलाजित|अश्वगन्धा)\b",
    ],

    "DOSHA": [
        r"\b(vata|vayu|vāta|वात|വാതം)\b",
        r"\b(pitta|pittam|pittā|पित्त|പിത്തം)\b",
        r"\b(kapha|kaphā|कफ|കഫം|kapham)\b",
        r"\b(tridosha|tridoṣa|त्रिदोष|ത്രിദോഷ)\b",
        r"\b(prakriti|prakṛti|प्रकृति|പ്രകൃതി)\b",
        r"\b(vikriti|vikṛti|विकृति)\b",
        r"\b(sama\s*dosha|sama\s*prakriti)\b",
        r"\b(vata\s*pitta|pitta\s*kapha|vata\s*kapha)\b",  # dual doshas
        r"\b(apana\s*vata|prana\s*vata|samana\s*vata|udana\s*vata|vyana\s*vata)\b",
        r"\b(pachaka\s*pitta|ranjaka\s*pitta|sadhaka\s*pitta|alochaka\s*pitta|bhrajaka\s*pitta)\b",
        r"\b(avalambaka\s*kapha|kledaka\s*kapha|bodhaka\s*kapha|tarpaka\s*kapha|shleshaka\s*kapha)\b",
    ],

    "DISEASE": [
        r"\b(jwara|jvara|fever|ज्वर|ജ്വരം)\b",
        r"\b(prameha|diabetes|मेह|പ്രമേഹം|madhumeha)\b",
        r"\b(amavata|rheumatism|rheumatoid\s*arthritis|आमवात)\b",
        r"\b(kushtha|skin\s*disease|dermatitis|कुष्ठ|കുഷ്ഠം)\b",
        r"\b(arsha|piles|haemorrhoids|hemorrhoids|अर्श|അർശസ്)\b",
        r"\b(atisara|diarrhoea|diarrhea|अतिसार|അതിസാരം)\b",
        r"\b(gulma|abdominal\s*lump|tumour|tumor|गुल्म)\b",
        r"\b(rakta\s*pitta|bleeding\s*disorder|रक्तपित्त)\b",
        r"\b(kasa|cough|खांसी|कास|ചുമ|kāsa)\b",
        r"\b(shvasa|asthma|dyspnoea|dyspnea|श्वास|ശ്വാസ)\b",
        r"\b(pratishyaya|rhinitis|cold|common\s*cold|प्रतिश्याय)\b",
        r"\b(hridroga|heart\s*disease|cardiac|हृद्रोग)\b",
        r"\b(shotha|oedema|edema|swelling|शोथ|ശോഥ)\b",
        r"\b(panduroga|anaemia|anemia|पांडु)\b",
        r"\b(kamala|jaundice|icterus|कामला)\b",
        r"\b(udara|ascites|abdominal\s*distension|उदर)\b",
        r"\b(grahani|irritable\s*bowel|malabsorption|ग्रहणी)\b",
        r"\b(ajirna|indigestion|dyspepsia|अजीर्ण)\b",
        r"\b(visarpa|erysipelas|spreading\s*skin|विसर्प)\b",
        r"\b(dadru|ringworm|tinea|दद्रु)\b",
        r"\b(shvitra|vitiligo|leucoderma|श्वित्र)\b",
        r"\b(apasmar|epilepsy|seizure|अपस्मार)\b",
        r"\b(unmada|psychosis|madness|उन्माद)\b",
        r"\b(vatavyadhi|neurological\s*disorder|वातव्याधि)\b",
        r"\b(gridhrasi|sciatica|गृध्रसी)\b",
        r"\b(avarana|obstruction\s*disease|आवरण)\b",
        r"\b(mutrakriccha|dysuria|painful\s*urination|मूत्रकृच्छ)\b",
        r"\b(ashmari|urinary\s*calculi|kidney\s*stones?|अश्मरी)\b",
        r"\b(vatarakta|gout|gouty\s*arthritis|वातरक्त)\b",
        r"\b(sthoulya|obesity|overweight|स्थौल्य)\b",
        r"\b(karshya|emaciation|underweight|कार्श्य)\b",
        r"\b(raktapitta|hemorrhage|रक्तपित्त)\b",
        r"\b(netraroga|eye\s*disease|ophthalmic|नेत्ररोग)\b",
        r"\b(karnashoola|earache|ear\s*pain|कर्णशूल)\b",
    ],

    "PROCEDURE": [
        # Panchakarma (5 main)
        r"\b(panchakarma|pañcakarma|पञ्चकर्म|പഞ്ചകർമ)\b",
        r"\b(vamana|emesis|vomiting\s*therapy|वमन)\b",
        r"\b(virechana|purgation|laxative\s*therapy|विरेचन)\b",
        r"\b(basti|enema|vasti|बस्ति|anuvasan|asthaapana)\b",
        r"\b(nasya|nasal\s*therapy|nasal\s*administration|नस्य)\b",
        r"\b(raktamokshana|bloodletting|leech\s*therapy|रक्तमोक्षण)\b",
        # Purvakarma (preparation)
        r"\b(snehana|oleation|oil\s*therapy|स्नेहन)\b",
        r"\b(swedana|sudation|sudotherapy|sweating\s*therapy|steam\s*therapy|स्वेदन)\b",
        # External therapies
        r"\b(abhyanga|oil\s*massage|अभ्यंग)\b",
        r"\b(shirodhara|शिरोधारा)\b",
        r"\b(shirobasti|शिरोबस्ति)\b",
        r"\b(pizhichil|sarvangadhara|oil\s*bath)\b",
        r"\b(navarakizhi|shashtika\s*shali\s*pinda\s*sweda)\b",
        r"\b(udwartanam|udvartana|powder\s*massage|उद्वर्तन)\b",
        r"\b(lepa|paste\s*application|poultice|लेप)\b",
        r"\b(avagaha|sitz\s*bath|immersion|अवगाह)\b",
        r"\b(tarpana|eye\s*nourishment|netra\s*tarpana|तर्पण)\b",
        r"\b(karnapurana|ear\s*filling|कर्णपूरण)\b",
        # Rasayana / Vajikarana
        r"\b(rasayana|rejuvenation\s*therapy|रसायन)\b",
        r"\b(vajikarana|aphrodisiac\s*therapy|वाजीकरण)\b",
        # Other treatments
        r"\b(agnikarma|cauterization|heat\s*therapy|अग्निकर्म)\b",
        r"\b(shastrakarma|surgery|surgical|शस्त्रकर्म)\b",
        r"\b(kshara\s*karma|alkaline\s*cautery|क्षारकर्म)\b",
        r"\b(marma\s*chikitsa|marma\s*therapy|vital\s*points)\b",
        r"\b(yoga\s*chikitsa|yoga\s*therapy|yogic\s*treatment)\b",
        r"\b(dhara|stream\s*pouring|ധാര)\b",
    ],

    "FORMULATION": [
        r"\b(triphala\s*churna|triphala\s*powder)\b",
        r"\b(dashamoola\s*kashaya|dasamoola\s*kwath)\b",
        r"\b(chyawanprash|chyavanaprasham)\b",
        r"\b(ashwagandha\s*churna|ashwagandha\s*powder)\b",
        r"\b(brahmi\s*ghrita|brahmi\s*ghee)\b",
        r"\b(mahanarayan\s*taila|mahanarayan\s*oil)\b",
        r"\b(trikatu\s*churna)\b",
        r"\b(arogya\s*vardhini)\b",
        r"\b(kanchanar\s*guggul|kanchnar\s*guggulu)\b",
        r"\b(punarnava\s*mandura)\b",
        r"\b(avipattikar\s*churna)\b",
        r"\b(hingvastaka\s*churna)\b",
        r"\b(saraswatarishta)\b",
        r"\b(ashwagandharishta)\b",
        r"\b(draksharishta)\b",
        r"\b(chandraprabha\s*vati)\b",
        r"\b(arogyavardhini\s*vati)\b",
        r"\b(mahamanjisthadi\s*kashaya)\b",
        r"\b(panchagavya\s*ghrita)\b",
        r"\b(guggulutiktaka\s*ghrita)\b",
        r"\b(dhanwantaram\s*taila|dhanvantaram\s*oil)\b",
        r"\b(kshirabala\s*taila)\b",
        r"\b(bala\s*taila)\b",
        r"\b(nilibhringadi\s*taila)\b",
        r"\b(pinda\s*taila)\b",
        r"\b(aragwadha\s*ghana\s*vati)\b",
        r"\b(triphala\s*guggul)\b",
        r"\b(yogaraja\s*guggul)\b",
        r"\b(mahayogaraja\s*guggul)\b",
    ],

    "SOURCE_REF": [
        r"\b(charaka\s*samhita|caraka\s*saṃhitā|चरक\s*संहिता)\b",
        r"\b(sushruta\s*samhita|suśruta\s*saṃhitā|सुश्रुत\s*संहिता)\b",
        r"\b(ashtanga\s*hridayam?|aṣṭāṅga\s*hṛdayam?|अष्टांग\s*हृदयम्|അഷ്ടാംഗഹൃദയം)\b",
        r"\b(ashtanga\s*sangraha|अष्टांग\s*संग्रह)\b",
        r"\b(madhava\s*nidana|माधव\s*निदान)\b",
        r"\b(sarangadhara\s*samhita|शार्ङ्गधर\s*संहिता)\b",
        r"\b(bhavaprakasha|भावप्रकाश)\b",
        r"\b(dhanvantari\s*nighantu|धन्वन्तरि\s*निघण्टु)\b",
        r"\b(raja\s*nighantu|राज\s*निघण्टु)\b",
        r"\b(kaideva\s*nighantu|कैदेव\s*निघण्टु)\b",
        r"\b(sharangadhara\s*samhita)\b",
        r"\b(charak|charaka|caraka)\b",
        r"\b(sushruta|suśruta)\b",
        r"\b(vagbhata|vāgbhaṭa|वाग्भट)\b",
        r"\b(dalhana|ḍalhaṇa)\b",
        r"\b(chakrapani|cakrapāṇi)\b",
        r"\b(sharangdhara|śārṅgadhara)\b",
        r"\b(kashyapa\s*samhita|kāśyapa\s*saṃhitā)\b",
        r"\b(harita\s*samhita|hārīta\s*saṃhitā)\b",
        r"\b(astanga\s*hridaya|aṣṭāṅga\s*hṛdaya)\b",
        r"\b(ഷ(ൃ|)ഷ്ടാംഗ\s*ഹൃദ|ചരക|സുശ്രുത)\b",
        # Shloka/verse references
        r"\b(shloka\s*\d+|śloka\s*\d+|verse\s*\d+|adhyaya\s*\d+)\b",
        r"\b(sutra\s*sthana|nidana\s*sthana|vimana\s*sthana|sharira\s*sthana|indriya\s*sthana|chikitsa\s*sthana|kalpa\s*sthana|uttara\s*sthana)\b",
    ],

    "QUANTITY": [
        r"\b(\d+\.?\d*)\s*(gram|gm|g|mg|kg|ml|litre|liter|tola|karsha|pala|masha|anjali|prastha|kudava|adhaka|drona|shukti|patra)\b",
        r"\b(twice|thrice|once|two|three|four)\s*(daily|a\s*day|per\s*day|times?\s*daily)\b",
        r"\b(morning|evening|night|afternoon|bedtime|noon)\s*(dose|dosage)?\b",
        r"\b\d+\s*times?\s*(a\s+)?(day|week|month)\b",
        r"\b(one|two|three|four|five)\s*(tablet|capsule|teaspoon|tablespoon|cup|pinch)\b",
        r"\b(\d+)\s*(tablet|capsule|drop|drop)s?\b",
        r"\b(half|quarter)\s*(teaspoon|tablespoon|cup)\b",
        r"\b(\d+\s*–\s*\d+|\d+\s*-\s*\d+)\s*(gram|gm|mg|ml|tola)\b",
    ],

    "PLANT_PART": [
        r"\b(root|bark|leaf|leaves|fruit|seed|flower|resin|latex|stem|rhizome|bulb|whole\s*plant)\b",
        r"\b(aerial\s*parts?|heartwood|exudate|gum|oil|juice|extract|powder|dried\s*\w+)\b",
        r"\b(मूल|त्वक्|पत्र|फल|बीज|पुष्प|निर्यास|कन्द|स्कन्ध|काण्ड|क्षीर|सार)\b",
        r"\b(വേര്|തൊലി|ഇല|ഫലം|വിത്ത്|പൂ|കിഴങ്ങ്|കാണ്ഡം|നിര്യാസം)\b",
    ],

    "BODY_PART": [
        r"\b(liver|heart|kidney|stomach|lung|intestine|skin|blood|brain|bone|muscle|spleen|colon)\b",
        r"\b(hridaya|yakrit|pleeha|mutra|rakta|asthi|majja|shukra|medas|mamsa|rasa\s*dhatu)\b",
        r"\b(ojas|tejas|prana)\b",
        r"\b(srotas|channel|dhatu|mala|urine|faeces|sweat)\b",
        r"\b(eye|ear|nose|throat|tongue|skin|head|neck|chest|abdomen|pelvis|limb)\b",
        r"\b(ഹൃദയം|കരൾ|മൂത്രം|രക്തം|അസ്ഥി|ത്വക്|ഞരമ്പ്|ആമാശയം)\b",
        r"\b(हृदय|यकृत|प्लीहा|मूत्र|रक्त|अस्थि|मज्जा|शुक्र|मेदस्|ओजस्)\b",
    ],

    "PROPERTY": [
        # Rasa (taste)
        r"\b(madhura|amla|lavana|katu|tikta|kashaya)\b",
        r"\b(sweet\s*taste|sour\s*taste|salty\s*taste|pungent\s*taste|bitter\s*taste|astringent\s*taste)\b",
        # Guna (quality)
        r"\b(laghu|guru|snigdha|ruksha|ushna|sheeta|tikshna|sthira|manda|mridu|kathina|vishada|picchila|slakshna|sthula|sukshma)\b",
        r"\b(light|heavy|oily|dry|hot|cold|sharp|stable|slow|soft|hard|non-slimy|smooth|gross|subtle)\s*(quality|property|guna)?\b",
        # Virya (potency)
        r"\b(ushna\s*virya|sheeta\s*virya|hot\s*potency|cold\s*potency)\b",
        # Vipaka (post-digestive)
        r"\b(madhura\s*vipaka|katu\s*vipaka|amla\s*vipaka)\b",
        # Prabhava
        r"\b(prabhava|special\s*potency)\b",
        r"\b(रस|गुण|वीर्य|विपाक|प्रभाव|मधुर|अम्ल|लवण|कटु|तिक्त|कषाय|लघु|गुरु|स्निग्ध|रूक्ष|उष्ण|शीत)\b",
    ],

    "DIET": [
        r"\b(pathya|apathya|wholesome|unwholesome|diet|ahara|anupana)\b",
        r"\b(milk|ghee|honey|rice|barley|wheat|lentil|sesame)\s*(is\s*)?(recommended|advised|beneficial|contraindicated)?\b",
        r"\b(pathya\s*ahara|pathyam|apathyam)\b",
        r"\b(पथ्य|अपथ्य|आहार|अनुपान)\b",
    ],
}


class AnnotationGuidelineApplier:
    def __init__(self, input_files, output_dir, logger, lang_hint="auto", prev_outputs=None):
        self.input_files = input_files
        self.output_dir  = Path(output_dir)
        self.logger      = logger
        self.lang_hint   = lang_hint

        # Compile patterns once; store (label, pattern, priority) tuples
        self.compiled = []
        for label, patterns in ENTITY_SEEDS.items():
            priority = LABEL_PRIORITY.get(label, 5)
            for p in patterns:
                self.compiled.append((label, re.compile(p, re.IGNORECASE | re.UNICODE), priority))

    # ------------------------------------------------------------------ #
    #  Annotation with overlap resolution                                  #
    # ------------------------------------------------------------------ #

    def annotate(self, text: str) -> list:
        """Return deduplicated, priority-resolved annotation spans."""
        raw_spans = []
        for label, pattern, priority in self.compiled:
            for m in pattern.finditer(text):
                raw_spans.append({
                    "start":    m.start(),
                    "end":      m.end(),
                    "text":     m.group(0),
                    "label":    label,
                    "priority": priority,
                    "length":   m.end() - m.start(),
                })

        # Sort: by start position, then by descending priority, then by length
        raw_spans.sort(key=lambda s: (s["start"], -s["priority"], -s["length"]))

        # Greedy non-overlapping selection
        deduped = []
        last_end = -1
        for s in raw_spans:
            if s["start"] >= last_end:
                deduped.append(s)
                last_end = s["end"]

        # Remove internal priority key from output
        for s in deduped:
            del s["priority"]
            del s["length"]

        return deduped

    def count_by_label(self, spans: list) -> dict:
        counts = {}
        for s in spans:
            counts[s["label"]] = counts.get(s["label"], 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    #  Runner                                                              #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        output_files = []
        total_entity_counts = {label: 0 for label in ENTITY_SEEDS}
        errors = []

        # Save guidelines doc once
        guide_path = self.output_dir / "annotation_guidelines.json"
        guide_path.write_text(
            json.dumps(ANNOTATION_GUIDELINES, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        for f in self.input_files:
            try:
                text  = f.read_text(encoding="utf-8", errors="replace")
                spans = self.annotate(text)
                file_counts = self.count_by_label(spans)

                for lbl, cnt in file_counts.items():
                    total_entity_counts[lbl] = total_entity_counts.get(lbl, 0) + cnt

                result = {
                    "source_file":    f.name,
                    "text":           text,
                    "annotations":    spans,
                    "entity_counts":  file_counts,
                    "total_entities": len(spans),
                }

                out_file = self.output_dir / (f.stem + "_annotated.json")
                out_file.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                output_files.append(f)
                self.logger.debug(f"  Annotated {f.name}: {len(spans)} spans across {len(file_counts)} label types")
            except Exception as e:
                errors.append(str(e))
                self.logger.error(f"  ERROR annotating {f.name}: {e}")

        total_entities = sum(total_entity_counts.values())
        summary = (
            f"{len(output_files)} files, {total_entities} entities tagged — "
            + ", ".join(f"{k}:{v}" for k, v in total_entity_counts.items() if v > 0)
        )
        return {
            "task":           "NLP-03",
            "output_files":   output_files,
            "entity_counts":  total_entity_counts,
            "total_entities": total_entities,
            "summary":        summary,
        }