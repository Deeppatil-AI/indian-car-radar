"""
Metadata and taxonomy for Indian Car Models and Generations.
"""

INDIAN_CAR_CLASSES = [
    {
        "id": "maruti_swift_gen3",
        "make": "Maruti Suzuki",
        "model": "Swift",
        "generation": "Gen 3 (2018-2023)",
        "year_span": "2018-2023",
        "body_type": "Hatchback",
        "segment": "B-Segment Hatch",
        "key_visual_cues": [
            "Hexagonal wide mesh front grille",
            "C-pillar integrated hidden rear door handles",
            "Swept-back LED projector headlamps with DRLs",
            "Floating roof effect with blacked-out A and B pillars"
        ],
        "engine": "1.2L K-Series DualJet Petrol",
        "power": "89 bhp / 113 Nm",
        "query_keywords": ["Maruti Suzuki Swift 2018 2022", "Swift Gen 3 car exterior"]
    },
    {
        "id": "tata_nexon_facelift",
        "make": "Tata Motors",
        "model": "Nexon",
        "generation": "Facelift Gen 2 (2023-Present)",
        "year_span": "2023-Present",
        "body_type": "Compact SUV",
        "segment": "Sub-4m SUV",
        "key_visual_cues": [
            "Bi-functional sequential LED DRLs on top bumper",
            "Split headlamp cluster integrated into lower bumper",
            "Connected X-factor rear LED lightbar with dynamic greeting",
            "Dual-tone coupe-like sloping roofline"
        ],
        "engine": "1.2L Turbo Revotron Petrol / 1.5L Revotorq Diesel",
        "power": "118 bhp / 170 Nm",
        "query_keywords": ["Tata Nexon 2023 2024 facelift", "Tata Nexon new model exterior"]
    },
    {
        "id": "mahindra_thar_gen2",
        "make": "Mahindra",
        "model": "Thar",
        "generation": "Gen 2 (2020-Present)",
        "year_span": "2020-Present",
        "body_type": "4x4 Offroader SUV",
        "segment": "Lifestyle 4x4",
        "key_visual_cues": [
            "Iconic vertical 6-slat retro front grille",
            "Round halogen/LED headlamps with fender DRLs",
            "Exposed door hinges and flared boxy wheel arches",
            "Tailgate-mounted full-size spare tire with rectangular LED taillamps"
        ],
        "engine": "2.0L mStallion Turbo Petrol / 2.2L mHawk Diesel",
        "power": "150 bhp / 320 Nm",
        "query_keywords": ["Mahindra Thar 2020 2023 2024", "Mahindra Thar 4x4 exterior"]
    },
    {
        "id": "hyundai_creta_gen2",
        "make": "Hyundai",
        "model": "Creta",
        "generation": "Gen 2 (2020-2023)",
        "year_span": "2020-2023",
        "body_type": "Mid-Size SUV",
        "segment": "C-Segment SUV",
        "key_visual_cues": [
            "Cascading parametric jewel 3D front grille",
            "Split crescent-shaped LED DRLs encircling trio headlamps",
            "Boomerang-style split LED taillamps on tailgate",
            "Contrasting silver/black lightning arch over C-pillar"
        ],
        "engine": "1.5L MPi Petrol / 1.5L CRDi Diesel",
        "power": "113 bhp / 250 Nm",
        "query_keywords": ["Hyundai Creta 2020 2022 2023", "Hyundai Creta second generation"]
    },
    {
        "id": "mahindra_scorpio_n",
        "make": "Mahindra",
        "model": "Scorpio-N",
        "generation": "Gen 3 / Scorpio-N (2022-Present)",
        "year_span": "2022-Present",
        "body_type": "Full-Size D-SUV",
        "segment": "D-Segment SUV",
        "key_visual_cues": [
            "Chrome-toothed twin-peak grille with new Mahindra logo",
            "Sting-like sequential LED turn indicators around fog lamps",
            "Tall vertical Volvo-style stacked LED taillamps",
            "Muscular ladder-frame upright tall stance"
        ],
        "engine": "2.2L mHawk Diesel / 2.0L mStallion TGDi",
        "power": "172 bhp / 400 Nm",
        "query_keywords": ["Mahindra Scorpio N 2022 2023 2024", "Mahindra Scorpio N exterior"]
    },
    {
        "id": "tata_punch",
        "make": "Tata Motors",
        "model": "Punch",
        "generation": "Gen 1 (2021-Present)",
        "year_span": "2021-Present",
        "body_type": "Micro SUV",
        "segment": "Sub-Compact SUV",
        "key_visual_cues": [
            "Humanity line high-set eyebrow LED DRLs with tri-arrow motifs",
            "Cladding-heavy rugged bumpers with 187mm ground clearance",
            "90-degree wide opening doors",
            "Compact Y-shaped taillight signatures"
        ],
        "engine": "1.2L Revotron 3-Cylinder Petrol",
        "power": "86 bhp / 115 Nm",
        "query_keywords": ["Tata Punch 2021 2023 2024", "Tata Punch micro SUV"]
    },
    {
        "id": "kia_seltos_gen1",
        "make": "Kia",
        "model": "Seltos",
        "generation": "Gen 1 (2019-2023)",
        "year_span": "2019-2023",
        "body_type": "Mid-Size SUV",
        "segment": "C-Segment SUV",
        "key_visual_cues": [
            "Signature knurled chrome Tiger Nose grille",
            "Crown Jewel LED headlamps extending into grille lightbar",
            "Heartbeat-pattern LED daytime running lights and fog towers",
            "Aggressive dual exhaust faux skid plates"
        ],
        "engine": "1.4L Turbo GDI / 1.5L Smartstream",
        "power": "138 bhp / 242 Nm",
        "query_keywords": ["Kia Seltos 2019 2022 2023", "Kia Seltos car exterior"]
    },
    {
        "id": "toyota_fortuner_gen2",
        "make": "Toyota",
        "model": "Fortuner",
        "generation": "Gen 2 / Legender (2016-Present)",
        "year_span": "2016-Present",
        "body_type": "Full-Size 4x4 SUV",
        "segment": "Premium Ladder-Frame SUV",
        "key_visual_cues": [
            "Massive high-riding prominent nose with chrome slats",
            "Slim bi-beam LED headlamps with distinct eyebrow guide lights",
            "Distinctive kick-up beltline at the rear quarter glass",
            "Bulky squared wheel arches with 18-inch alloy wheels"
        ],
        "engine": "2.8L 4-Cylinder D-4D Turbo Diesel",
        "power": "201 bhp / 500 Nm",
        "query_keywords": ["Toyota Fortuner 2017 2021 2023", "Toyota Fortuner Legender"]
    },
    {
        "id": "maruti_baleno_gen2",
        "make": "Maruti Suzuki",
        "model": "Baleno",
        "generation": "Gen 2 (2022-Present)",
        "year_span": "2022-Present",
        "body_type": "Premium Hatchback",
        "segment": "Premium Hatch",
        "key_visual_cues": [
            "Crafted Futurism 3-piece NEXTre LED DRL signature",
            "Wide wave-shaped honeycomb front grille with chrome surround",
            "Sharp split wrap-around LED taillamps extending onto bootlid",
            "Liquid flow aerodynamic sculpted side shoulder line"
        ],
        "engine": "1.2L DualJet Dual VVT Petrol with Idle Start-Stop",
        "power": "88 bhp / 113 Nm",
        "query_keywords": ["Maruti Suzuki Baleno 2022 2023 2024", "Baleno new model exterior"]
    },
    {
        "id": "honda_city_gen5",
        "make": "Honda",
        "model": "City",
        "generation": "Gen 5 (2020-Present)",
        "year_span": "2020-Present",
        "body_type": "Sedan",
        "segment": "Mid-Size Sedan",
        "key_visual_cues": [
            "Solid Wing face with bold chrome upper grille bar",
            "9-LED array inline headlamps with integrated L-shaped DRLs",
            "Z-shaped 3D wrap-around LED taillights",
            "Long executive silhouette with katana-blade character line"
        ],
        "engine": "1.5L i-VTEC DOHC Petrol / e:HEV Strong Hybrid",
        "power": "119 bhp / 145 Nm",
        "query_keywords": ["Honda City 2020 2022 2023 2024", "Honda City Gen 5 sedan"]
    },
    {
        "id": "mahindra_xuv700",
        "make": "Mahindra",
        "model": "XUV700",
        "generation": "Gen 1 (2021-Present)",
        "year_span": "2021-Present",
        "body_type": "Premium Mid-SUV",
        "segment": "D-Segment SUV",
        "key_visual_cues": [
            "Large C-shaped striking LED DRLs flowing into bumper",
            "Smart flush-fitting pop-out door handles",
            "Satin-finish vertical slats on piano-black front grille",
            "Arrow-tip sculpted LED rear taillamps with flared rear haunches"
        ],
        "engine": "2.0L mStallion Turbo Petrol / 2.2L mHawk Diesel",
        "power": "197 bhp / 380 Nm",
        "query_keywords": ["Mahindra XUV700 2021 2023 2024", "Mahindra XUV700 exterior"]
    },
    {
        "id": "hyundai_i20_gen3",
        "make": "Hyundai",
        "model": "i20",
        "generation": "Gen 3 (2020-Present)",
        "year_span": "2020-Present",
        "body_type": "Premium Hatchback",
        "segment": "B+ Segment Hatch",
        "key_visual_cues": [
            "Sensuous Sportiness parametric cascading black grille",
            "Z-shaped distinctive LED taillights connected by chrome strip",
            "Sharp angular projector headlamps with eyebrow LED DRLs",
            "Shark-fin C-pillar with dark quarter glass trim"
        ],
        "engine": "1.2L Kappa Petrol / 1.0L Turbo GDi",
        "power": "118 bhp / 172 Nm",
        "query_keywords": ["Hyundai i20 2020 2022 2023", "Hyundai i20 third generation"]
    }
]

CLASS_MAP = {c["id"]: c for c in INDIAN_CAR_CLASSES}
CLASS_NAMES = [c["id"] for c in INDIAN_CAR_CLASSES]
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}

def get_class_info(class_id_or_index):
    if isinstance(class_id_or_index, int):
        class_id = INDEX_TO_LABEL.get(class_id_or_index, None)
    else:
        class_id = class_id_or_index
    return CLASS_MAP.get(class_id, None)
