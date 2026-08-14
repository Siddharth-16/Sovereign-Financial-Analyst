"""Regression/dev benchmark for Sovereign Financial Analyst.

These 28 questions have already been used during development and are NOT a
clean held-out test set.

Gold evidence is deterministic corpus evidence. `text` is the primary verbatim
span. `alternatives` contains only verbatim indexed passages that were audited
as expressing the same factual label; this prevents a question with no requested
fiscal year from failing solely because an equivalent indexed-year passage was
retrieved. Changed facts/segment structures are NOT treated as alternatives.

Expected facts are scored separately from groundedness with deterministic
literal/regex patterns. A future held-out set should live in dataset_test.py and
be frozen before application optimization.
"""

DATASET_NAME = "sovereign-financial-analyst-regression"
DATASET_VERSION = "dev-v5-tesla-2025-gold-2026-08-14"
DATASET_SPLIT = "dev"

EVAL_QUESTIONS = [{'id': 'biz-01',
  'question': "What are Nvidia's main business segments and how does the company describe its core products?",
  'expected_company': 'nvidia',
  'expected_section': 'business',
  'category': 'business',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'segments-compute-networking',
                     'text': 'We report our business results in two segments. The Compute & Networking '
                             'segment includes our Data Center accelerated computing platforms and AI '
                             'solutions and software; networking; automotive platforms and autonomous and '
                             'electric vehicle solutions; Jetson for robotics and other embedded platforms; '
                             'and DGX Cloud computing services.',
                     'content_fingerprint': '3e9b4fe3fae572051a54',
                     'source_path': 'data/raw/nvidia/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['Headquartered in Santa Clara, California, NVIDIA was incorporated in '
                                      'California in April 1993 and reincorporated in Delaware in April '
                                      '1998. Termination of the Arm Share Purchase Agreement In February '
                                      '2022, NVIDIA and SoftBank Group Corp., or SoftBank, announced the '
                                      'termination of the Share Purchase Agreement whereby NVIDIA would have '
                                      'acquired Arm Limited, or Arm, from SoftBank. The parties agreed to '
                                      'terminate because of significant regulatory challenges preventing the '
                                      'completion of the transaction. We recorded an acquisition termination '
                                      'cost of $1.35 billion in fiscal year 2023 reflecting the write-off of '
                                      'the prepayment provided at signing. Our Businesses We report our '
                                      'business results in two segments. The Compute & Networking segment '
                                      'includes our Data Center accelerated computing platform; networking; '
                                      'automotive AI Cockpit, autonomous driving development agreements, and '
                                      'autonomous vehicle solutions; electric vehicle computing platforms; '
                                      'Jetson for robotics and other embedded platforms; NVIDIA AI '
                                      'Enterprise and other software; and cryptocurrency mining processors, '
                                      'or CMP.',
                                      'Professional artists, architects and designers use NVIDIA partner '
                                      'products accelerated with our GPUs and software platform for a range '
                                      'of creative and design use cases, such as creating visual effects in '
                                      'movies or designing buildings and products. In addition, generative '
                                      'AI is expanding the market for our workstation-class GPUs, as more '
                                      'enterprise customers develop and deploy AI applications with their '
                                      'data on-premises. Headquartered in Santa Clara, California, NVIDIA '
                                      'was incorporated in California in April 1993 and reincorporated in '
                                      'Delaware in April 1998. Our Businesses We report our business results '
                                      'in two segments. The Compute & Networking segment is comprised of our '
                                      'Data Center accelerated computing platforms and end-to-end networking '
                                      'platforms including Quantum for InfiniBand and Spectrum for Ethernet; '
                                      'our NVIDIA DRIVE automated-driving platform and automotive '
                                      'development agreements; Jetson robotics and other embedded platforms; '
                                      'NVIDIA AI Enterprise and other software; and DGX Cloud software and '
                                      'services.']},
                    {'id': 'segments-graphics',
                     'text': 'The Graphics segment includes GeForce GPUs for gaming and PCs, the GeForce NOW '
                             'game streaming service and related infrastructure, and solutions for gaming '
                             'platforms; Quadro/NVIDIA RTX GPUs for enterprise workstation graphics; virtual '
                             'GPU, or vGPU, software for cloud-based visual and virtual computing; '
                             'automotive platforms for infotainment systems; and Omniverse Enterprise '
                             'software for building and operating industrial AI and digital twin '
                             'applications.',
                     'content_fingerprint': 'bc48e980c398fbb0a83e',
                     'source_path': 'data/raw/nvidia/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['The Graphics segment includes GeForce GPUs for gaming and PCs, the '
                                      'GeForce NOW game streaming service and related infrastructure, and '
                                      'solutions for gaming platforms; Quadro/NVIDIA RTX GPUs for enterprise '
                                      'workstation graphics; virtual GPU, or vGPU, software for cloud-based '
                                      'visual and virtual computing; automotive platforms for infotainment '
                                      'systems; and Omniverse Enterprise software for building and operating '
                                      'metaverse and 3D internet applications. Our Markets We specialize in '
                                      'markets in which our computing platforms can provide tremendous '
                                      'acceleration for applications. These platforms incorporate '
                                      'processors, interconnects, software, algorithms, systems, and '
                                      'services to deliver unique value. Our platforms address four large '
                                      'markets where our expertise is critical: Data Center, Gaming, '
                                      'Professional Visualization, and Automotive. Data Center',
                                      'The Graphics segment includes GeForce GPUs for gaming and PCs, the '
                                      'GeForce NOW game streaming service and related infrastructure; '
                                      'Quadro/NVIDIA RTX GPUs for enterprise workstation graphics; virtual '
                                      'GPU, or vGPU, software for cloud-based visual and virtual computing; '
                                      'automotive platforms for infotainment systems; and Omniverse '
                                      'Enterprise software for building and operating metaverse and 3D '
                                      'internet applications. Our Markets We specialize in markets where our '
                                      'computing platforms can provide tremendous acceleration for '
                                      'applications. These platforms incorporate processors, interconnects, '
                                      'software, algorithms, systems, and services to deliver unique value. '
                                      'Our platforms address four large markets where our expertise is '
                                      'critical: Data Center, Gaming, Professional Visualization, and '
                                      'Automotive. Data Center']}],
  'expected_facts': [{'id': 'compute-networking',
                      'description': 'Compute & Networking is a reportable segment',
                      'patterns': ['Compute & Networking'],
                      'regex_patterns': ['Compute\\s*(?:&|and)\\s*Networking']},
                     {'id': 'graphics',
                      'description': 'Graphics is a reportable segment',
                      'patterns': ['Graphics segment'],
                      'regex_patterns': ['\\bGraphics\\b.*\\bsegment\\b|\\bsegment\\b.*\\bGraphics\\b']},
                     {'id': 'core-products',
                      'description': 'Core offerings include GPU/RTX and data-center/AI/networking platforms',
                      'patterns': [],
                      'regex_patterns': ['(?:GeForce|RTX|GPU).*(?:Data Center|AI|network)|(?:Data '
                                         'Center|AI|network).*(?:GeForce|RTX|GPU)']}]},
 {
    "id": "biz-02",
    "question": "What is Tesla's business model and what are its primary product lines?",
    "expected_company": "tesla",
    "expected_section": "business",
    "category": "business",
    "answerable": True,
    "gold_fiscal_year": 2025,

    "gold_evidence": [
        {
            "id": "primary-product-categories",
            "text": (
                "We operate as two reportable segments: (i) automotive and "
                "(ii) energy generation and storage. The automotive segment "
                "includes the design, development, manufacturing, sales and "
                "leasing of high-performance fully electric vehicles as well "
                "as sales of automotive regulatory credits. Additionally, the "
                "automotive segment also includes services and other, which "
                "includes sales of used vehicles, non-warranty maintenance "
                "services and collision, paid Supercharging sessions, "
                "automotive insurance business revenue, part sales and retail "
                "merchandise sales. The energy generation and storage segment "
                "includes sales, leasing, and financing of energy generation "
                "and storage products, services related to such products and "
                "sales of energy generation incentives."
            ),
            "alternatives": [
                (
                    "ITEM 1. BUSINESS Overview We design, develop, manufacture, "
                    "sell and lease high-performance fully electric vehicles "
                    "and energy generation and storage systems, and offer "
                    "services related to our products."
                )
            ],
            "source_path": "data/raw/tesla/2025_10k.html",
            "source_fiscal_year": 2025,
        },
        {
            "id": "business-model",
            "text": (
                "We believe that this mission, along with our engineering "
                "expertise, advancements in real-world AI, vertically "
                "integrated business model, and focus on user experience "
                "differentiate us from other companies."
            ),
            "alternatives": [
                (
                    "We believe that this mission, along with our engineering "
                    "expertise, vertically integrated business model and focus "
                    "on user experience differentiate us from other companies."
                )
            ],
            "source_path": "data/raw/tesla/2025_10k.html",
            "source_fiscal_year": 2025,
        },
    ],

    "expected_facts": [
        {
            "id": "electric-vehicles",
            "description": "Primary products include fully electric vehicles",
            "patterns": ["fully electric vehicles"],
            "regex_patterns": [
                r"(?:fully\s+)?electric\s+vehicles?"
            ],
        },
        {
            "id": "energy-products",
            "description": (
                "Primary products include energy generation and storage products"
            ),
            "patterns": ["energy generation and storage"],
            "regex_patterns": [
                r"energy\s+generation\s+and\s+storage(?:\s+(?:systems|products))?"
            ],
        },
        {
            "id": "business-model",
            "description": "Business model includes vertical integration",
            "patterns": ["vertically integrated business model"],
            "regex_patterns": [
                r"vertically\s+integrated\s+business\s+model"
            ],
        },
    ],

    "annotation_note": (
        "Gold now targets Tesla's FY2025 10-K because unspecified-year "
        "application queries resolve to the latest indexed filing. FY2024 "
        "passages are retained only where they express the same underlying fact."
    ),
},
 {'id': 'biz-03',
  'question': "What are JPMorgan Chase's main reportable business segments?",
  'expected_company': 'jpmorgan_chase',
  'expected_section': 'business',
  'category': 'business',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'reportable-segments',
                     'text': 'As a result of the reorganization, the Firm has three reportable business '
                             'segments – Consumer & Community Banking (“CCB”), Commercial & Investment Bank '
                             '(“CIB”) and Asset & Wealth Management (“AWM”) – with the remaining activities '
                             'in Corporate.',
                     'content_fingerprint': '832a75beb925da3e06a2',
                     'source_path': 'data/raw/jpmorgan_chase/2025_10k.html',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'ccb',
                      'description': 'Consumer & Community Banking',
                      'patterns': ['Consumer & Community Banking'],
                      'regex_patterns': ['Consumer\\s*(?:&|and)\\s*Community\\s+Banking|\\bCCB\\b']},
                     {'id': 'cib',
                      'description': 'Commercial & Investment Bank',
                      'patterns': ['Commercial & Investment Bank'],
                      'regex_patterns': ['Commercial\\s*(?:&|and)\\s*Investment\\s+Bank|\\bCIB\\b']},
                     {'id': 'awm',
                      'description': 'Asset & Wealth Management',
                      'patterns': ['Asset & Wealth Management'],
                      'regex_patterns': ['Asset\\s*(?:&|and)\\s*Wealth\\s+Management|\\bAWM\\b']}]},
 {'id': 'biz-04',
  'question': 'How does Walmart describe its business strategy and store formats?',
  'expected_company': 'walmart',
  'expected_section': 'business',
  'category': 'business',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'strategy',
                     'text': 'Our strategy is to make every day easier for busy families, operate with '
                             'discipline, sharpen our culture and become more digital, and make trust a '
                             'competitive advantage. Making life easier for busy families includes our '
                             'commitment to price leadership, which has been and will remain a cornerstone '
                             'of our business, as well as increasing convenience to save our customers time.',
                     'content_fingerprint': '0347423739df50854249',
                     'source_path': 'data/raw/walmart/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['Our strategy is to make every day easier for busy families, operate '
                                      'with discipline, sharpen our culture and become more digital, and '
                                      'make trust a competitive advantage. Making life easier for busy '
                                      'families includes our commitment to price leadership, which has been '
                                      'and will remain a cornerstone of our business, as well as increasing '
                                      'convenience to save our customers time. By leading on price, we earn '
                                      'the trust of our customers every day by providing a broad assortment '
                                      'of quality merchandise and services at everyday low prices ("EDLP"). '
                                      'EDLP is our pricing philosophy under which we price items at a low '
                                      'price every day so our customers trust that our prices will not '
                                      'change under frequent promotional activity. Everyday low cost '
                                      '("EDLC") is our commitment to control expenses so our cost savings '
                                      'can be passed along to our customers.']},
                    {'id': 'store-formats',
                     'text': 'Supercenters (general merchandise and grocery) 69,000 260,000 178,000 Discount '
                             'stores (general merchandise and limited grocery) 30,000 206,000 105,000 '
                             'Neighborhood markets (1) (grocery) 28,000 65,000 42,000',
                     'content_fingerprint': '086e7b623b6e27570145',
                     'source_path': 'data/raw/walmart/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['n 4,600 pickup locations and more than 3,900 same-day delivery '
                                      'locations. Our Walmart+ membership offering provides enhanced '
                                      'omni-channel shopping benefits including unlimited free shipping on '
                                      'eligible items with no order minimum, unlimited delivery from store, '
                                      'fuel discounts, access to Paramount+ streaming service, and mobile '
                                      'scan & go for a streamlined in-store shopping experience. We have '
                                      'several eCommerce websites, the largest of which is walmart.com. We '
                                      'define eCommerce sales as sales initiated by customers digitally and '
                                      'fulfilled by a number of methods including our dedicated eCommerce '
                                      'fulfillment centers and leveraging our stores, as well as certain '
                                      'other business offerings that are part of our flywheel strategy, such '
                                      'as our Walmart Connect advertising business. The following table '
                                      'provides the approximate size of our retail stores as of January 31, '
                                      '2023: Minimum Square Feet Maximum Square Feet Average Square Feet '
                                      'Supercenters (general merchandise and grocery) 69,000 260,000 178,000 '
                                      'Discount stores (general merchandise and limited grocery) 30,000 '
                                      '206,000 105,000 Neighborhood markets (1) (grocery) 28,000 65,000 '
                                      '42,000 (1) Excludes other small formats. Merchandise.',
                                      'pickup locations and more than 4,300 locations offer same-day '
                                      'delivery. Our Walmart+ membership offering provides enhanced '
                                      'omni-channel shopping benefits including unlimited free shipping on '
                                      'eligible items with no order minimum, unlimited delivery from store, '
                                      'fuel discounts, mobile Scan & Go and access to additional member '
                                      'benefits. We define eCommerce sales as sales initiated by customers '
                                      'digitally and fulfilled by a number of methods including our '
                                      'dedicated eCommerce fulfillment centers and leveraging our stores, as '
                                      'well as certain other business offerings that are part of our '
                                      'ecosystem, such as our Walmart Connect advertising business. The '
                                      'following table provides the approximate size of our retail stores as '
                                      'of January 31, 2024: Minimum Square Feet Maximum Square Feet Average '
                                      'Square Feet Supercenters (general merchandise and grocery) 69,000 '
                                      '260,000 178,000 Discount stores (general merchandise and limited '
                                      'grocery) 30,000 206,000 105,000 Neighborhood markets (1) (grocery) '
                                      '28,000 65,000 42,000 (1) Excludes other small formats. Merchandise. '
                                      'Walmart U.S. does business primarily in three strategic merchandise '
                                      'units, listed below: •']}],
  'expected_facts': [{'id': 'strategy-price-convenience',
                      'description': 'Strategy emphasizes price leadership and convenience',
                      'patterns': ['price leadership', 'convenience'],
                      'regex_patterns': ['(?:price leadership|everyday low '
                                         'prices|EDLP).*(?:convenience|save.*time)|(?:convenience|save.*time).*(?:price '
                                         'leadership|everyday low prices|EDLP)']},
                     {'id': 'supercenters',
                      'description': 'Supercenters store format',
                      'patterns': ['Supercenters'],
                      'regex_patterns': ['\\bSupercenters?\\b']},
                     {'id': 'discount-neighborhood',
                      'description': 'Discount stores and Neighborhood Markets',
                      'patterns': [],
                      'regex_patterns': ['Discount\\s+stores?.*Neighborhood\\s+(?:markets?|Markets?)|Neighborhood\\s+(?:markets?|Markets?).*Discount\\s+stores?']}]},
 {'id': 'biz-05',
  'question': "What are Eli Lilly's main therapeutic areas and business segments?",
  'expected_company': 'eli_lilly',
  'expected_section': 'business',
  'category': 'business',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'single-segment',
                     'text': 'We discover, develop, manufacture, and market products in a single business '
                             'segment—human pharmaceutical products.',
                     'content_fingerprint': '2ae2b2501866f267ea23',
                     'source_path': 'data/raw/eli_lilly/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['Item 1. Business Eli Lilly and Company (referred to as the company, '
                                      'Lilly, we, or us) was incorporated in 1901 in Indiana to succeed to '
                                      'the drug manufacturing business founded in Indianapolis, Indiana, in '
                                      '1876 by Colonel Eli Lilly. We discover, develop, manufacture, and '
                                      'market products in a single business segment&#8212;human '
                                      'pharmaceutical products. Our purpose is to unite caring with '
                                      'discovery to create medicines that make life better for people around '
                                      'the world. Most of the products that we sell today were discovered or '
                                      'developed by our own scientists, and our long-term success depends on '
                                      'our ability to continually discover or acquire, develop, and '
                                      'commercialize innovative medicines. We manufacture and distribute our '
                                      'products through facilities in the United States (U.S.), including '
                                      'Puerto Rico, and 7 other countries. Our products are sold in '
                                      'approximately 110 countries. Products Our products include: Diabetes '
                                      'products , including: &#8226; Basaglar &#174; , in collaboration with '
                                      'Boehringer Ingelheim, a long-acting human insulin analog for the '
                                      'treatment of diabetes.',
                                      'Item 1. Business Eli Lilly and Company (referred to as the company, '
                                      'Lilly, we, or us) was incorporated in 1901 in Indiana to succeed to '
                                      'the drug manufacturing business founded in Indianapolis, Indiana, in '
                                      '1876 by Colonel Eli Lilly. We discover, develop, manufacture, and '
                                      'market products in a single business segment&#8212;human '
                                      'pharmaceutical products. Our purpose is to unite caring with '
                                      'discovery to create medicines that make life better for people around '
                                      'the world. Most of the products that we sell today were discovered or '
                                      'developed by our own scientists, and our long-term success depends on '
                                      'our ability to continually discover or acquire, develop, and '
                                      'commercialize innovative medicines. We manufacture and distribute our '
                                      'products through facilities in the United States (U.S.), including '
                                      'Puerto Rico, and in Europe and Asia. Our products are sold in '
                                      'approximately 105 countries. Products Our products include: '
                                      '##TABLE_START Therapeutic area Products Certain Indications Diabetes, '
                                      'Obesity and Other Cardiometabolic products Basaglar &#174; In '
                                      'collaboration with Boehringer Ingelheim, a long-acting human insulin '
                                      'analog for the treatment of diabetes.']},
                    {'id': 'therapeutic-areas',
                     'text': 'Our internal pharmaceutical research focuses primarily on the areas of '
                             'immunology, metabolism (including diabetes, obesity and cardiovascular), '
                             'neuroscience, and oncology.',
                     'content_fingerprint': 'b57a157361a5b204c62f',
                     'source_path': 'data/raw/eli_lilly/2025_10k.html',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'single-pharma-segment',
                      'description': 'Single human pharmaceutical products business segment',
                      'patterns': ['single business segment', 'human pharmaceutical products'],
                      'regex_patterns': ['single\\s+(?:business\\s+)?segment.*human\\s+pharmaceutical|human\\s+pharmaceutical.*single\\s+(?:business\\s+)?segment']},
                     {'id': 'therapeutic-areas',
                      'description': 'Primary research areas: immunology, metabolism, neuroscience, oncology',
                      'patterns': [],
                      'regex_patterns': ['immunology.*(?:metabolism|diabetes|obesity).*neuroscience.*oncology']}]},
 {'id': 'biz-06',
  'question': "What are Boeing's primary business segments?",
  'expected_company': 'boeing',
  'expected_section': 'business',
  'category': 'business',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'three-segments',
                     'text': 'We operate in three reportable segments: • Commercial Airplanes (BCA); • '
                             'Defense, Space & Security (BDS); • Global Services (BGS).',
                     'content_fingerprint': '0e9b6cae8376aa3628c3',
                     'source_path': 'data/raw/boeing/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['Item 1. Business The Boeing Company, together with its subsidiaries '
                                      '(herein referred to as “Boeing,” the “Company,” “we,” “us,” “our”), '
                                      'is one of the world’s major aerospace firms. We are organized based '
                                      'on the products and services we offer. We operate in three reportable '
                                      'segments: • Commercial Airplanes (BCA); • Defense, Space & Security '
                                      '(BDS); • Global Services (BGS). Commercial Airplanes Segment This '
                                      'segment develops, produces and markets commercial jet aircraft '
                                      'principally to the commercial airline industry worldwide. We are a '
                                      'leading producer of commercial aircraft and offer a family of '
                                      'commercial jetliners designed to meet a broad spectrum of global '
                                      'passenger and cargo requirements of airlines. This family of '
                                      'commercial jet aircraft in production includes the 737 narrow-body '
                                      'model and the 767, 777 and 787 wide-body models. Development '
                                      'continues on the 777X program and the 737-7 and 737-10 derivatives. '
                                      'Defense, Space & Security Segment',
                                      'Item 1. Business The Boeing Company, together with its subsidiaries '
                                      '(herein referred to as “Boeing,” the “Company,” “we,” “us,” “our”), '
                                      'is one of the world’s major aerospace firms. We are organized based '
                                      'on the products and services we offer. We operate in four reportable '
                                      'segments: • Commercial Airplanes (BCA); • Defense, Space & Security '
                                      '(BDS); • Global Services (BGS); • Boeing Capital (BCC). Commercial '
                                      'Airplanes Segment This segment develops, produces and markets '
                                      'commercial jet aircraft principally to the commercial airline '
                                      'industry worldwide. We are a leading producer of commercial aircraft '
                                      'and offer a family of commercial jetliners designed to meet a broad '
                                      'spectrum of global passenger and cargo requirements of airlines. This '
                                      'family of commercial jet aircraft in production includes the 737 '
                                      'narrow-body model and the 767, 777 and 787 wide-body models. We ended '
                                      'production of the 747 wide-body model in 2022. Development continues '
                                      'on the 777X program and the 737-7 and 737-10 derivatives. Defense, '
                                      'Space & Security Segment']}],
  'expected_facts': [{'id': 'bca',
                      'description': 'Commercial Airplanes',
                      'patterns': ['Commercial Airplanes'],
                      'regex_patterns': ['Commercial\\s+Airplanes|\\bBCA\\b']},
                     {'id': 'bds',
                      'description': 'Defense, Space & Security',
                      'patterns': ['Defense, Space & Security'],
                      'regex_patterns': ['Defense[,\\s]+Space\\s*(?:&|and)\\s*Security|\\bBDS\\b']},
                     {'id': 'bgs',
                      'description': 'Global Services',
                      'patterns': ['Global Services'],
                      'regex_patterns': ['Global\\s+Services|\\bBGS\\b']}]},
 {'id': 'biz-07',
  'question': 'How does Visa describe its core business and revenue model?',
  'expected_company': 'visa',
  'expected_section': 'business',
  'category': 'business',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'core-business',
                     'text': 'Visa earns revenue by facilitating money movement across more than 200 '
                             'countries and territories, among a global set of consumers, sellers, financial '
                             'institutions and government entities, through innovative technologies.',
                     'content_fingerprint': '6fc855e759ee22da3816',
                     'source_path': 'data/raw/visa/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['The acquirer pays the amount of the purchase, minus the merchant '
                                      'discount rate (MDR), to the merchant. Visa earns revenue by '
                                      'facilitating money movement across more than 200 countries and '
                                      'territories among a global set of consumers, merchants, financial '
                                      'institutions and government entities through innovative technologies. '
                                      '5 Table of Contents Our net revenues in fiscal year 2023 consisted of '
                                      'the following: SERVICE REVENUES Earned for services provided in '
                                      'support of client usage of Visa payment services OTHER REVENUES '
                                      'Consist mainly of value added services related to advisory, marketing '
                                      'and certain card benefits; license fees for use of the Visa brand or '
                                      'technology; and fees for account holder services, certification and '
                                      'licensing DATA PROCESSING REVENUES Earned for authorization, '
                                      'clearing, settlement; value added services related to issuing, '
                                      'acceptance, and risk and identity solutions; network access; and '
                                      'other maintenance and support services that facilitate transaction '
                                      'and information processing among our clients globally CLIENT '
                                      'INCENTIVES',
                                      'The acquirer pays the amount of the purchase, minus the merchant '
                                      'discount rate (MDR), to the merchant. Visa earns revenue by '
                                      'facilitating money movement across more than 200 countries and '
                                      'territories among a global set of consumers, merchants, financial '
                                      'institutions and government entities through innovative technologies. '
                                      '5 Table of Contents Our net revenue in fiscal 2024 consisted of the '
                                      'following: SERVICE REVENUE Earned for services provided in support of '
                                      'client usage of Visa payment services OTHER REVENUE Consist mainly of '
                                      'value-added services related to advisory, marketing and certain card '
                                      'benefits; license fees for use of the Visa brand or technology; and '
                                      'fees for account holder services, certification and licensing DATA '
                                      'PROCESSING REVENUE Earned for authorization, clearing and settlement; '
                                      'value-added services related to issuing, acceptance, and risk and '
                                      'identity solutions; network access; and other maintenance and support '
                                      'services that facilitate transaction and information processing among '
                                      'our clients globally CLIENT INCENTIVES']},
                    {'id': 'revenue-model',
                     'text': 'In the context of Visa-branded card transactions on our network, we provide '
                             'authorization, clearing and settlement services and may earn service, data '
                             'processing, international transaction or other revenue.',
                     'content_fingerprint': 'cdc89b94c67d2b89f3ce',
                     'source_path': 'data/raw/visa/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['to our consolidated financial statements included in Item 8 of this '
                                      'report, which include disclosures on how we earn and recognize our '
                                      'revenue. Visa provides payment processing for both non-Visa-branded '
                                      'and Visa-branded card transactions. In the context of '
                                      'non-Visa-branded card transactions, we facilitate payment processing '
                                      'by providing gateway routing services to other payment networks. At '
                                      'the client’s request, we may provide authorization, clearing or '
                                      'settlement services on our network before or after we route the '
                                      'transaction to the other payments network. In those instances, Visa '
                                      'may earn data processing revenue for the specific services provided. '
                                      'In the context of Visa-branded card transactions on our network, we '
                                      'provide authorization, clearing and settlement services and may earn '
                                      'service, data processing, international transaction or other revenue. '
                                      'Depending on applicable regulations, some payment processors may or '
                                      'may not use our network to process Visa-branded card transactions. If '
                                      'they use our network, we may earn service revenue and data processing '
                                      'revenue. If they do not use our network, we earn only service '
                                      'revenue.',
                                      'included in Item 8 of this report, which include disclosures on how '
                                      'we earn and recognize our revenues. Visa provides payment processing '
                                      'for both non-Visa-branded and Visa-branded card transactions. In the '
                                      'context of non-Visa-branded card transactions, we facilitate payment '
                                      'processing by providing gateway routing services to other payment '
                                      'networks. At the client’s request, we may provide authorization, '
                                      'clearing or settlement services on our network before or after we '
                                      'route the transaction to the other payments network. In those '
                                      'instances, Visa may earn data processing revenues for the specific '
                                      'services provided. In the context of Visa-branded card transactions '
                                      'on our network, we provide authorization, clearing and settlement '
                                      'services and may earn service, data processing, international '
                                      'transaction, or other revenues. Depending on applicable regulations, '
                                      'some payment processors may or may not use our network to process '
                                      'Visa-branded card transactions. If they use our network, we may earn '
                                      'service revenues and data processing revenues. If they do not use our '
                                      'network, we earn only service revenues.']}],
  'expected_facts': [{'id': 'money-movement',
                      'description': 'Core business facilitates money movement/payment processing',
                      'patterns': ['facilitating money movement', 'payment processing'],
                      'regex_patterns': ['(?:facilitat\\w+\\s+money\\s+movement|payment\\s+processing)']},
                     {'id': 'processing-services',
                      'description': 'Provides authorization, clearing and settlement',
                      'patterns': ['authorization, clearing and settlement'],
                      'regex_patterns': ['authorization.*clearing.*settlement']},
                     {'id': 'revenue-streams',
                      'description': 'Revenue includes service, data processing and international '
                                     'transaction revenue',
                      'patterns': [],
                      'regex_patterns': ['service\\s+revenue.*data\\s+processing.*international\\s+transaction|service.*data\\s+processing.*international\\s+transaction']}]},
 {'id': 'risk-01',
  'question': 'What supply chain risks does Nvidia disclose in its 10-K?',
  'expected_company': 'nvidia',
  'expected_section': 'risk_factors',
  'category': 'risk_factors',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'third-party-dependency',
                     'text': 'Dependency on third-party suppliers and their technology to manufacture, '
                             'assemble, test, or package our products reduces our control over product '
                             'quantity and quality, manufacturing yields, and product delivery schedules and '
                             'could harm our business.',
                     'content_fingerprint': '0e492fb3134f9cd1fcb3',
                     'source_path': 'data/raw/nvidia/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['Additionally, we depend on developers and other third parties to '
                                      'build accelerated computing applications that leverage our platforms. '
                                      'We also rely on third-party content providers and publishers to make '
                                      'their content available on our platforms such as GeForce NOW. Failure '
                                      'by developers to build applications that leverage our platforms, or '
                                      'failure by third-party content providers or publishers to make their '
                                      'content available on reasonable terms or at all for use by our '
                                      'customers or end users on our platforms, could adversely affect '
                                      'customer demand. Dependency on third-party suppliers and their '
                                      'technology to manufacture, assemble, test, package or design our '
                                      'products reduces our control over product quantity and quality, '
                                      'manufacturing yields, development, enhancement and product delivery '
                                      'schedules and could harm our business.',
                                      'Dependency on third-party suppliers and their technology to '
                                      'manufacture, assemble, test, or package our products reduces our '
                                      'control over product quantity and quality, manufacturing yields, and '
                                      'product delivery schedules and could harm our business. We depend on '
                                      'foundries to manufacture our semiconductor wafers using their '
                                      'fabrication equipment and techniques. We do not assemble, test, or '
                                      'package our products, but instead contract with independent '
                                      'subcontractors. These subcontractors assist with procuring components '
                                      'used in our systems, boards, and products. We face several risks '
                                      'which have adversely affected or could adversely affect our ability '
                                      'to meet customer demand and scale our supply chain, negatively impact '
                                      'longer-term demand for our products and services, and adversely '
                                      'affect our business operations, gross margin, revenue and/or '
                                      'financial results, including: • lack of guaranteed supply of wafer, '
                                      'component and capacity or decommitment and potential higher wafer and '
                                      'component prices, from incorrectly estimating demand and failing to '
                                      'place orders with our suppliers with sufficient quantities or in a '
                                      'timely manner; •']},
                    {'id': 'capacity-concentration',
                     'text': '• failure by our foundries or contract manufacturers to procure raw materials '
                             'or provide adequate levels of manufacturing or test capacity for our products; '
                             '• failure by our foundries to develop, obtain, or successfully implement high '
                             'quality process technologies, including transitions to smaller geometry '
                             'process technologies such as advanced process node technologies and memory '
                             'designs needed to manufacture our products; • failure by our suppliers to '
                             'comply with our policies and expectations and emerging regulatory '
                             'requirements; • limited number and geographic concentration of global '
                             'suppliers, foundries, contract manufacturers, assembly and test providers and '
                             'memory manufacturers;',
                     'content_fingerprint': '8d1956bd40ceb7aa77e1',
                     'source_path': 'data/raw/nvidia/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['• failure by our foundries or contract manufacturers to procure raw '
                                      'materials or provide adequate levels of manufacturing or test '
                                      'capacity for our products; • failure by our foundries to develop, '
                                      'obtain or successfully implement high quality process technologies, '
                                      'including transitions to smaller geometry process technologies such '
                                      'as advanced process node technologies and memory designs needed to '
                                      'manufacture our products; • failure by our suppliers to comply with '
                                      'our policies and expectations and emerging regulatory requirements; • '
                                      'limited number and geographic concentration of global suppliers, '
                                      'foundries, contract manufacturers, assembly and test providers and '
                                      'memory manufacturers; • loss of a supplier and additional expense '
                                      'and/or production delays as a result of qualifying a new foundry or '
                                      'subcontractor and commencing volume production or testing in the '
                                      'event of a loss, addition or change of a supplier; • lack of direct '
                                      'control over product quantity, quality and delivery schedules; • '
                                      'suppliers or their suppliers failing to supply high quality products '
                                      'and/or making changes to their products without our qualification; •',
                                      '20 • failure by our foundries or contract manufacturers to procure '
                                      'raw materials or to provide adequate levels of manufacturing or test '
                                      'capacity for our products; • failure by our foundries to develop, '
                                      'obtain or successfully implement high quality process technologies, '
                                      'including transitions to smaller geometry process technologies such '
                                      'as advanced process node technologies and memory designs needed to '
                                      'manufacture our products; • limited number and geographic '
                                      'concentration of global suppliers, foundries, contract manufacturers, '
                                      'assembly and test providers, and memory manufacturers; • loss of a '
                                      'supplier and additional expense and/or production delays as a result '
                                      'of qualifying a new foundry or subcontractor and commencing volume '
                                      'production or testing in the event of a loss of or a decision to add '
                                      'or change a supplier; • lack of direct control over product quantity, '
                                      'quality and delivery schedules; • suppliers or their suppliers '
                                      'failing to supply high quality products and/or making changes to '
                                      'their products without our qualification; •']}],
  'expected_facts': [{'id': 'supplier-dependency',
                      'description': 'Third-party supplier dependence reduces control over '
                                     'quality/quantity/delivery',
                      'patterns': [],
                      'regex_patterns': ['third[- '
                                         ']party\\s+suppliers?.*(?:quality|quantity|delivery|yield)|(?:quality|quantity|delivery|yield).*third[- '
                                         ']party\\s+suppliers?']},
                     {'id': 'capacity-risk',
                      'description': 'Raw material/manufacturing capacity shortages',
                      'patterns': [],
                      'regex_patterns': ['(?:raw '
                                         'materials?|manufacturing|test)\\s+capacity|lack\\s+of\\s+(?:guaranteed\\s+)?supply|capacity\\s+shortage']},
                     {'id': 'supplier-concentration',
                      'description': 'Limited/geographically concentrated suppliers or supplier loss',
                      'patterns': [],
                      'regex_patterns': ['(?:limited|concentrat\\w+).*(?:suppliers?|foundries)|(?:loss|failure)\\s+of\\s+(?:a\\s+)?supplier']}]},
 {'id': 'risk-02',
  'question': 'What regulatory risks does Apple highlight related to its global operations?',
  'expected_company': 'apple',
  'expected_section': 'risk_factors',
  'category': 'risk_factors',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'global-regulation',
                     'text': 'The Company’s global operations are subject to complex and changing laws and '
                             'regulations worldwide on subjects including antitrust; privacy, data security '
                             'and data localization; online safety; age verification; consumer protection; '
                             'advertising, sales, billing and e-commerce; financial services and technology; '
                             'product liability; intellectual property ownership and infringement; digital '
                             'platforms; machine learning and artificial intelligence; internet, '
                             'telecommunications and mobile communications; media, television, film and '
                             'digital content; availability of third-party software applications and '
                             'services; labor and employment; anticorruption; import, export and trade; '
                             'foreign exchange controls and cash repatriation restrictions; anti–money '
                             'laundering; foreign ownership and investment; national security; tax; and '
                             'environmental, health and safety, including electronic waste, recycling, '
                             'product design and climate change.',
                     'content_fingerprint': '03e1a26983487cbc0361',
                     'source_path': 'data/raw/apple/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['Apple Inc. | 2023 Form 10-K | 12 The Company is subject to complex '
                                      'and changing laws and regulations worldwide, which exposes the '
                                      'Company to potential liabilities, increased costs and other adverse '
                                      'effects on the Company’s business. The Company’s global operations '
                                      'are subject to complex and changing laws and regulations on subjects, '
                                      'including antitrust; privacy, data security and data localization; '
                                      'consumer protection; advertising, sales, billing and e-commerce; '
                                      'financial services and technology; product liability; intellectual '
                                      'property ownership and infringement; digital platforms; machine '
                                      'learning and artificial intelligence; internet, telecommunications '
                                      'and mobile communications; media, television, film and digital '
                                      'content; availability of third-party software applications and '
                                      'services; labor and employment; anticorruption; import, export and '
                                      'trade; foreign exchange controls and cash repatriation restrictions; '
                                      'anti–money laundering; foreign ownership and investment; tax; and '
                                      'environmental, health and safety, including electronic waste, '
                                      'recycling, product design and climate change.',
                                      'The Company is subject to complex and changing laws and regulations '
                                      'worldwide, which exposes the Company to potential liabilities, '
                                      'increased costs and other adverse effects on the Company’s business. '
                                      'The Company’s global operations are subject to complex and changing '
                                      'laws and regulations on subjects, including antitrust; privacy, data '
                                      'security and data localization; consumer protection; advertising, '
                                      'sales, billing and e-commerce; financial services and technology; '
                                      'product liability; intellectual property ownership and infringement; '
                                      'digital platforms; machine learning and artificial intelligence; '
                                      'internet, telecommunications and mobile communications; media, '
                                      'television, film and digital content; availability of third-party '
                                      'software applications and services; labor and employment; '
                                      'anticorruption; import, export and trade; foreign exchange controls '
                                      'and cash repatriation restrictions; anti–money laundering; foreign '
                                      'ownership and investment; tax; and environmental, health and safety, '
                                      'including electronic waste, recycling, product design and climate '
                                      'change.']},
                    {'id': 'regulatory-impact',
                     'text': 'Such changes in business practices can also otherwise adversely affect the '
                             'experience for users of the Company’s products and services, and result in '
                             'harm to the Company’s reputation, loss of competitive advantage, poor market '
                             'acceptance, reduced demand for products and services, lost sales, and lower '
                             'profit margins.',
                     'content_fingerprint': '0bf8903de89667db506a',
                     'source_path': 'data/raw/apple/2025_10k.html',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'broad-global-laws',
                      'description': 'Global operations face complex changing laws including antitrust, '
                                     'privacy and trade',
                      'patterns': [],
                      'regex_patterns': ['(?:complex|changing).*(?:laws|regulations).*(?:antitrust|privacy).*(?:trade|import|export)|(?:antitrust|privacy).*(?:trade|import|export).*(?:laws|regulations)']},
                     {'id': 'business-impact',
                      'description': 'Regulatory changes can hurt reputation/competitive '
                                     'position/demand/margins',
                      'patterns': [],
                      'regex_patterns': ['(?:regulat\\w+|laws?).*(?:reputation|competitive|demand|sales|margin)|(?:reputation|competitive|demand|sales|margin).*(?:regulat\\w+|laws?)']}]},
 {
    "id": "risk-03",
    "question": "What manufacturing and production risks does Tesla discuss?",
    "expected_company": "tesla",
    "expected_section": "risk_factors",
    "category": "risk_factors",
    "answerable": True,
    "gold_fiscal_year": 2025,

    "gold_evidence": [
        {
            "id": "production-ramp",
            "text": (
                "We may experience issues or delays in developing, launching "
                "and ramping the production of our products, services and "
                "features, or we may be unable to control our manufacturing costs."
            ),
            "alternatives": [
                (
                    "Any delay or other complication in ramping the production "
                    "of our current products or the development, manufacture, "
                    "launch and production ramp of our future products, features "
                    "and services, or in doing so cost-effectively and with high "
                    "quality, may harm our brand, business, prospects, financial "
                    "condition and operating results."
                )
            ],
            "source_path": "data/raw/tesla/2025_10k.html",
            "source_fiscal_year": 2025,
        },
        {
            "id": "gigafactory-components",
            "text": (
                "We may experience issues with lithium-ion cells or other "
                "components manufactured at our Gigafactories, which may harm "
                "the production and profitability of our vehicle and energy "
                "storage products."
            ),
            "source_path": "data/raw/tesla/2025_10k.html",
            "source_fiscal_year": 2025,
        },
    ],

    "expected_facts": [
        {
            "id": "ramp-delays",
            "description": (
                "Developing, launching, or ramping production can face delays"
            ),
            "patterns": [],
            "regex_patterns": [
                (
                    r"(?:issues|delays?).*"
                    r"(?:ramp(?:ing)?(?:\s+up)?\s+(?:the\s+)?production)"
                    r"|"
                    r"(?:ramp(?:ing)?(?:\s+up)?\s+(?:the\s+)?production).*"
                    r"(?:issues|delays?)"
                )
            ],
        },
        {
            "id": "gigafactory-components",
            "description": (
                "Lithium-ion cell/component issues at Gigafactories can harm "
                "production or profitability"
            ),
            "patterns": [],
            "regex_patterns": [
                (
                    r"(?:lithium[-\s]?ion|components?).*"
                    r"(?:Gigafactor(?:y|ies)|production|profitability)"
                    r"|"
                    r"(?:Gigafactor(?:y|ies)).*"
                    r"(?:lithium[-\s]?ion|components?|production|profitability)"
                )
            ],
        },
    ],

    "annotation_note": (
        "FY2025 replaces the prior FY2024 supplier-unavailability label with "
        "Tesla's current Gigafactory cell/component manufacturing-risk disclosure."
    ),
},
 {'id': 'risk-04',
  'question': 'What environmental and regulatory risks does ExxonMobil disclose?',
  'expected_company': 'exxonmobil',
  'expected_section': 'risk_factors',
  'category': 'risk_factors',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'environmental-laws',
                     'text': '• changes in environmental regulations or other laws that penalize us for past '
                             'or current production of legal and/or permitted products and operations, '
                             'increase our cost of operation or compliance or reduce or delay available '
                             'business opportunities, including changes in laws affecting offshore drilling '
                             'operations, standards to complete decommissioning, water use, production of '
                             'our products, emissions, hydraulic fracturing, or production or use of new or '
                             'recycled plastics, as well as laws and regulations affecting trading, carbon '
                             'capture and storage, hydrogen, lower-emission fuels, Proxxima TM systems, '
                             'carbon materials, or lithium;',
                     'content_fingerprint': '98048257bd825f3d3b48',
                     'source_path': 'data/raw/exxonmobil/2025_10k.html',
                     'source_fiscal_year': 2025},
                    {'id': 'climate-frameworks',
                     'text': 'Driven by concern over the risks of climate change, a number of countries have '
                             'adopted, or are considering the adoption of, regulatory frameworks to report '
                             'on or reduce greenhouse gas emissions, including emissions from the production '
                             'and use of oil and gas and their products, as well as increase the use of or '
                             'support for different emission-reduction technologies.',
                     'content_fingerprint': '8a143429eda3c123a21d',
                     'source_path': 'data/raw/exxonmobil/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['Driven by concern over the risks of climate change, a number of '
                                      'countries have adopted, or are considering the adoption of, '
                                      'regulatory frameworks to reduce greenhouse gas emissions including '
                                      'emissions from the production and use of oil and gas and their '
                                      'products as well as the use or support for different '
                                      'emission-reduction technologies. These actions are being taken both '
                                      'independently by national and regional governments and within the '
                                      'framework of United Nations Conference of the Parties summits under '
                                      'which many countries of the world have endorsed objectives to reduce '
                                      'the atmospheric concentration of carbon dioxide (CO2) over the coming '
                                      'decades, with an ambition ultimately to achieve “net zero”. Net zero '
                                      'means that emissions of greenhouse gases from human activities would '
                                      'be balanced by actions that remove such gases from the atmosphere. '
                                      'Expectations for transition of the world’s energy system to '
                                      'lower-emission sources, and ultimately net-zero, derive from '
                                      'hypothetical scenarios that reflect many assumptions about the future '
                                      'and reflect substantial uncertainties. The company’s objective to '
                                      'play a leading role in the energy transition, including the company’s '
                                      'announced ambition']}],
  'expected_facts': [{'id': 'compliance-cost-opportunities',
                      'description': 'Environmental rules can raise costs or reduce/delay opportunities',
                      'patterns': [],
                      'regex_patterns': ['environmental\\s+(?:regulations?|laws?).*(?:cost|compliance|reduce|delay).*(?:opportunit|operations?)|(?:cost|compliance).*(?:environmental\\s+(?:regulations?|laws?))']},
                     {'id': 'ghg-regulation',
                      'description': 'Climate regulation targets greenhouse gas emissions',
                      'patterns': ['greenhouse gas emissions'],
                      'regex_patterns': ['(?:climate|greenhouse\\s+gas).*(?:regulat\\w+|framework|emissions)']}]},
 {'id': 'risk-05',
  'question': 'What risks does Meta describe related to data privacy and regulation?',
  'expected_company': 'meta',
  'expected_section': 'risk_factors',
  'category': 'risk_factors',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'privacy-laws',
                     'text': '• complex and evolving U.S. and foreign privacy, data use, data combination, '
                             'data protection, content and content moderation, competition, youth, safety, '
                             'consumer protection, advertising, and other laws and regulations, including '
                             'the General Data Protection Regulation (GDPR), Digital Markets Act (DMA), '
                             'Digital Services Act (DSA), Artificial Intelligence Act (EU AI Act), and the '
                             'UK Digital Markets, Competition and Consumer Act (DMCC);',
                     'content_fingerprint': '6972b2a360794f2d9805',
                     'source_path': 'data/raw/meta/2025_10k.html',
                     'source_fiscal_year': 2025},
                    {'id': 'noncompliance-consequences',
                     'text': 'If we are unable to successfully implement and comply with the mandates of the '
                             'FTC consent order (including any future modifications to the order), GDPR, '
                             'U.S. state privacy laws, youth social media laws, ePrivacy Directive, DMA, '
                             'DSA, or other regulatory or legislative requirements, or if any relevant '
                             'authority believes that we are in violation of the consent order or other '
                             'applicable requirements, we may be subject to regulatory or governmental '
                             'investigations or lawsuits, which may result in significant monetary fines or '
                             'damages (including for loss of control of data without other damage), '
                             'judgments, penalties, or other remedies, and we may also be required to make '
                             'additional changes to our business practices.',
                     'content_fingerprint': '004bdd3139f9bff42e92',
                     'source_path': 'data/raw/meta/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['If we are unable to successfully implement and comply with the '
                                      'mandates of the FTC consent order (including any future modifications '
                                      'to the order), GDPR, U.S. state privacy laws, including the CCPA, '
                                      'ePrivacy Directive, DMA, DSA, or other regulatory or legislative '
                                      'requirements, or if any relevant authority believes that we are in '
                                      'violation of the consent order or other applicable requirements, we '
                                      'may be subject to regulatory or governmental investigations or '
                                      'lawsuits, which may result in significant monetary fines, judgments, '
                                      'penalties, or other remedies, and we may also be required to make '
                                      'additional changes to our business practices. Any of these events '
                                      'could have a material adverse effect on our business, reputation, and '
                                      'financial results. 43 Table of Contents We may incur liability as a '
                                      'result of information retrieved from or transmitted over the internet '
                                      'or published using our products or as a result of claims related to '
                                      'our products, and legislation regulating content on our platform may '
                                      'require us to change our products or business practices and may '
                                      'adversely affect our business and financial results.']}],
  'expected_facts': [{'id': 'privacy-regulation',
                      'description': 'Complex evolving privacy/data regulation including GDPR',
                      'patterns': ['GDPR'],
                      'regex_patterns': ['(?:privacy|data\\s+protection).*(?:GDPR|DMA|DSA|laws?|regulations?)']},
                     {'id': 'penalties-business-changes',
                      'description': 'Noncompliance can lead to investigations/fines/penalties and '
                                     'business-practice changes',
                      'patterns': [],
                      'regex_patterns': ['(?:noncompliance|violation|comply).*(?:investigation|lawsuit|fine|penalt|business\\s+practices)|(?:fine|penalt|investigation).*(?:privacy|GDPR|regulat\\w+)']}]},
 {'id': 'risk-06',
  'question': 'What risk factors does Pfizer disclose related to drug development and clinical trials?',
  'expected_company': 'pfizer',
  'expected_section': 'risk_factors',
  'category': 'risk_factors',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'clinical-development-risks',
                     'text': '• We may have difficulties recruiting and enrolling patients for clinical '
                             'trials on a consistent basis. • Product candidates can and do fail at any '
                             'stage of the process, including as the result of unfavorable pre-clinical and '
                             'clinical trial results, or unfavorable new pre-clinical or clinical data and '
                             'further analyses of existing pre-clinical or clinical data, including results '
                             'that may not support further clinical development of the product candidate or '
                             'indication. • We may need to amend our clinical trial protocols or conduct '
                             'additional clinical trials under certain circumstances, for example, to '
                             'further assess appropriate dosage or collect additional safety data.',
                     'content_fingerprint': '8e5bd22c430a83a658fe',
                     'source_path': 'data/raw/pfizer/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['• We may have difficulties recruiting and enrolling patients for '
                                      'clinical trials on a consistent basis. • Product candidates can and '
                                      'do fail at any stage of the process, including as the result of '
                                      'unfavorable pre-clinical and clinical trial results, or unfavorable '
                                      'new pre-clinical or clinical data and further analyses of existing '
                                      'pre-clinical or clinical data, including results that may not support '
                                      'further clinical development of the product candidate or indication. '
                                      '• We may need to amend our clinical trial protocols or conduct '
                                      'additional clinical trials under certain circumstances, for example, '
                                      'to further assess appropriate dosage or collect additional safety '
                                      'data. • We may not be able to meet anticipated pre-clinical or '
                                      'clinical endpoints, commencement and/or completion dates for our '
                                      'pre-clinical or clinical trials, regulatory submission dates, '
                                      'regulatory approval dates and/or launch dates. • We may not be able '
                                      'to successfully address all the comments received from regulatory '
                                      'authorities such as the FDA and the EMA, or be able to obtain '
                                      'approval for new products and indications from regulators.']},
                    {'id': 'regulatory-approval-risk',
                     'text': 'Regulatory approvals of our products depend on myriad factors, including '
                             'regulatory determinations as to the product’s safety and efficacy.',
                     'content_fingerprint': 'aa9999884ff4e8b2d39d',
                     'source_path': 'data/raw/pfizer/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['Regulatory approvals of our products depend on myriad factors, '
                                      'including regulatory determinations as to the product’s safety and '
                                      'efficacy. In the context of public health emergencies like the '
                                      'COVID-19 pandemic, regulators evaluate various factors and criteria '
                                      'to potentially allow for marketing authorization on an emergency or '
                                      'conditional basis. Additionally, clinical trial and other product '
                                      'data are subject to differing interpretations and assessments by '
                                      'regulatory authorities. As a result of regulatory interpretations and '
                                      'assessments or other developments that occur during the review '
                                      'process, and even after a product is authorized or approved for '
                                      'marketing, a product’s commercial potential could be adversely '
                                      'affected by potential emerging concerns or regulatory decisions '
                                      'regarding or impacting labeling or marketing, manufacturing '
                                      'processes, safety and/or other matters, including decisions relating '
                                      'to emerging developments regarding potential product impurities. '
                                      'Pfizer Inc. 2022 Form 10-K 17',
                                      'Regulatory approvals of our products depend on myriad factors, '
                                      'including regulatory determinations as to the product’s safety and '
                                      'efficacy. In the context of public health emergencies like the '
                                      'COVID-19 pandemic, regulators evaluate various factors and criteria '
                                      'to potentially allow for marketing authorization on an emergency or '
                                      'conditional basis. Additionally, clinical trial and other product '
                                      'data are subject to differing interpretations and assessments by '
                                      'regulatory authorities. As a result of regulatory interpretations and '
                                      'assessments or other developments that may occur during the review '
                                      'process, or even after a product is authorized or approved for '
                                      'marketing, a product’s commercial potential could be adversely '
                                      'affected by potential emerging concerns or regulatory decisions '
                                      'regarding or impacting the scope of indicated patient populations, '
                                      'labeling or marketing, manufacturing processes, safety issues and/or '
                                      'other matters, including decisions relating to emerging developments '
                                      'regarding potential product impurities. Also, certain of our products '
                                      'have received and may in the future receive approvals under '
                                      'accelerated approval pathways where continued approval may be '
                                      'contingent upon']}],
  'expected_facts': [{'id': 'recruitment',
                      'description': 'Patient recruitment/enrollment difficulties',
                      'patterns': [],
                      'regex_patterns': ['(?:recruit|enroll)\\w*.*patients?.*clinical\\s+trials?|clinical\\s+trials?.*(?:recruit|enroll)']},
                     {'id': 'candidate-failure',
                      'description': 'Candidates may fail based on unfavorable trial results/data',
                      'patterns': [],
                      'regex_patterns': ['(?:product\\s+)?candidates?.*(?:fail|failure).*(?:clinical|trial|data)|(?:clinical|trial)\\s+(?:results?|data).*(?:fail|failure)']},
                     {'id': 'safety-efficacy-approval',
                      'description': 'Approval depends on safety/efficacy and may require additional trials',
                      'patterns': [],
                      'regex_patterns': ['(?:approval|regulat\\w+).*(?:safety|efficacy)|(?:additional|amend).*(?:clinical\\s+trials?|protocols?)']}]},
 {'id': 'risk-07',
  'question': 'What risk factors does General Electric disclose related to its industrial operations?',
  'expected_company': 'general_electric',
  'expected_section': 'risk_factors',
  'category': 'risk_factors',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'operational-risk-categories',
                     'text': 'Operational risk relates to risks arising from systems, processes, people and '
                             'external events that affect the operation of our business. It includes risks '
                             'related to product safety, quality and performance; supply chain and business '
                             'disruption; operational execution across product and service life cycles; and '
                             'information management and data protection and security, including '
                             'cybersecurity.',
                     'content_fingerprint': 'ed4b0d69475dc601b5f7',
                     'source_path': 'data/raw/general_electric/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['2023 FORM 10-K 31 OPERATIONAL RISKS. Operational risk relates to '
                                      'risks arising from systems, processes, people and external events '
                                      'that affect the operation of our businesses. It includes risks '
                                      'related to product and service life cycle and execution; product '
                                      'safety and performance; information management and data protection '
                                      'and security, including cybersecurity; and supply chain and business '
                                      'disruption. Operational execution - Operational challenges could have '
                                      'a material adverse effect on our business, reputation, financial '
                                      'position, results of operations and cash flows.',
                                      'OPERATIONAL RISKS. Operational risk relates to risks arising from '
                                      'systems, processes, people and external events that affect the '
                                      'operation of our businesses. It includes risks related to product and '
                                      'service lifecycle and execution; product safety and performance; '
                                      'information management and data protection and security, including '
                                      'cybersecurity; and supply chain and business disruption. Operational '
                                      'execution - Operational challenges could have a material adverse '
                                      'effect on our business, reputation, financial position, results of '
                                      'operations and cash flows.']},
                    {'id': 'supplier-production-risk',
                     'text': 'In addition, some of our suppliers or their sub-suppliers are limited- or '
                             'sole-source suppliers, and our ability to meet our obligations to customers '
                             'depends on the performance, product quality, continued product availability '
                             'and stability of such suppliers.',
                     'content_fingerprint': '8caad6877635f409ce75',
                     'source_path': 'data/raw/general_electric/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['operations and financial performance for some period of time. For '
                                      'example, successfully executing the significant production ramp-up '
                                      'efforts at our Aerospace business in connection with both newer '
                                      'engine platforms such as the LEAP and the aviation sector’s ongoing '
                                      'recovery from the COVID-19 pandemic, depends in part on our suppliers '
                                      'having access to the materials and skilled labor they require and '
                                      'making timely deliveries to us, as well as meeting the required '
                                      'quality and performance standards for commercial aviation. In '
                                      'addition, some of our suppliers or their sub-suppliers are limited- '
                                      'or sole-source suppliers, and our ability to meet our obligations to '
                                      'customers depends on the performance, product quality and stability '
                                      'of such suppliers. We also have internal dependencies on certain key '
                                      'GE manufacturing or other facilities. Disruptions in deliveries, '
                                      'capacity constraints, production disruptions up- or down-stream, '
                                      'price increases, or decreased availability of raw materials or '
                                      'commodities, including as a result of war, natural disasters '
                                      '(including the effects of climate change such as sea level rise, '
                                      'drought, flooding, wildfires and more intense weather events), actual '
                                      'or',
                                      'us, as well as meeting the required quality and performance standards '
                                      'for commercial aviation. In addition, some of our suppliers or their '
                                      'sub-suppliers are limited- or sole-source suppliers, and our ability '
                                      'to meet our obligations to customers depends on the performance, '
                                      'product quality and stability of such suppliers. We also have '
                                      'internal dependencies on certain key GE manufacturing or other '
                                      'facilities. Disruptions in deliveries, capacity constraints, '
                                      'production disruptions up- or down-stream, price increases, or '
                                      'decreased availability of raw materials or commodities, including as '
                                      'a result of war, natural disasters (including the effects of climate '
                                      'change such as sea level rise, drought, flooding, wildfires and more '
                                      'intense weather events), actual or threatened public health pandemics '
                                      'or emergencies or other business continuity events, adversely affect '
                                      'our operations and, depending on the length and severity of the '
                                      'disruption, can limit our ability to meet our commitments to '
                                      'customers or significantly impact our operating profit or cash flows. '
                                      'Quality, capability, compliance and sourcing issues experienced by '
                                      'third-party providers can also adversely affect our costs, margin']}],
  'expected_facts': [{'id': 'safety-quality',
                      'description': 'Product safety/quality/performance risks',
                      'patterns': [],
                      'regex_patterns': ['product\\s+(?:safety|quality).*(?:quality|performance)|(?:safety|quality|performance).*(?:product)']},
                     {'id': 'supply-disruption',
                      'description': 'Supply-chain/business disruption risk',
                      'patterns': ['supply chain', 'business disruption'],
                      'regex_patterns': ['supply\\s+chain.*(?:disruption|supplier)|(?:disruption|supplier).*supply\\s+chain']},
                     {'id': 'sole-source',
                      'description': 'Limited/sole-source supplier dependency',
                      'patterns': [],
                      'regex_patterns': ['(?:limited|sole)[- '
                                         ']source\\s+suppliers?|supplier.*(?:capacity|availability|stability)']}]},
 {'id': 'mdna-01',
  'question': "What does Microsoft's MD&A say about revenue growth drivers?",
  'expected_company': 'microsoft',
  'expected_section': 'mdna',
  'category': 'mdna',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'revenue-growth-drivers',
                     'text': 'Revenue increased $36.6 billion or 15% with growth across each of our '
                             'segments. Intelligent Cloud revenue increased driven by Azure. Productivity '
                             'and Business Processes revenue increased driven by Microsoft 365 Commercial '
                             'cloud. More Personal Computing revenue increased driven by Gaming and Search '
                             'and news advertising.',
                     'content_fingerprint': '0e3188ce968a4ac99ed8',
                     'source_path': 'data/raw/microsoft/2025_10k.html',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'overall-growth',
                      'description': 'Revenue increased $36.6B / 15%',
                      'patterns': ['$36.6 billion', '15%'],
                      'regex_patterns': ['(?:revenue.*(?:36\\.6\\s*billion|15\\s*%)|(?:36\\.6\\s*billion|15\\s*%).*revenue)']},
                     {'id': 'azure-driver',
                      'description': 'Intelligent Cloud growth driven by Azure',
                      'patterns': ['Azure'],
                      'regex_patterns': ['Intelligent\\s+Cloud.*Azure|Azure.*Intelligent\\s+Cloud']},
                     {'id': 'm365-driver',
                      'description': 'Productivity and Business Processes growth driven by Microsoft 365 '
                                     'Commercial cloud',
                      'patterns': ['Microsoft 365 Commercial cloud'],
                      'regex_patterns': ['(?:Productivity.*Business\\s+Processes|PBP).*(?:Microsoft\\s*365|Commercial\\s+cloud)|Microsoft\\s*365.*(?:Productivity.*Business\\s+Processes|PBP)']},
                     {'id': 'mpc-driver',
                      'description': 'More Personal Computing growth driven by Gaming and Search/news '
                                     'advertising',
                      'patterns': [],
                      'regex_patterns': ['More\\s+Personal\\s+Computing.*(?:Gaming|Search).*advertising|(?:Gaming|Search).*advertising.*More\\s+Personal\\s+Computing']}]},
 {'id': 'mdna-02',
  'question': "What does Amazon's MD&A discuss about operating margin trends?",
  'expected_company': 'amazon',
  'expected_section': 'mdna',
  'category': 'mdna',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'annotation_note': 'The filing passage directly discusses operating income rather than an explicit '
                     'operating-margin percentage; this item should be interpreted as '
                     'operating-profitability trend evidence.',
  'gold_evidence': [{'id': 'operating-income-trend',
                     'text': 'Operating income was $36.9 billion and $68.6 billion for 2023 and 2024.',
                     'content_fingerprint': '9105b3d14efc02ab5ff7',
                     'source_path': 'data/sections/amazon/2025/mdna.txt',
                     'source_fiscal_year': 2025},
                    {'id': 'segment-drivers',
                     'text': 'The increase in North America operating income in 2024, compared to the prior '
                             'year, is primarily due to increased unit sales and increased advertising '
                             'sales, partially offset by increased fulfillment and shipping costs. The '
                             'International operating income in 2024, as compared to the operating loss in '
                             'the prior year, is primarily due to increased unit sales and increased '
                             'advertising sales, partially offset by increased shipping and fulfillment '
                             'costs.',
                     'content_fingerprint': '9105b3d14efc02ab5ff7',
                     'source_path': 'data/sections/amazon/2025/mdna.txt',
                     'source_fiscal_year': 2025,
                     'alternatives': ['The North America operating income in 2023, as compared to the '
                                      'operating loss in the prior year, is primarily due to increased unit '
                                      'sales and increased advertising sales, partially offset by increased '
                                      'shipping and fulfillment costs and increased technology and '
                                      'infrastructure costs. The decrease in International operating loss in '
                                      'absolute dollars in 2023, compared to the prior year, is primarily '
                                      'due to increased unit sales and increased advertising sales, '
                                      'partially offset by increased fulfillment and shipping costs and '
                                      'increased technology and infrastructure costs. Changes in foreign '
                                      'exchange rates positively impacted operating loss by $246 million in '
                                      '2023. The increase in AWS operating income in absolute dollars in '
                                      '2023, compared to the prior year, is primarily due to increased '
                                      'sales, partially offset by increased payroll and related expenses and '
                                      'spending on technology infrastructure, both of which were primarily '
                                      'driven by additional investments to support AWS business growth. '
                                      'Changes in foreign exchange rates positively impacted operating '
                                      'income by $220 million in 2023. 25 Operating Expenses Information '
                                      'about operating expenses is as follows (in millions):']}],
  'expected_facts': [{'id': 'operating-income-increase',
                      'description': 'Consolidated operating income rose from $36.9B to $68.6B',
                      'patterns': [],
                      'regex_patterns': ['(?:36\\.9|36,852).*(?:68\\.6|68,593)|(?:68\\.6|68,593).*(?:36\\.9|36,852)']},
                     {'id': 'north-america-driver',
                      'description': 'North America increase driven by unit and advertising sales, offset by '
                                     'fulfillment/shipping costs',
                      'patterns': [],
                      'regex_patterns': ['North\\s+America.*(?:unit\\s+sales).*(?:advertising).*(?:fulfillment|shipping)']},
                     {'id': 'international-turnaround',
                      'description': 'International moved from loss to income with similar sales/ad drivers',
                      'patterns': [],
                      'regex_patterns': ['International.*(?:operating\\s+loss|loss).*(?:operating\\s+income|income)|International.*(?:unit\\s+sales).*(?:advertising)']}]},
 {'id': 'mdna-03',
  'question': "What does AMD's MD&A say about revenue trends by segment?",
  'expected_company': 'amd',
  'expected_section': 'mdna',
  'category': 'mdna',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'segment-revenue-trends',
                     'text': 'Data Center net revenue of $16.6 billion increased by 32% compared to $12.6 '
                             'billion in 2024, primarily driven by strong demand for our 5th generation AMD '
                             'EPYC™ processors and AMD Instinct™ MI350 Series GPUs. Client and Gaming '
                             'segment net revenue of $14.6 billion in 2025 increased by 51% compared to $9.6 '
                             'billion in 2024, primarily driven by strong demand for our AMD Ryzen™ '
                             'processors, semi-custom game consoles SoCs and Radeon™ gaming GPUs. The '
                             'increase in annual net revenue was partially offset by a decrease in net '
                             'revenue in our Embedded segment. Embedded net revenue of $3.5 billion '
                             'decreased by 3% compared to net revenue of $3.6 billion in 2024, as certain '
                             'end market demand remained mixed.',
                     'content_fingerprint': 'dc2638fcbf15a468bbeb',
                     'source_path': 'data/raw/amd/2025_10k.html',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'data-center',
                      'description': 'Data Center revenue $16.6B, +32%',
                      'patterns': [],
                      'regex_patterns': ['Data\\s+Center.*16\\.6\\s*billion.*32\\s*%|Data\\s+Center.*32\\s*%.*16\\.6\\s*billion']},
                     {'id': 'client-gaming',
                      'description': 'Client and Gaming revenue $14.6B, +51%',
                      'patterns': [],
                      'regex_patterns': ['Client\\s+and\\s+Gaming.*14\\.6\\s*billion.*51\\s*%|Client\\s+and\\s+Gaming.*51\\s*%.*14\\.6\\s*billion']},
                     {'id': 'embedded',
                      'description': 'Embedded revenue $3.5B, down 3%',
                      'patterns': [],
                      'regex_patterns': ['Embedded.*3\\.5\\s*billion.*(?:decreas|down).*3\\s*%|Embedded.*(?:decreas|down).*3\\s*%.*3\\.5\\s*billion']}]},
 {'id': 'mdna-04',
  'question': "What does Broadcom's MD&A discuss regarding gross margin trends?",
  'expected_company': 'broadcom',
  'expected_section': 'mdna',
  'category': 'mdna',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'gross-margin-trend',
                     'text': 'Gross margin was $43,294 million for fiscal year 2025 compared to $32,509 '
                             'million for fiscal year 2024. The increase was primarily due to higher '
                             'software revenue and strong product demand for our AI-related semiconductor '
                             'solutions. As a percentage of net revenue, gross margin was 68% and 63% of net '
                             'revenue for the fiscal years 2025 and 2024, respectively. The increase was '
                             'primarily due to higher revenue impact on margin and higher infrastructure '
                             'software gross margin percentage, driven by an increase in license revenue and '
                             'lower infrastructure software labor costs following our integration of the '
                             'VMware business.',
                     'content_fingerprint': '39603ee600ac940a7151',
                     'source_path': 'data/sections/broadcom/2025/mdna.txt',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'gross-margin-dollars',
                      'description': 'Gross margin increased to $43.294B from $32.509B',
                      'patterns': [],
                      'regex_patterns': ['(?:43,294|43\\.294).*(?:32,509|32\\.509)|(?:32,509|32\\.509).*(?:43,294|43\\.294)']},
                     {'id': 'gross-margin-percent',
                      'description': 'Gross margin percentage rose to 68% from 63%',
                      'patterns': [],
                      'regex_patterns': ['68\\s*%.*63\\s*%|63\\s*%.*68\\s*%']},
                     {'id': 'drivers',
                      'description': 'Drivers included software revenue/AI semiconductor demand and higher '
                                     'infrastructure-software margin',
                      'patterns': [],
                      'regex_patterns': ['(?:software\\s+revenue|infrastructure\\s+software).*(?:AI|semiconductor|license|VMware)|(?:AI|semiconductor|license|VMware).*(?:software\\s+revenue|infrastructure\\s+software)']}]},
 {'id': 'mdna-05',
  'question': "What does Caterpillar's MD&A say about sales and revenue trends?",
  'expected_company': 'caterpillar',
  'expected_section': 'mdna',
  'category': 'mdna',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'sales-revenue-trend',
                     'text': 'Total sales and revenues for 2024 were $64.809 billion, a decrease of $2.251 '
                             'billion, or 3 percent, compared with $67.060 billion in 2023. The decrease was '
                             'primarily driven by lower sales volume of $3.543 billion, partially offset by '
                             'favorable price realization of $1.238 billion. The decrease in sales volume '
                             'was mainly driven by lower sales of equipment to end users. In addition, '
                             'changes in dealer inventories had an unfavorable impact to sales volume.',
                     'content_fingerprint': '999ff3a7c896abae2bef',
                     'source_path': 'data/sections/caterpillar/2025/mdna.txt',
                     'source_fiscal_year': 2025},
                    {'id': 'segment-direction',
                     'text': 'In the three primary segments, sales were lower in Construction Industries and '
                             'Resource Industries and higher in Energy & Transportation.',
                     'content_fingerprint': 'a1b6a96be3e406fe26ed',
                     'source_path': 'data/raw/caterpillar/2025_10k.html',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'total-decline',
                      'description': 'Sales/revenues $64.809B, down 3% from $67.060B',
                      'patterns': [],
                      'regex_patterns': ['64\\.809\\s*billion.*(?:3\\s*(?:percent|%)|67\\.060)|(?:3\\s*(?:percent|%)|67\\.060).*64\\.809\\s*billion']},
                     {'id': 'volume-price',
                      'description': 'Lower volume partly offset by favorable price realization',
                      'patterns': [],
                      'regex_patterns': ['lower\\s+sales\\s+volume.*(?:favorable\\s+)?price\\s+realization|price\\s+realization.*lower\\s+sales\\s+volume']},
                     {'id': 'segment-direction',
                      'description': 'Construction/Resource lower; Energy & Transportation higher',
                      'patterns': [],
                      'regex_patterns': ['Construction\\s+Industries.*Resource\\s+Industries.*(?:Energy\\s*(?:&|and)\\s*Transportation).*higher']}]},
 {'id': 'mdna-06',
  'question': "What does Goldman Sachs' MD&A discuss about net income drivers?",
  'expected_company': 'goldman_sachs',
  'expected_section': 'mdna',
  'category': 'mdna',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'net-earnings',
                     'text': 'We generated net earnings of $14.28 billion for 2024, compared with $8.52 '
                             'billion for 2023.',
                     'content_fingerprint': '9c0731357f09fcad4492',
                     'source_path': 'data/raw/goldman_sachs/2025_10k.html',
                     'source_fiscal_year': 2025},
                    {'id': 'net-revenue',
                     'text': 'Net revenues were $53.51 billion for 2024, 16% higher than 2023, primarily '
                             'reflecting higher net revenues in Global Banking & Markets and Asset & Wealth '
                             'Management.',
                     'content_fingerprint': '9c0731357f09fcad4492',
                     'source_path': 'data/raw/goldman_sachs/2025_10k.html',
                     'source_fiscal_year': 2025},
                    {'id': 'revenue-drivers',
                     'text': 'The increase in net revenues in Global Banking & Markets primarily reflected '
                             'higher net revenues in Equities, significantly higher Investment banking fees '
                             'and higher net revenues in Fixed Income, Currency and Commodities (FICC). The '
                             'increase in net revenues in Asset & Wealth Management primarily reflected '
                             'significantly higher net revenues in Equity investments and higher Management '
                             'and other fees.',
                     'content_fingerprint': '9c0731357f09fcad4492',
                     'source_path': 'data/raw/goldman_sachs/2025_10k.html',
                     'source_fiscal_year': 2025,
                     'alternatives': ['Executive Overview We generated net earnings of $8.52 billion for '
                                      '2023, compared with $11.26 billion for 2022. Diluted earnings per '
                                      'common share (EPS) was $22.87 for 2023, compared with $30.06 for '
                                      '2022. ROE was 7.5% for 2023, compared with 10.2% for 2022. Book value '
                                      'per common share was $313.56 as of December 2023, 3.3% higher '
                                      'compared with December 2022. Net revenues were $46.25 billion for '
                                      '2023, 2% lower than 2022, reflecting lower net revenues in Global '
                                      'Banking & Markets, largely offset by higher net revenues in Platform '
                                      'Solutions and Asset & Wealth Management. The decrease in net revenues '
                                      'in Global Banking & Markets, compared with a strong prior year, '
                                      'reflected lower net revenues in Fixed Income, Currency and '
                                      'Commodities (FICC) and lower Investment banking fees. The increase in '
                                      'net revenues in Platform Solutions reflected significantly higher net '
                                      'revenues in Consumer platforms. The increase in net revenues in Asset '
                                      '& Wealth Management primarily reflected higher Management and other '
                                      'fees.']}],
  'expected_facts': [{'id': 'net-earnings',
                      'description': 'Net earnings $14.28B vs $8.52B',
                      'patterns': [],
                      'regex_patterns': ['14\\.28\\s*billion.*8\\.52\\s*billion|8\\.52\\s*billion.*14\\.28\\s*billion']},
                     {'id': 'net-revenue-growth',
                      'description': 'Net revenues $53.51B, up 16%',
                      'patterns': [],
                      'regex_patterns': ['53\\.51\\s*billion.*16\\s*%|16\\s*%.*53\\.51\\s*billion']},
                     {'id': 'business-drivers',
                      'description': 'Higher GBM and AWM revenues drove growth',
                      'patterns': [],
                      'regex_patterns': ['(?:Global\\s+Banking\\s*(?:&|and)\\s*Markets|GBM).*(?:Asset\\s*(?:&|and)\\s*Wealth\\s+Management|AWM)|(?:Asset\\s*(?:&|and)\\s*Wealth\\s+Management|AWM).*(?:Global\\s+Banking\\s*(?:&|and)\\s*Markets|GBM)']}]},
 {'id': 'mdna-07',
  'question': "What does Alphabet's MD&A say about advertising revenue trends?",
  'expected_company': 'alphabet',
  'expected_section': 'mdna',
  'category': 'mdna',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'search-youtube-growth',
                     'text': 'Google Search & other revenues increased $23.1 billion from 2023 to 2024. The '
                             'overall growth was driven by interrelated factors including increases in '
                             'search queries resulting from growth in user adoption and usage on mobile '
                             'devices; growth in advertiser spending; and improvements we have made in ad '
                             'formats and delivery. YouTube ads YouTube ads revenues increased $4.6 billion '
                             'from 2023 to 2024. The growth was driven by our brand advertising products '
                             'followed by our direct response advertising products, both of which benefited '
                             'from increased spending by our advertisers.',
                     'content_fingerprint': 'd2840effea279857ce84',
                     'source_path': 'data/raw/alphabet/2025_10k.html',
                     'source_fiscal_year': 2025},
                    {'id': 'network-decline',
                     'text': 'Google Network revenues decreased $953 million from 2023 to 2024, primarily '
                             'driven by a decrease in Google Ad Manager and AdMob revenues. Additionally, '
                             'Google Network revenues were adversely affected by changes in foreign currency '
                             'exchange rates.',
                     'content_fingerprint': '35496032441da4abe082',
                     'source_path': 'data/raw/alphabet/2025_10k.html',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'search-growth',
                      'description': 'Search & other revenue +$23.1B',
                      'patterns': [],
                      'regex_patterns': ['Search\\s*(?:&|and)\\s*other.*23\\.1\\s*billion']},
                     {'id': 'youtube-growth',
                      'description': 'YouTube ads revenue +$4.6B',
                      'patterns': [],
                      'regex_patterns': ['YouTube\\s+ads?.*4\\.6\\s*billion']},
                     {'id': 'network-decline',
                      'description': 'Google Network revenue down $953M',
                      'patterns': [],
                      'regex_patterns': ['Google\\s+Network.*(?:decreas|down).*953\\s*(?:million|m)|953\\s*(?:million|m).*Google\\s+Network']},
                     {'id': 'ad-spend-driver',
                      'description': 'Advertiser spending was a growth driver',
                      'patterns': ['advertiser spending'],
                      'regex_patterns': ['advertis(?:er|ing)\\s+spend\\w*']}]},
 {'id': 'fin-01',
  'question': "What do Johnson & Johnson's financial statements show about total assets?",
  'expected_company': 'johnson_and_johnson',
  'expected_section': 'financial_statements',
  'category': 'financial_statements',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'total-assets',
                     'text': 'Total assets $ 180,104 167,558',
                     'content_fingerprint': '46f73c9f290bba4e8399',
                     'source_path': 'data/raw/johnson_and_johnson/2025_10k.html',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'latest-total-assets',
                      'description': 'Latest reported total assets are $180.104B (180,104 million)',
                      'patterns': ['180,104'],
                      'regex_patterns': ['\\$?\\s*180(?:\\.104|\\.1)\\s*billion']}]},
 {'id': 'fin-02',
  'question': "What does Walmart's balance sheet show regarding total liabilities?",
  'expected_company': 'walmart',
  'expected_section': 'financial_statements',
  'category': 'financial_statements',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'annotation_note': "The indexed consolidated balance-sheet chunk does not print a standalone 'Total "
                     "liabilities' subtotal. Gold completeness therefore uses directly reported liability "
                     'components and does not invent a derived subtotal.',
  'gold_evidence': [{'id': 'liability-components',
                     'text': 'Total current liabilities 96,584 92,415 Long-term debt 33,401 36,132 Long-term '
                             'operating lease obligations 12,825 12,943 Long-term finance lease obligations '
                             '5,923 5,709 Deferred income taxes and other 14,398 14,629',
                     'content_fingerprint': 'c9ad9c8aa0bc81f07c90',
                     'source_path': 'data/raw/walmart/2025_10k.html',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'current-liabilities',
                      'description': 'Total current liabilities $96.584B',
                      'patterns': ['96,584'],
                      'regex_patterns': ['\\$?\\s*96(?:\\.584|\\.6)\\s*billion']},
                     {'id': 'long-term-debt',
                      'description': 'Long-term debt $33.401B',
                      'patterns': ['33,401'],
                      'regex_patterns': ['\\$?\\s*33(?:\\.401|\\.4)\\s*billion']}]},
 {'id': 'fin-03',
  'question': "What do Visa's financial statements disclose about cash flow from operations?",
  'expected_company': 'visa',
  'expected_section': 'financial_statements',
  'category': 'financial_statements',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'operating-cash-flow',
                     'text': 'Net cash provided by (used in) operating activities 23,059 19,950 20,755',
                     'content_fingerprint': 'ddc74fc172355e814e49',
                     'source_path': 'data/raw/visa/2025_10k.html',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'latest-operating-cash-flow',
                      'description': 'Latest net cash provided by operating activities $23.059B',
                      'patterns': ['23,059'],
                      'regex_patterns': ['\\$?\\s*23(?:\\.059|\\.1)\\s*billion']}]},
 {'id': 'fin-04',
  'question': "What does Boeing's balance sheet disclose about its debt levels?",
  'expected_company': 'boeing',
  'expected_section': 'financial_statements',
  'category': 'financial_statements',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'total-debt',
                     'text': 'Total debt $ 53,864',
                     'content_fingerprint': '3206ba37bbb40130b565',
                     'source_path': 'data/sections/boeing/2025/financial_statements.txt',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'total-debt',
                      'description': 'Total debt $53.864B vs $52.307B',
                      'patterns': ['53,864'],
                      'regex_patterns': ['\\$?\\s*53(?:\\.864|\\.9)\\s*billion']},
                     {'id': 'current-debt',
                      'description': 'Short-term/current portion $1.278B',
                      'patterns': ['1,278'],
                      'regex_patterns': ['\\$?\\s*1(?:\\.278|\\.3)\\s*billion']}]},
 {'id': 'fin-05',
  'question': "What do Nvidia's financial statements show for total revenue?",
  'expected_company': 'nvidia',
  'expected_section': 'financial_statements',
  'category': 'financial_statements',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'total-revenue',
                     'text': 'Total revenue $ 130,497 $ 60,922 $ 26,974',
                     'content_fingerprint': 'b83cd0689c87f7b740f5',
                     'source_path': 'data/raw/nvidia/2025_10k.html',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'latest-total-revenue',
                      'description': 'Latest total revenue $130.497B',
                      'patterns': ['130,497'],
                      'regex_patterns': ['\\$?\\s*130(?:\\.497|\\.5)\\s*billion']}]},
 {'id': 'fin-06',
  'question': "What does Pfizer's income statement show regarding net income?",
  'expected_company': 'pfizer',
  'expected_section': 'financial_statements',
  'category': 'financial_statements',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'gold_evidence': [{'id': 'net-income',
                     'text': 'Net income attributable to Pfizer Inc. common shareholders $ 8,031 $ 2,119 $ '
                             '31,372',
                     'content_fingerprint': 'a421aa16f8909070a75e',
                     'source_path': 'data/raw/pfizer/2025_10k.html',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'latest-net-income',
                      'description': '2024 net income attributable to Pfizer common shareholders $8.031B',
                      'patterns': ['8,031'],
                      'regex_patterns': ['\\$?\\s*8(?:\\.031|\\.0)\\s*billion']}]},
 {'id': 'fin-07',
  'question': "What does ExxonMobil's financial statements disclose about capital expenditures?",
  'expected_company': 'exxonmobil',
  'expected_section': 'financial_statements',
  'category': 'financial_statements',
  'answerable': True,
  'gold_fiscal_year': 2025,
  'annotation_note': "The filing distinguishes GAAP cash-flow additions to PP&E from management's Capex and "
                     'Cash Capex measures; gold facts preserve that distinction.',
  'gold_evidence': [{'id': 'ppe-additions',
                     'text': 'Additions to property, plant and equipment ( 24,306 ) ( 21,919 ) ( 18,407 )',
                     'content_fingerprint': '38f7ff8e3be9caa8dcaf',
                     'source_path': 'data/raw/exxonmobil/2025_10k.html',
                     'source_fiscal_year': 2025},
                    {'id': 'capex-reconciliation',
                     'text': 'Capital and Exploration Expenditures (Capex) 27,551 26,325 ExxonMobil’s share '
                             'of Capex for equity companies (2,546) (2,741) Exploration expenses excluding '
                             'prior year dry holes (755) (567) Other activities including finance leases 56 '
                             '(1,098) Additions to property, plant and equipment 24,306 21,919 Additional '
                             'investments and advances 3,299 2,995 Other investing activities including '
                             'collection of advances (1,926) (1,562) Inflows from noncontrolling interests '
                             'for major projects (32) (124) Total Cash Capex (Non-GAAP) 25,647 23,228',
                     'content_fingerprint': 'c4b9aaa6c90ff53d55a6',
                     'source_path': 'data/raw/exxonmobil/2025_10k.html',
                     'source_fiscal_year': 2025}],
  'expected_facts': [{'id': 'ppe-additions',
                      'description': 'Additions to PP&E $24.306B',
                      'patterns': ['24,306'],
                      'regex_patterns': ['\\$?\\s*24(?:\\.306|\\.3)\\s*billion']},
                     {'id': 'capex',
                      'description': 'Capital and Exploration Expenditures $27.551B',
                      'patterns': ['27,551'],
                      'regex_patterns': ['\\$?\\s*27(?:\\.551|\\.6)\\s*billion']},
                     {'id': 'cash-capex',
                      'description': 'Cash Capex $25.647B',
                      'patterns': ['25,647'],
                      'regex_patterns': ['\\$?\\s*25(?:\\.647|\\.6)\\s*billion']}]}]